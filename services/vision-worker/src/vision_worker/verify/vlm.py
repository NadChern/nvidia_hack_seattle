"""Ask a vision-language model what the window actually shows.

Every other stage in this service measures: where is the box, how far away,
did the position change. That works until the object stops being visible --
and the moment it stops being visible is the moment that matters. A hand
covers the keys, the next clean frame shows an empty desk, and no measurement
can recover what happened in between because the frames contain no keys to
measure.

They do contain the answer, though. Looked at by something that understands a
scene rather than a bounding box, the sequence is unambiguous: a hand arrives,
closes, and lifts. This module is where that question gets asked.

Measured on `media/clips` with a 4B model over 17 frames at 4fps:

  - the pickup window: "A hand picks up the keys from the white table",
    resolved to `picked_up`, ~20s
  - a control window where nothing happens: `nothing_happened`, no invented
    event -- which is the test that actually matters
  - a window where only the camera moved: also `nothing_happened`, which is
    the same discrimination the geometry path was built for

**The failure mode this is designed against.** A model asked "what happened to
the keys?" will produce a plausible story whether or not the video shows one.
That is this project's original sin in a new costume: a confidently wrong
explanation instead of a confidently wrong place. So `nothing_happened` and
`unknown` are first-class answers the schema makes easy to give, the model is
told in the prompt that finding nothing is a correct outcome, and anything
short of a clear reading becomes `unverified` -- retained as a diagnostic,
never written to memory.

Talks to any OpenAI/Ollama-compatible chat endpoint over HTTP. No torch, no
model runtime in-process: the model runs wherever it runs, which is what lets
this service stay deployable on a machine that cannot host one.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import (
    MEMORY_ACTIONS,
    CandidateAction,
    CandidateEvent,
    DetectorRef,
    TrackSample,
    VerifierResult,
)

logger = logging.getLogger(__name__)

#: Bumped whenever the prompt or schema changes in a way that could move a
#: verdict. Carried on every result, per docs/06, so an evaluation run can
#: say which wording produced it.
PROMPT_VERSION = "vlm-v1"

CONFIRMED = "vlm_confirms"
#: The model looked and found no event -- the answer that keeps a false
#: pickup out of memory.
NOTHING_HAPPENED = "vlm_saw_no_event"
#: The model contradicted the claim with a different event.
CONTRADICTED = "vlm_reports_different_event"
#: Everything inconclusive collapses here: an unparseable reply, a model that
#: said it could not tell, a transport failure. All identical downstream --
#: retained, never promoted.
UNCERTAIN = "vlm_uncertain"
UNREACHABLE = "vlm_unreachable"
MALFORMED = "vlm_response_malformed"

#: The model's own vocabulary. Everything except the last two maps directly
#: onto a `CandidateAction`; those two are how it declines.
_ACTIONS = ["picked_up", "placed", "carried", "observed", "nothing_happened", "unknown"]

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "action": {"type": "string", "enum": _ACTIONS},
        "location_description": {"type": "string"},
        "certain": {"type": "boolean"},
    },
    "required": ["answer", "action", "certain"],
}

_PROMPT = """These frames are consecutive moments from one continuous video, in \
order, recorded from a camera worn on someone's head.

The object of interest is: {label}.

Report ONLY what these frames actually show happening to it.

If the {label} is simply sitting somewhere and nobody interacts with it, answer \
with action "nothing_happened". That is a correct and expected answer, not a \
failure -- most of the time nothing is happening, and saying so is the right \
call. Do not invent an event that is not visible.

If you genuinely cannot tell, answer "unknown" and set certain to false. \
Guessing is worse than admitting uncertainty here.

Also describe where the {label} is, in terms a person would recognise -- the \
surface it is on and the things around it."""


@dataclass(frozen=True, slots=True)
class VlmVerifierConfig:
    """Where the model lives, and how much of the window it sees."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:4b"
    #: Frames per window handed to the model. Each costs roughly 550 tokens,
    #: so this and `num_ctx` move together -- 17 frames overflowed a default
    #: 4096-token window during the first real run.
    max_frames: int = 16
    #: Cosmos and most video-tuned models are trained around 4fps. Sampling
    #: the window down to roughly that rate costs nothing and matches what
    #: they expect.
    target_fps: float = 4.0
    num_ctx: int = 16384
    timeout_s: float = 180.0


class VlmVerifier:
    """Judges a candidate by asking a model what the window shows."""

    def __init__(self, config: VlmVerifierConfig | None = None) -> None:
        self._config = config or VlmVerifierConfig()
        self._ref = DetectorRef(name="vlm", checkpoint=self._config.model, revision=PROMPT_VERSION)

    @property
    def config(self) -> VlmVerifierConfig:
        return self._config

    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: Sequence[bytes],
        samples: Sequence[TrackSample] = (),
        decoded: Sequence[NDArray[np.uint8]] = (),
    ) -> VerifierResult:
        # `frames` are already JPEG bytes from the evidence ring, so they go
        # to the model as-is. Touching `decoded` would re-encode what we
        # already have, and pay to decode frames nothing else reads.
        del samples, decoded
        started = time.perf_counter()

        if not frames:
            return self._result(candidate, "unverified", UNCERTAIN, started)

        selected = _subsample(frames, self._config.max_frames)
        try:
            reply = await asyncio.to_thread(self._ask_blocking, candidate.label, selected)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "VLM unreachable; candidate left unverified",
                extra={"candidate_id": candidate.candidate_id, "error": str(exc)},
            )
            return self._result(candidate, "unverified", UNREACHABLE, started)

        parsed = _parse(reply)
        if parsed is None:
            logger.warning(
                "VLM reply was not valid JSON for the requested schema",
                extra={"candidate_id": candidate.candidate_id, "reply": reply[:200]},
            )
            return self._result(candidate, "unverified", MALFORMED, started)

        return self._judge(candidate, parsed, started)

    def _judge(
        self, candidate: CandidateEvent, parsed: dict[str, Any], started: float
    ) -> VerifierResult:
        reported = str(parsed.get("action", "unknown"))
        certain = bool(parsed.get("certain", False))
        description = parsed.get("location_description") or None
        answer = str(parsed.get("answer", ""))[:400]

        logger.info(
            "VLM verdict",
            extra={
                "candidate_id": candidate.candidate_id,
                "claimed": candidate.action,
                "reported": reported,
                "certain": certain,
                "answer": answer,
            },
        )

        if reported == "unknown" or not certain:
            # It looked and could not tell. Nothing is written, and the
            # object keeps whatever state it already had.
            return self._result(candidate, "unverified", UNCERTAIN, started, description)

        if reported == "nothing_happened":
            # For a claim, this is a contradiction: the pipeline thought
            # something happened and the video disagrees. For a `vanished`
            # question, it means the object is still where it was -- also a
            # rejection, and the correct one: nothing to record.
            return self._result(candidate, "rejected", NOTHING_HAPPENED, started, description)

        if reported not in MEMORY_ACTIONS:  # pragma: no cover -- schema-constrained
            return self._result(candidate, "unverified", MALFORMED, started, description)

        resolved: CandidateAction = reported  # type: ignore[assignment]
        if candidate.action == "vanished":
            # A question, now answered. The resolved action is what actually
            # becomes the observation -- see `emit/memory.py`.
            return self._result(
                candidate, "confirmed", CONFIRMED, started, description, resolved=resolved
            )

        if resolved != candidate.action:
            # The pipeline claimed one thing and the video shows another.
            # Confirm what was actually seen rather than what was guessed --
            # the model looked at the frames and the state machine did not.
            return self._result(
                candidate, "confirmed", CONTRADICTED, started, description, resolved=resolved
            )

        return self._result(candidate, "confirmed", CONFIRMED, started, description)

    def _result(
        self,
        candidate: CandidateEvent,
        outcome: str,
        reason_code: str,
        started: float,
        description: str | None = None,
        resolved: CandidateAction | None = None,
    ) -> VerifierResult:
        return VerifierResult(
            candidate_id=candidate.candidate_id,
            outcome=outcome,  # type: ignore[arg-type]
            reason_code=reason_code,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            verifier=self._ref,
            prompt_version=PROMPT_VERSION,
            occurred_at=dt.datetime.now(dt.UTC),
            resolved_action=resolved,
            description=description,
        )

    # ------ Blocking work, always run off the event loop -------------------

    def _ask_blocking(self, label: str, frames: Sequence[bytes]) -> str:
        payload = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": _PROMPT.format(label=label or "the object"),
                    "images": [base64.b64encode(f).decode() for f in frames],
                }
            ],
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self._config.num_ctx},
        }
        request = urllib.request.Request(
            f"{self._config.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
            body = json.load(response)
        return str(body.get("message", {}).get("content", ""))


def _subsample(frames: Sequence[bytes], limit: int) -> Sequence[bytes]:
    """Thin a window evenly to at most `limit` frames, keeping both ends.

    The endpoints carry the before and after that make a disappearance
    legible; dropping either leaves the model guessing at the very thing it
    was asked about.
    """
    if len(frames) <= limit:
        return frames
    positions = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[index] for index in dict.fromkeys(int(p) for p in positions)]


def _parse(reply: str) -> dict[str, Any] | None:
    """Parse the model's reply, tolerating a reasoning preamble.

    Structured output and chain-of-thought have a documented history of
    fighting: a reasoning model may narrate before the JSON despite a schema.
    Rather than fail the candidate over formatting, take the last balanced
    object in the reply -- and if there isn't one, say `unverified` instead of
    guessing at intent.
    """
    reply = reply.strip()
    if not reply:
        return None
    parsed: object
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        start, end = reply.find("{"), reply.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(reply[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    # json.loads is typed as returning Any; the isinstance narrows it to a
    # dict but leaves the key/value types unknown. Cast once here rather than
    # ignoring the same complaint at every read site.
    return cast("dict[str, Any]", parsed)


__all__ = [
    "CONFIRMED",
    "CONTRADICTED",
    "MALFORMED",
    "NOTHING_HAPPENED",
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA",
    "UNCERTAIN",
    "UNREACHABLE",
    "VlmVerifier",
    "VlmVerifierConfig",
]
