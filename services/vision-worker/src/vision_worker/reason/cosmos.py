"""Ask Cosmos 3 Nano what a short window shows, and where.

Cosmos runs behind vLLM's OpenAI-compatible API (`/v1/chat/completions`), so
this talks plain HTTP with no torch in-process -- the model lives wherever it
lives, which is what lets this service stay deployable on a machine that cannot
host one. Same discipline as the verifier this replaces: `nothing_happened` and
`unknown` are first-class answers, the model is told finding nothing is
correct, and anything unparseable becomes an empty result rather than an
invented event.

**Two hard-won call constraints (see the `cosmos-grounding-constraints` note):**

1. **Never send `response_format: json_schema`.** vLLM guided decoding wrecks box
   accuracy -- a keys box measured IoU 0.05-0.16 under a forced schema versus
   0.55 free-form. So boxes come back in Cosmos's native grounding format
   `<ref>label</ref><box>[x_min, y_min, x_max, y_max]</box>` and are parsed with
   a regex; only the non-spatial action/location comes back as a free-form JSON
   tail, parsed tolerantly.
2. **Coordinates are 0-1000, top-left origin, xyxy.** Cosmos self-reports this
   and it checks out.

Warm latency is ~5s for a single image and more for a multi-frame window, so the
pipeline drives this on a slow, roughly non-overlapping cadence, off the frame
loop -- exactly as the verifier ran.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from visual_memory_vision_contract.protocol import BoundingBox, DetectorRef

from vision_worker.reason.base import WindowEvent

logger = logging.getLogger(__name__)

#: Bumped whenever the prompt changes in a way that could move an answer, so a
#: recorded run can say which wording produced it -- carried on the DetectorRef.
PROMPT_VERSION = "cosmos-reason-v1"

#: The model's own action vocabulary. The first three are memory events; the
#: last two are how it declines. `carried` covers an object in transit.
_ACTIONS = ("placed", "picked_up", "carried", "nothing_happened", "unknown")

#: `<ref>keys</ref><box>[370, 581, 520, 781]</box>` -- Cosmos's native grounding
#: format. The label and the bracketed numbers are captured separately; the
#: numbers are pulled out with `_NUMBERS` so stray nesting or spacing is fine.
_GROUNDING = re.compile(r"<ref>(?P<label>.*?)</ref>\s*<box>(?P<box>.*?)</box>", re.DOTALL)
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")

_PROMPT = """These {count} frames are consecutive moments from one continuous video, in \
order, recorded from a camera worn on someone's head.

Look only for these objects: {labels}.

For every one of those objects that is visible in the LAST frame, output exactly one \
grounding tag on its own line, using coordinates from 0 to 1000 with the origin at the \
top-left of the LAST frame:
<ref>LABEL</ref><box>[x_min, y_min, x_max, y_max]</box>

Then, after the tags, output a single JSON array. One entry per object you tagged:
[{{"label": "LABEL", "action": "ACTION", "location": "a short phrase a person would \
recognise, e.g. on the kitchen table next to a mug"}}]

ACTION must be one of: placed, picked_up, carried, nothing_happened, unknown.
Use "placed" only if the object is set down and left at rest during these frames, \
"picked_up" if a hand lifts it, "carried" if it is moving with a person, and \
"nothing_happened" if it just sits there untouched -- that is a correct and expected \
answer, not a failure. Report only objects you actually see. Do not invent anything."""


@dataclass(frozen=True, slots=True)
class CosmosReasonerConfig:
    """Where the model lives, and how much of the window it sees."""

    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "nvidia/Cosmos3-Nano"
    #: Frames sent per window. Kept small: each image is hundreds of tokens
    #: against an 8192 context, and more frames means more latency on a call
    #: that is already ~5s+. 3-4 is enough to read a placement.
    max_frames: int = 4
    #: A default event confidence -- Cosmos gives no numeric score, and the
    #: memory contract needs one for `confidence.event`. Identity confidence is
    #: computed separately from the C-RADIOv4 cosine downstream.
    event_confidence: float = 0.85
    max_tokens: int = 320
    timeout_s: float = 120.0


class CosmosReasoner:
    """A `WindowReasoner` backed by Cosmos 3 Nano over vLLM's OpenAI API."""

    def __init__(self, config: CosmosReasonerConfig | None = None) -> None:
        self._config = config or CosmosReasonerConfig()
        self._ref = DetectorRef(
            name="cosmos", checkpoint=self._config.model, revision=PROMPT_VERSION
        )

    @property
    def config(self) -> CosmosReasonerConfig:
        return self._config

    @property
    def ref(self) -> DetectorRef:
        return self._ref

    async def analyze(
        self, frames: Sequence[bytes], *, labels: Sequence[str]
    ) -> Sequence[WindowEvent]:
        if not frames or not labels:
            return ()
        selected = _subsample(frames, self._config.max_frames)
        try:
            reply = await asyncio.to_thread(self._ask_blocking, selected, labels)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Cosmos unreachable; window yields no events", extra={"error": str(exc)})
            return ()
        return self._parse(reply, labels)

    async def localize(self, frame: bytes, label: str) -> BoundingBox | None:
        """Registration helper: the box of `label` in a single frame, or None.

        Reuses the exact grounding path the pipeline uses, so an enrolled crop
        is framed the same way a query crop will be -- the crop-parity the
        identity match depends on.
        """
        events = await self.analyze((frame,), labels=(label,))
        for event in events:
            if event.label == label:
                return event.box
        return events[0].box if events else None

    # ------ Parsing --------------------------------------------------------

    def _parse(self, reply: str, labels: Sequence[str]) -> tuple[WindowEvent, ...]:
        boxes = _parse_boxes(reply)
        if not boxes:
            return ()
        actions = _parse_action_tail(reply)
        canonical = {label.lower(): label for label in labels}
        events: list[WindowEvent] = []
        for raw_label, box in boxes:
            label = canonical.get(raw_label.strip().lower())
            if label is None:
                # A box for something we did not ask about -- ignore it rather
                # than promote a class the registry has no gallery for.
                continue
            entry = actions.get(raw_label.strip().lower(), {})
            action = str(entry.get("action", "unknown")).strip().lower()
            if action not in _ACTIONS:
                action = "unknown"
            location = entry.get("location")
            events.append(
                WindowEvent(
                    label=label,
                    box=box,
                    action=action,
                    location_description=str(location) if location else None,
                    confidence=self._config.event_confidence,
                )
            )
        return tuple(events)

    # ------ Blocking work, always run off the event loop -------------------

    def _ask_blocking(self, frames: Sequence[bytes], labels: Sequence[str]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _PROMPT.format(count=len(frames), labels=", ".join(labels)),
            }
        ]
        for frame in frames:
            encoded = base64.b64encode(frame).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            )
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": self._config.max_tokens,
        }
        request = urllib.request.Request(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
            body = json.load(response)
        logger.info(
            "Cosmos window analyzed",
            extra={"latency_ms": (time.perf_counter() - started) * 1000.0, "labels": list(labels)},
        )
        choices = body.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))


def _subsample(frames: Sequence[bytes], limit: int) -> Sequence[bytes]:
    """Thin a window evenly to at most `limit` frames, keeping both ends.

    The endpoints carry the before and after that make a placement legible, so
    the last frame -- the one whose coordinates the boxes are in, and the one
    the pipeline crops -- is always included.
    """
    if len(frames) <= limit:
        return frames
    positions = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[index] for index in dict.fromkeys(int(p) for p in positions)]


def _parse_boxes(reply: str) -> list[tuple[str, BoundingBox]]:
    """Every `<ref>label</ref><box>[...]</box>` in the reply as (label, box).

    Coordinates are 0-1000 xyxy top-left; normalized to 0..1 here, clamped, and
    reordered so min<max. Degenerate (zero-area) boxes are dropped -- a crop of
    nothing helps no one.
    """
    boxes: list[tuple[str, BoundingBox]] = []
    for match in _GROUNDING.finditer(reply):
        numbers = _NUMBERS.findall(match.group("box"))
        if len(numbers) < 4:
            continue
        x0, y0, x1, y1 = (float(n) / 1000.0 for n in numbers[:4])
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))
        x_min, y_min = max(0.0, x_min), max(0.0, y_min)
        x_max, y_max = min(1.0, x_max), min(1.0, y_max)
        if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
            continue
        boxes.append(
            (
                match.group("label"),
                BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )
        )
    return boxes


def _parse_action_tail(reply: str) -> dict[str, dict[str, Any]]:
    """The JSON array of {label, action, location} keyed by lowercased label.

    Tolerant of a reasoning preamble and of the grounding tags before it: takes
    the last balanced `[...]` in the reply. A missing or malformed tail yields
    an empty map, so every object falls back to `unknown` -- no invented events.
    """
    start, end = reply.rfind("["), reply.rfind("]")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in cast("list[Any]", parsed):
        if isinstance(item, dict) and "label" in item:
            out[str(item["label"]).strip().lower()] = cast("dict[str, Any]", item)
    return out


__all__ = ["PROMPT_VERSION", "CosmosReasoner", "CosmosReasonerConfig"]
