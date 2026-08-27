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

from vision_worker.reason.base import LocalizedFrame, WindowEvent

logger = logging.getLogger(__name__)

#: Bumped whenever the prompt changes in a way that could move an answer, so a
#: recorded run can say which wording produced it -- carried on the DetectorRef.
#: v2 states the box extent (`_EXTENT_RULE`, spike 12c).
PROMPT_VERSION = "cosmos-reason-v2"

#: Registration uses a separate single-frame prompt. Event reasoning asks what
#: happened across time, while enrollment must abstain unless the crop contains
#: the physical target itself. Keeping the versions separate avoids pretending
#: an enrollment-only safety change moved event semantics. The -v4/-v2 bumps add
#: the extent rule and stop excluding the attached parts it now includes.
LOCALIZE_PROMPT_VERSION = "cosmos-enrollment-localize-v4"
TEMPORAL_LOCALIZE_PROMPT_VERSION = "cosmos-enrollment-temporal-v2"
REFERENCE_PROMPT_VERSION = "cosmos-enrollment-reference-v1"

#: The extent rule (spike 12c). Naming what to include takes the worst noun from
#: grounding IoU 0.12 -> 0.92 and kills a temperature-0 bimodal swing: containment
#: was always ~100%, but without this the model silently boxes the single most
#: recognizable *part* -- a keyring's bare metal keys -- and crops away the ring,
#: fob and lanyard that tell one instance from another (spike 3c: identity F1
#: 0.776 -> 0.939 on the same images and backbone). The closing clause guards the
#: failure that held an earlier attempt: "box the whole object" was once read as
#: "box the whole frame", which re-admits the keyboard the validity rule rejects.
#: Label-agnostic on purpose, so it drops into the multi-label event prompt too.
_EXTENT_RULE = (
    "Draw each box around the whole object a person would point to as theirs, "
    "including the parts permanently attached to it -- a keyring's ring, fob and "
    "lanyard, a mug's handle, or eyeglasses' arms -- not only its single most "
    "recognizable part. Keep the box tight to that object itself: never let it "
    "grow to cover the hand holding it, the surface beneath it, or the whole frame."
)

#: The model's own action vocabulary. The first three are memory events; the
#: last two are how it declines. `carried` covers an object in transit.
_ACTIONS = ("placed", "picked_up", "carried", "nothing_happened", "unknown")

#: `<ref>keys</ref><box>[370, 581, 520, 781]</box>` -- Cosmos's native grounding
#: format. The label and the bracketed numbers are captured separately; the
#: numbers are pulled out with `_NUMBERS` so stray nesting or spacing is fine.
_GROUNDING = re.compile(r"<ref>(?P<label>.*?)</ref>\s*<box>(?P<box>.*?)</box>", re.DOTALL)
_FRAME_GROUNDING = re.compile(
    r"<frame>(?P<frame>\d+)</frame>\s*"
    r"<ref>(?P<label>.*?)</ref>\s*<box>(?P<box>.*?)</box>",
    re.DOTALL,
)
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")

_PROMPT = """These {count} frames are consecutive moments from one continuous video, in \
order, recorded from a camera worn on someone's head.

Look only for these objects: {labels}.

For every one of those objects that is visible in the LAST frame, output exactly one \
grounding tag on its own line, using coordinates from 0 to 1000 with the origin at the \
top-left of the LAST frame:
<ref>LABEL</ref><box>[x_min, y_min, x_max, y_max]</box>

{extent}

Then, after the tags, output a single JSON array. One entry per object you tagged:
[{{"label": "LABEL", "action": "ACTION", "location": "a short phrase a person would \
recognise, e.g. on the kitchen table next to a mug"}}]

ACTION must be one of: placed, picked_up, carried, nothing_happened, unknown.
Use "placed" only if the object is set down and left at rest during these frames, \
"picked_up" if a hand lifts it, "carried" if it is moving with a person, and \
"nothing_happened" if it just sits there untouched -- that is a correct and expected \
answer, not a failure. Report only objects you actually see. Do not invent anything."""

_LOCALIZE_REQUIREMENTS = {
    "keys": """KEYS means portable metal house, apartment, office, or vehicle keys, usually on a \
keyring with a ring, fob, lanyard, or colored charm. Computer keyboard keys, piano keys, keypad \
buttons, laptop parts, and images of keys on a screen are NOT the target and must produce \
NO_OBJECT. A recognizable physical metal key blade must be present for the target to be valid; \
the box itself then covers the whole keyring the wearer carries, ring and fob and lanyard \
included, not the blade alone.""",
    "wallet": "WALLET means the physical wallet body, not a card, phone, or image on a screen.",
    "glasses": "GLASSES means wearable eyeglasses or sunglasses, not glassware or a screen image.",
    "mug": "MUG means the physical drinking vessel body, not a screen image or isolated handle.",
}

_LOCALIZE_PROMPT = """You are validating one reference image for personal-object \
registration. The target label is: {label}.

Target-specific meaning:
{requirements}

Find the physical target object itself. Do not box the holder's hand, an image displayed on a \
monitor, or the surrounding floor, wall, table, or other background in place of the target. If a \
recognizable {label} is not clearly visible, output exactly NO_OBJECT.

{extent}

If a recognizable {label} is clearly visible, output exactly one line and nothing else:
<ref>{label}</ref><box>[x_min, y_min, x_max, y_max]</box>
Coordinates are 0 to 1000, top-left origin."""

_TEMPORAL_LOCALIZE_PROMPT = """These {count} frames are ordered moments from one \
continuous glasses-camera registration capture. The wearer is deliberately presenting and slowly \
rotating one physical personal object labeled: {label}.

Target-specific meaning:
{requirements}

Search across time for the deliberately presented physical target. Temporal continuity matters: \
the same handheld object should persist or rotate across nearby frames. Ignore static background, \
the wearer's hand by itself, laptop keyboards, screen images, and label homonyms. A frame \
with no clearly recognizable physical target is correctly omitted.

{extent}

For every frame where the target itself is clearly recognizable, output one line:
<frame>INDEX</frame><ref>{label}</ref><box>[x_min, y_min, x_max, y_max]</box>

INDEX is the zero-based frame number shown before each image. Coordinates are 0 to 1000 with \
top-left origin. Output exactly NO_OBJECT if no frame qualifies. Output no prose."""

_REFERENCE_REQUIREMENTS = {
    "keys": """VALID requires a clearly visible metal key blade with its shaft and cut \
teeth or grooves. A ring, fob, tag, tracker, cord, lanyard, or colored accessory without a \
clearly visible metal key blade is REJECT.""",
    "wallet": """VALID requires the wallet body to be clearly visible. A hand, card, phone, \
or surrounding surface without the wallet body is REJECT.""",
    "glasses": """VALID requires recognizable lenses and frame. A case, strap, hand, or \
reflection without the glasses themselves is REJECT.""",
    "mug": """VALID requires the mug body or its body and handle. A hand, coaster, table, \
or isolated handle without the mug body is REJECT.""",
}
_REFERENCE_PROMPT = """Act as a strict quality-control inspector, not an object detector. \
Decide whether this image is suitable as a personal reference photo for {label}. Reply with \
exactly VALID or REJECT.

{requirements}

Blur, severe occlusion, or a tiny partial target is REJECT. When uncertain, REJECT."""


def _localize_requirements(label: str) -> str:
    return _LOCALIZE_REQUIREMENTS.get(
        label.strip().casefold(),
        f"The target is the physical {label} itself, not text, a related object, "
        "or an image displayed on a screen.",
    )


def _localize_prompt(label: str) -> str:
    return _LOCALIZE_PROMPT.format(
        label=label, requirements=_localize_requirements(label), extent=_EXTENT_RULE
    )


def _temporal_localize_prompt(label: str, count: int) -> str:
    return _TEMPORAL_LOCALIZE_PROMPT.format(
        label=label,
        count=count,
        requirements=_localize_requirements(label),
        extent=_EXTENT_RULE,
    )


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

    async def localize_sequence(
        self, frames: Sequence[bytes], label: str
    ) -> Sequence[LocalizedFrame]:
        """Search a short ordered sequence before expensive frame grounding."""
        if not frames or not label:
            return ()
        try:
            reply = await asyncio.to_thread(self._ask_temporal_localize_blocking, frames, label)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "Cosmos unreachable; temporal enrollment search yields no frames",
                extra={"error": str(exc)},
            )
            return ()
        wanted = label.strip().casefold()
        return tuple(
            LocalizedFrame(index=index, box=box)
            for index, raw_label, box in _parse_temporal_boxes(reply)
            if index < len(frames) and raw_label.strip().casefold() == wanted
        )

    async def localize(self, frame: bytes, label: str) -> BoundingBox | None:
        """Return one strict enrollment box, abstaining on accessories/background.

        The temporal event prompt is intentionally not reused here. A single
        enrollment frame has no action to classify, and asking for one made
        Cosmos confidently box a key cord or the floor beneath it. The dedicated
        prompt makes `NO_OBJECT` the required result when the physical target is
        not visually recognizable.
        """
        if not frame or not label:
            return None
        try:
            reply = await asyncio.to_thread(self._ask_localize_blocking, frame, label)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "Cosmos unreachable; enrollment frame yields no box",
                extra={"error": str(exc)},
            )
            return None
        wanted = label.strip().casefold()
        for raw_label, box in _parse_boxes(reply):
            if raw_label.strip().casefold() == wanted:
                return box
        return None

    async def validate_reference(self, crop: bytes, label: str) -> bool:
        """Accept only a crop where the target itself is clearly recognizable."""
        if not crop or not label:
            return False
        try:
            reply = await asyncio.to_thread(self._ask_reference_blocking, crop, label)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                "Cosmos unreachable; enrollment reference rejected",
                extra={"error": str(exc)},
            )
            return False
        return reply.strip().upper() == "VALID"

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

    def _ask_temporal_localize_blocking(self, frames: Sequence[bytes], label: str) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _temporal_localize_prompt(label, len(frames)),
            }
        ]
        for index, frame in enumerate(frames):
            content.extend(
                (
                    {"type": "text", "text": f"FRAME {index}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(frame).decode()}"
                        },
                    },
                )
            )
        started = time.perf_counter()
        reply = self._complete_blocking(content, max_tokens=256)
        logger.info(
            "Cosmos enrollment sequence searched",
            extra={
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "label": label,
                "frames": len(frames),
                "prompt_version": TEMPORAL_LOCALIZE_PROMPT_VERSION,
                "found": len(_parse_temporal_boxes(reply)),
            },
        )
        return reply

    def _ask_localize_blocking(self, frame: bytes, label: str) -> str:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": _localize_prompt(label)},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(frame).decode()}"},
            },
        ]
        started = time.perf_counter()
        reply = self._complete_blocking(content, max_tokens=96)
        logger.info(
            "Cosmos enrollment frame localized",
            extra={
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "label": label,
                "prompt_version": LOCALIZE_PROMPT_VERSION,
                "found": bool(_parse_boxes(reply)),
            },
        )
        return reply

    def _ask_reference_blocking(self, crop: bytes, label: str) -> str:
        normalized = label.strip().casefold()
        requirements = _REFERENCE_REQUIREMENTS.get(
            normalized,
            f"VALID requires the physical {label} itself to be clearly recognizable. "
            "An accessory, hand, or background without the target is REJECT.",
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _REFERENCE_PROMPT.format(label=label, requirements=requirements),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(crop).decode()}"},
            },
        ]
        started = time.perf_counter()
        reply = self._complete_blocking(content, max_tokens=16)
        logger.info(
            "Cosmos enrollment reference validated",
            extra={
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "label": label,
                "prompt_version": REFERENCE_PROMPT_VERSION,
                "valid": reply.strip().upper() == "VALID",
            },
        )
        return reply

    def _ask_blocking(self, frames: Sequence[bytes], labels: Sequence[str]) -> str:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _PROMPT.format(
                    count=len(frames), labels=", ".join(labels), extent=_EXTENT_RULE
                ),
            }
        ]
        for frame in frames:
            encoded = base64.b64encode(frame).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
            )
        started = time.perf_counter()
        reply = self._complete_blocking(content, max_tokens=self._config.max_tokens)
        logger.info(
            "Cosmos window analyzed",
            extra={"latency_ms": (time.perf_counter() - started) * 1000.0, "labels": list(labels)},
        )
        return reply

    def _complete_blocking(self, content: Sequence[dict[str, Any]], *, max_tokens: int) -> str:
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": list(content)}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self._config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
            body = json.load(response)
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


def _parse_temporal_boxes(
    reply: str,
) -> list[tuple[int, str, BoundingBox]]:
    """Parse frame-indexed enrollment grounding without inventing indices."""
    found: list[tuple[int, str, BoundingBox]] = []
    for match in _FRAME_GROUNDING.finditer(reply):
        box = _normalized_box(match.group("box"))
        if box is not None:
            found.append((int(match.group("frame")), match.group("label"), box))
    return found


def _parse_boxes(reply: str) -> list[tuple[str, BoundingBox]]:
    """Every `<ref>label</ref><box>[...]</box>` in the reply as (label, box).

    Coordinates are 0-1000 xyxy top-left; normalized to 0..1 here, clamped, and
    reordered so min<max. Degenerate (zero-area) boxes are dropped -- a crop of
    nothing helps no one.
    """
    boxes: list[tuple[str, BoundingBox]] = []
    for match in _GROUNDING.finditer(reply):
        box = _normalized_box(match.group("box"))
        if box is not None:
            boxes.append((match.group("label"), box))
    return boxes


def _normalized_box(raw: str) -> BoundingBox | None:
    numbers = _NUMBERS.findall(raw)
    if len(numbers) < 4:
        return None
    x0, y0, x1, y1 = (float(n) / 1000.0 for n in numbers[:4])
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(1.0, x_max), min(1.0, y_max)
    if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


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
            typed_item = cast("dict[str, Any]", item)
            out[str(typed_item["label"]).strip().lower()] = typed_item
    return out


__all__ = [
    "LOCALIZE_PROMPT_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_PROMPT_VERSION",
    "TEMPORAL_LOCALIZE_PROMPT_VERSION",
    "CosmosReasoner",
    "CosmosReasonerConfig",
]
