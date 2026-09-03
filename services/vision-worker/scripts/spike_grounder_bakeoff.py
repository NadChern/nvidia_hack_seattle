#!/usr/bin/env python3
"""Which open-source grounder should ship? (Grounder bake-off, 2026-09)

The hackathon is over and this is now a customer-discovery question, so the
answer has to survive being shown to someone: **the winner must be a model we
are allowed to ship** under the privacy-first, open-source story, and it must be
fast enough for realtime video.

"Open source" is applied as **open weights we may actually ship**, which is not
the same as OSI-approved:

| Arm | Licence | Shippable |
|---|---|---|
| Qwen3-VL-4B / 8B | Apache-2.0 | yes |
| MOSS-VL-Realtime | Apache-2.0 | yes |
| Cosmos3-Nano | NVIDIA Open Model License | incumbent, permissive-ish |
| LFM2.5-VL-3B | LFM Open License -- free below $10M revenue | yes, with a cap |

LFM2.5-VL-3B carries a commercial-use cap above $10M annual revenue and is still
a full contender (owner decision, 2026-09-02): the threshold is distant, and a
grounder is a replaceable component whose switching cost is a bake-off re-run.
The cost is to the *claim* rather than the code -- "fully open source" becomes
"open weights, commercially licensed above $10M" if a customer asks.

**Moondream 3 is a different case and stays excluded.** BSL 1.1's "No
Third-Party Service" grant is not a revenue threshold: it forbids offering the
model as a service at *any* size, which is precisely what this product is. No
amount of growth makes that one legal, so it is not scored at all.

## Two tasks, because the reasoner does two things

The pipeline asks one model both *where is it* (`--task grounding`) and *what
just happened to it* (`--task events`), and an arm can be excellent at one and
useless at the other.

### Grounding axes

Grounding IoU alone picked the wrong model once already (spike 3c: containment
was ~100% while identity F1 sat at 0.776, because the box was on the right
object but the wrong *extent*). So every arm reports:

1. **IoU** against the 36 human-annotated boxes in `clips/spike1-arms/_cache`.
2. **Containment + area ratio** -- separates "wrong object" from "right object,
   wrong extent". The first is a grounding failure, the second a prompt fix.
3. **Latency** per call. This is the axis "realtime" actually lives on: the
   pipeline fires a window every `reason_interval_seconds` = 7 s, so an arm
   slower than that backs the queue up no matter how well it grounds.
4. **No-box rate.** A model that silently returns nothing is worse than one that
   returns a loose box, because the pipeline reads it as "object not present".

### Event axes

`placed` / `picked_up` / `carried` / `nothing_happened` across a 20 s window, on
the recordings annotated by `annotate_placement.py`. Only `placed` reaches
memory today, because `promote_motion_events` is False on the impression that
"at ~1fps Cosmos hallucinates handling on a resting object" -- an impression
nobody has ever measured. The headline numbers are **per-event placement
recall**, **false-placement rate** on quiet windows (a false placement writes a
wrong location, which is worse for the product than missing one), and
**eight-frame latency**, which is the realtime figure that actually binds.

See the long comment above `ClipFrame` for why the axis is shaped this way, and
RESULTS.md for the gates, which were fixed before any number existed.

## Prompt parity

Every arm gets the **production** enrollment-localize prompt, extent rule
included, rather than a prompt tuned per model. Tuning per arm measures prompt
engineering effort, not model quality, and spike 12c showed a single sentence
moves the worst noun 0.12 -> 0.92 -- an effect large enough to swamp the
between-model differences this is trying to find. `--check-prompt-drift` verifies
the inlined copy still matches `reason/cosmos.py`, since this script cannot
import it (dependency isolation, below).

## Why arms run one at a time, in isolated environments

LFM2.5-VL needs `transformers>=5.0`; the service pins 4.57.6 for C-RADIOv4. They
cannot share an interpreter -- the same split that forced every grounding spike
to run isolated. The HTTP arms sidestep this entirely by talking to a server
over the OpenAI chat shape, which is also how the real pipeline calls its
reasoner, so their latency numbers include the serialization the deployment
actually pays.

Serving commands per arm are in `docs/spikes/grounder-bakeoff/README.md`.

Usage::

    # HTTP arms -- start the server first (see the README), then:
    uv run python scripts/spike_grounder_bakeoff.py --arm qwen3-vl-8b \\
        --dataset ../../clips/identity-probe --cache ../../clips/spike1-arms/_cache \\
        --out ../../docs/spikes/grounder-bakeoff/runs/qwen3-vl-8b.json

    # the event axis, once the recordings are annotated
    uv run python scripts/spike_grounder_bakeoff.py --arm qwen3-vl-8b --task events \\
        --out ../../docs/spikes/grounder-bakeoff/runs/qwen3-vl-8b.events.json

    # every arm is an HTTP arm; only the serve command differs (see the README)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# --- Prompt, copied from the production reasoner -----------------------------
# Verbatim from services/vision-worker/src/vision_worker/reason/cosmos.py
# (_EXTENT_RULE, _LOCALIZE_PROMPT, _LOCALIZE_REQUIREMENTS). Copied rather than
# imported because the ceiling arm runs under `uv run --isolated` with a
# different transformers major and cannot see the service package at all.
# `--check-prompt-drift` re-reads the real file when it is reachable, so this
# copy going stale is a loud failure rather than a silently invalid comparison.

_EXTENT_RULE = (
    "Draw each box around the whole object a person would point to as theirs, "
    "including the parts permanently attached to it -- a keyring's ring, fob and "
    "lanyard, a mug's handle, or eyeglasses' arms -- not only its single most "
    "recognizable part. Keep the box tight to that object itself: never let it "
    "grow to cover the hand holding it, the surface beneath it, or the whole frame."
)

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


#: The reasoner's action vocabulary, verbatim from `reason/cosmos.py::_ACTIONS`.
#: The first three become memory writes; the last two are how the model
#: declines. `nothing_happened` is a correct answer, not a failure, and the
#: event axis is largely a measurement of whether an arm can bring itself to
#: say it.
ACTIONS = ("placed", "picked_up", "carried", "nothing_happened", "unknown")

#: Only `placed` reaches memory today: `promote_motion_events` is False because
#: "at ~1fps Cosmos hallucinates handling on a resting object, and a single
#: false pickup flips a confirmed placement to 'moved afterward'". Whether that
#: policy can be lifted is one of the questions this axis answers.
MEMORY_ACTIONS = ("placed", "picked_up", "carried")

#: Verbatim from `reason/cosmos.py::_PROMPT` -- the window prompt the running
#: pipeline sends. Copied for the same reason the localize prompt is, and
#: checked by `--check-prompt-drift` for the same reason.
_WINDOW_PROMPT = """These {count} frames are consecutive moments from one continuous video, in \
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


def window_prompt(labels: list[str], count: int) -> str:
    return _WINDOW_PROMPT.format(count=count, labels=", ".join(labels), extent=_EXTENT_RULE)


def localize_prompt(label: str) -> str:
    requirements = _LOCALIZE_REQUIREMENTS.get(
        label, f"{label.upper()} means the physical object itself, not a screen image of one."
    )
    return _LOCALIZE_PROMPT.format(label=label, requirements=requirements, extent=_EXTENT_RULE)


def check_prompt_drift(repo_root: Path) -> str:
    """Compare every inlined prompt against the production one.

    Covers the extent rule (both axes), the window prompt (event axis) and the
    action vocabulary, because an event score against a stale prompt is worse
    than no event score: it looks like a model result and is a diff.

    Returns a status string rather than raising: on the rented box the service
    source may not be checked out beside the harness, and "could not check" must
    not read the same as "checked and matched".
    """
    source = repo_root / "services/vision-worker/src/vision_worker/reason/cosmos.py"
    if not source.is_file():
        return "unchecked (reason/cosmos.py not found)"
    text = source.read_text(encoding="utf-8")
    # The rule is a parenthesised implicit-concatenation of string literals, so
    # compare on collapsed whitespace and stripped quotes rather than exact text.
    match = re.search(r"_EXTENT_RULE\s*=\s*\((?P<body>.*?)\)\n", text, re.DOTALL)
    if not match:
        return "unchecked (_EXTENT_RULE not parseable)"
    literal = "".join(re.findall(r'"([^"]*)"', match.group("body")))
    drifted: list[str] = []
    if _collapse(literal) != _collapse(_EXTENT_RULE):
        drifted.append("_EXTENT_RULE")

    window = re.search(r'_PROMPT = """(?P<body>.*?)"""', text, re.DOTALL)
    if window is None:
        return "unchecked (_PROMPT not parseable)"
    if _collapse(window.group("body").replace("\\\n", "")) != _collapse(
        _WINDOW_PROMPT.replace("\\\n", "")
    ):
        drifted.append("_PROMPT (window/event)")

    actions = re.search(r"_ACTIONS = \((?P<body>[^)]*)\)", text)
    if actions is None:
        return "unchecked (_ACTIONS not parseable)"
    if tuple(re.findall(r'"([^"]*)"', actions.group("body"))) != ACTIONS:
        drifted.append("_ACTIONS")

    if drifted:
        return f"DRIFTED -- {', '.join(drifted)} no longer matches reason/cosmos.py"
    return "ok (extent rule, window prompt and action vocabulary match reason/cosmos.py)"


def _collapse(text: str) -> str:
    return " ".join(text.split())


# --- The arms ----------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One contender.

    `shippable` is not decoration. It is the field that decides whether a good
    number here can become a product decision, and it is the reason two of the
    five arms exist only to bound the others from above.
    """

    name: str
    model: str
    transport: str  # "openai" (vLLM/SGLang) or "transformers" (local weights)
    licence: str
    shippable: bool
    params_b: float
    #: Serving runtime, which is a deployment cost the score does not show:
    #: an SGLang-only winner means the deployment runs two inference runtimes,
    #: since Nemotron is already on vLLM.
    runtime: str
    notes: str
    #: `auto` scores every reply under both readings and reports which the model
    #: actually uses. Guessing this wrong made Cosmos look far worse than it was
    #: in an earlier spike, so it is measured, never assumed.
    coord_order: str = "auto"
    port: int = 8001


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm(
            name="qwen3-vl-4b",
            model="Qwen/Qwen3-VL-4B-Instruct",
            transport="openai",
            licence="Apache-2.0",
            shippable=True,
            params_b=4.0,
            runtime="vLLM >=0.13 (FP8 native, ~1.9x over BF16)",
            notes="Speed arm. If it clears the IoU gate it is almost certainly the pick: "
            "smallest weights, cheapest card, cleanest licence.",
        ),
        Arm(
            name="qwen3-vl-8b",
            model="Qwen/Qwen3-VL-8B-Instruct",
            transport="openai",
            licence="Apache-2.0",
            shippable=True,
            params_b=9.0,
            runtime="vLLM >=0.13",
            notes="Quality arm. Strongest open grounding on RefCOCO/ODinW; "
            "text-timestamp alignment matters for the event axis, not this one.",
        ),
        Arm(
            name="moss-vl-realtime",
            model="OpenMOSS-Team/MOSS-VL-Realtime",
            transport="openai",
            licence="Apache-2.0",
            shippable=True,
            params_b=11.3,
            runtime="SGLang (official); vLLM support not established",
            notes="Streaming-native: cross-attention + XRoPE, keeps watching while it "
            "generates, and ships a WebSocket service taking external JPEG frames -- "
            "the shape media-gateway already emits. Scored here on the same "
            "single-image protocol as everyone else, which under-sells that.",
        ),
        Arm(
            name="cosmos3-nano",
            model="nvidia/Cosmos3-Nano",
            transport="openai",
            licence="NVIDIA Open Model License",
            shippable=True,
            params_b=16.0,
            runtime="vLLM (BF16 only -- no official FP8/FP4 path)",
            notes="The incumbent, and the baseline the others must beat. 32 GiB of BF16 "
            "weights is ~45% of the whole VRAM budget on its own.",
            coord_order="xyxy",  # self-reported and confirmed; see cosmos-grounding-constraints
        ),
        Arm(
            name="lfm2.5-vl-3b",
            model="LiquidAI/LFM2.5-VL-3B",
            # Served, not loaded in-process. LFM2.5-VL is native to vLLM as
            # `Lfm2VlForConditionalGeneration` (no --trust-remote-code), so it
            # can be measured on the same serving path as every other arm.
            # That matters specifically for the latency axis: a
            # transformers `.generate()` against vLLM-served arms would not be
            # a comparison, and latency is what decides realtime.
            transport="openai",
            licence="LFM Open License (free below $10M revenue)",
            # Full contender, by owner decision 2026-09-02. The revenue cap is
            # real but distant, and a grounder is a replaceable component on a
            # roughly six-month cycle -- by the time the cap binds, the arm that
            # replaces this one probably does not exist yet. Switching cost is a
            # bake-off re-run, not a rewrite, which is what makes the lock-in
            # cheap enough to accept.
            #
            # What it does cost is the claim, not the code: "fully open source"
            # becomes "open weights, commercially licensed above $10M" if a
            # customer asks. Revisit when revenue is within one year of the cap.
            shippable=True,
            params_b=3.0,
            runtime="vLLM >=0.23 (LFM2.5-VL landed later than the Qwen arms)",
            notes="Current measured best (IoU 0.889, 0.92 with the extent rule, spike 12c) "
            "and the smallest arm at 6.02 GiB resident -- if it wins, the deployment card "
            "drops with it. Licence caps commercial use above $10M revenue; judged an "
            "acceptable trade rather than a disqualification.",
        ),
    )
}


# --- Transports --------------------------------------------------------------


def encode_data_url(image, long_edge: int) -> str:
    """JPEG at the size the pipeline sends, so latency stays comparable.

    The pipeline hands the reasoner JPEG frames off the evidence ring, not
    original stills, so scoring full-resolution HEICs would measure a call the
    deployment never makes.
    """
    from PIL import Image

    copy = image.copy()
    copy.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(data_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url}}


def ask_openai(
    base_url: str, model: str, content: list[dict], timeout: float, max_tokens: int
) -> str:
    """One chat turn of mixed text and images -- and never `response_format`.

    Content blocks rather than a single image because the event axis sends
    eight frames per call, exactly as `CosmosReasoner._ask_blocking` does. Both
    axes put the **text first**, matching production: block order is part of
    the prompt a VLM sees, so an image-first harness against a text-first
    deployment would not be the parity this bake-off claims.

    Guided decoding against a box schema measured 0.05-0.16 IoU where free-form
    native `<box>` tokens measured 0.55 on the same images. The constraint is
    inherited, not rediscovered; see the cosmos-grounding-constraints note.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            # Local vLLM/SGLang ignore this; it is here so the same code path
            # works against a hosted endpoint during a capability probe.
            "authorization": f"Bearer {os.environ.get('BAKEOFF_API_KEY', 'local')}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"] or ""


class TransformersArm:
    """Local-weights transport, for an arm with no serving path.

    No registered arm currently uses this -- LFM2.5-VL moved to vLLM once its
    native support was confirmed. Kept as the escape hatch for a future
    contender that transformers can load and no server can, which is a
    recurring shape in this space (Moondream 3 is exactly that today).

    Holds the model for the whole run so the reported latency is warm, matching
    the HTTP arms -- a cold first call would otherwise flatter the servers.
    """

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()

    def ask(self, image, prompt: str, max_tokens: int) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = output[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(trimmed, skip_special_tokens=False)


# --- Parsing and metrics -----------------------------------------------------

#: Cosmos's native grounding format, which Qwen3-VL and MOSS-VL are asked for
#: too because the prompt names it explicitly.
_GROUNDING = re.compile(r"<ref>(?P<label>.*?)</ref>\s*<box>(?P<box>.*?)</box>", re.DOTALL)
#: The structured form several models emit instead, whatever the prompt asked.
_BOX_JSON = re.compile(r'"(?:bbox|box|bbox_2d|box_2d)"\s*:\s*\[(?P<box>[^\]]+)\]')
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")
COORD_SCALE = 1000.0


def parse_box(text: str, order: str) -> tuple[float, float, float, float] | None:
    """Native tokens, then structured JSON, then a bare quad anywhere in the reply.

    Deliberately permissive in that order: a model that ignores the requested
    format but grounds correctly should be scored on its grounding, not punished
    for its formatting. `NO_OBJECT` is a real answer and returns None.
    """
    if "NO_OBJECT" in text.upper():
        return None
    candidates = [m.group("box") for m in _GROUNDING.finditer(text)]
    candidates += [m.group("box") for m in _BOX_JSON.finditer(text)]
    if not candidates:
        candidates = [text]
    for blob in candidates:
        numbers = [float(n) for n in _NUMBERS.findall(blob)]
        if len(numbers) < 4:
            continue
        quad = numbers[:4]
        scale = 1.0 if max(quad) <= 1.0 else COORD_SCALE
        a, b, c, d = (v / scale for v in quad)
        x0, y0, x1, y1 = (b, a, d, c) if order == "yxyx" else (a, b, c, d)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return None


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(predicted, truth) -> tuple[float, float]:
    """Fraction of the prediction inside truth, and their area ratio.

    IoU alone conflates two failures needing opposite fixes. A box on the wrong
    object scores low with low containment; a box on the right object at the
    wrong extent -- bare keys instead of the whole keyring -- scores low with
    containment near 1. Only the first is a grounding defect; the second is what
    the extent rule exists to fix, and it cost this project an identity
    regression before anyone measured it separately.
    """
    ix0, iy0 = max(predicted[0], truth[0]), max(predicted[1], truth[1])
    ix1, iy1 = min(predicted[2], truth[2]), min(predicted[3], truth[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_p = max(1e-9, (predicted[2] - predicted[0]) * (predicted[3] - predicted[1]))
    area_t = max(1e-9, (truth[2] - truth[0]) * (truth[3] - truth[1]))
    return inter / area_p, area_p / area_t


def load_rgb(path: Path):
    from PIL import Image, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def gpu_used_mib() -> int | None:
    """Total card usage, not this process's -- one arm is served at a time.

    Reported so the quality table carries its own VRAM cost. It is a coarse
    number by design: the precise per-model accounting is `vram_probe.py`'s job.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()
    return int(first[0]) if first else None


# --- The event axis ----------------------------------------------------------
#
# Grounding is half of what the reasoner does. The other half is saying what
# happened -- `placed` / `picked_up` / `carried` / `nothing_happened` -- and an
# arm can ground beautifully while hallucinating handling on an object that
# never moved. That failure is not hypothetical: `promote_motion_events` is
# False in production precisely because of it, which means today only `placed`
# reaches memory and the movement timeline is thrown away. Nobody has ever
# measured it, so the policy rests on an impression.
#
# What this axis measures, in order of how much it decides:
#
# 1. **Per-event placement recall.** Windows overlap (20 s span, 7 s interval),
#    so one placement is offered to several windows and the pipeline only needs
#    one of them to see it. That is the product's number.
# 2. **False-placement rate** on windows where nothing happened. A false
#    `placed` writes a wrong location into memory, which is the worst outcome
#    the product has -- worse than missing the event, because the user is told
#    something confidently untrue.
# 3. **False-handling rate** -- `picked_up`/`carried` on a resting object. This
#    is the specific number that decides whether `promote_motion_events` can be
#    turned on.
# 4. **Latency per window.** Eight frames per call, not one, so this is the
#    realtime figure that actually binds; the grounding axis's single-image
#    latency flatters every arm.
#
# The location phrase is recorded and never scored. Judging "on the kitchen
# table next to a mug" against an annotation needs a human who watched the
# clip, and a previous attempt at scoring it automatically marked a correct
# phrase as a hallucination because the wide frame really did show that
# surface.


@dataclass(frozen=True)
class ClipFrame:
    index: int
    t: float
    data_url: str


@dataclass(frozen=True)
class Window:
    """One reasoner firing: a span of clip time and the frames sent for it."""

    start: float
    end: float
    frames: list[ClipFrame]


@dataclass
class ClipTruth:
    """One annotated recording, as `annotate_placement.py` writes it."""

    path: Path
    clip: str
    object_id: str
    label: str
    boxes: dict[int, tuple[float, float, float, float]]
    events: list[dict]
    fps: float
    frame_stride: int
    t_start: float
    duration_s: float
    width: int
    height: int
    rotate: int
    notes: str

    @property
    def visible(self) -> tuple[int, int]:
        """The frame range the object is annotated as visible in.

        The annotated box span *is* the visibility span, by convention (stated
        in the annotator and in truth/README.md): the annotator boxes the first
        and last frame the object appears in, and outside that range the object
        is absent. This is what lets "the model returned no box" be scored as
        correct in one window and a miss in another, instead of being a single
        undifferentiated no-box rate.
        """
        keys = sorted(self.boxes)
        return (keys[0], keys[-1]) if keys else (0, -1)

    def is_visible(self, index: int) -> bool:
        low, high = self.visible
        return low <= index <= high

    def box_at(self, index: int) -> tuple[float, float, float, float] | None:
        """The annotated box, or a linear interpolation between neighbours.

        Same rule as `eval_harness.GroundTruth.box_at`, reimplemented rather
        than imported: this harness runs on a rented box under `uv run
        --isolated` and cannot see the service package.
        """
        if index in self.boxes:
            return self.boxes[index]
        keys = sorted(self.boxes)
        before = [k for k in keys if k < index]
        after = [k for k in keys if k > index]
        if not before or not after:
            return None
        lo, hi = before[-1], after[0]
        weight = (index - lo) / (hi - lo)
        a, b = self.boxes[lo], self.boxes[hi]
        return tuple(a[i] + (b[i] - a[i]) * weight for i in range(4))  # type: ignore[return-value]

    def expected(self, start: float, end: float, last_index: int) -> tuple[str, bool]:
        """What the pipeline should record for this window, and whether the
        object is absent from the frame the boxes must be drawn in.

        The second half is not a detail. Boxes are in the **last** frame's
        coordinates, and `CosmosReasoner._parse` returns no events at all when
        the reply carries no box -- so a window whose last frame no longer
        shows the object *cannot* report what happened in it, however clearly
        the earlier frames showed it. The expected answer there is silence, and
        an arm that boxes something anyway has hallucinated.
        """
        if not self.is_visible(last_index):
            return "nothing_happened", True
        return self.action_in(start, end), False

    def action_in(self, start: float, end: float) -> str:
        """What truth says happened in `(start, end]`.

        A span with no annotated event is `nothing_happened`, not "unlabelled":
        the annotator saw the whole clip and marked what occurred, so silence
        is a positive claim that nothing did. When several events land in one
        span the memory-bearing one wins, since that is the one the pipeline
        would have to get right.
        """
        found = {str(event["action"]) for event in self.events if start < float(event["t"]) <= end}
        for action in (*MEMORY_ACTIONS, "unknown"):
            if action in found:
                return action
        return "nothing_happened"


def load_truth(path: Path) -> ClipTruth:
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("clip", "fps", "frame_stride", "width", "height") if k not in raw]
    if missing:
        raise SystemExit(
            f"{path.name} is missing {', '.join(missing)} -- it predates "
            "annotate_placement.py and cannot be placed on a time axis"
        )
    return ClipTruth(
        path=path,
        clip=str(raw["clip"]),
        object_id=str(raw["object_id"]),
        label=str(raw["label"]),
        boxes={int(k): tuple(float(x) for x in v) for k, v in (raw.get("boxes") or {}).items()},
        events=list(raw.get("events") or []),
        fps=float(raw["fps"]),
        frame_stride=int(raw["frame_stride"]),
        t_start=float(raw.get("t_start", 0.0)),
        duration_s=float(raw.get("duration_s", 0.0)),
        width=int(raw["width"]),
        height=int(raw["height"]),
        # Absent means the file predates rotation handling, when everything
        # decoded sideways. Zero reproduces what that annotator saw.
        rotate=int(raw.get("rotate", 0)),
        notes=str(raw.get("notes", "")),
    )


def decode_clip(path: Path, stride: int, long_edge: int, rotate: int) -> list[ClipFrame]:
    """Decode to JPEG data URLs at the stride and rotation the annotation used.

    Same stride, so a frame index in the reply's coordinates is a frame index
    in the truth file with no second mapping to get wrong. JPEG at
    `--long-edge` because that is what the evidence ring hands the reasoner --
    scoring pristine 4K would measure a call the deployment never makes.

    Same rotation because **PyAV ignores rotation metadata**. These recordings
    are shot portrait and carry `rotation=-90`, so every `av.open` path in this
    repository decodes them on their side unless told otherwise -- including
    `media-gateway`'s virtual-glasses `--file` publisher. The annotator records
    what it applied; applying anything else here puts every box 90 degrees out.
    """
    import av
    from PIL import Image

    frames: list[ClipFrame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        time_base = stream.time_base
        for position, decoded in enumerate(container.decode(stream)):
            if position % stride:
                continue
            # Same fallback as the annotator, so a window's span and the
            # annotation it is scored against are on one time axis.
            usable = decoded.pts is not None and time_base is not None
            t = float(decoded.pts * time_base) if usable else position / fps
            image = Image.fromarray(decoded.to_ndarray(format="rgb24"))
            if rotate % 360:
                image = image.transpose(
                    {
                        90: Image.Transpose.ROTATE_270,
                        180: Image.Transpose.ROTATE_180,
                        270: Image.Transpose.ROTATE_90,
                    }[rotate % 360]
                )
            frames.append(ClipFrame(position, t, encode_data_url(image, long_edge)))
    return frames


def subsample(frames: list[ClipFrame], limit: int) -> list[ClipFrame]:
    """Thin evenly, keeping both ends -- `reason/cosmos.py::_subsample`.

    The endpoints carry the before and after that make a placement legible, and
    the last frame is the one the boxes are in.
    """
    if len(frames) <= limit:
        return frames
    import numpy as np

    positions = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[index] for index in dict.fromkeys(int(p) for p in positions)]


def build_windows(
    frames: list[ClipFrame], *, window_s: float, interval_s: float, max_frames: int
) -> list[Window]:
    """The firing schedule the pipeline runs: a `window_s` span every `interval_s`.

    Windows overlap by design (span > interval) so a placement is seen several
    times; `event_cooldown_seconds` is what stops that becoming several memory
    writes. Early windows are clipped to the frames that exist, which is what
    the evidence ring does at the start of a session too.
    """
    if not frames:
        return []
    first, last = frames[0].t, frames[-1].t
    out: list[Window] = []
    end = first + interval_s
    while end <= last + 1e-6:
        span = [f for f in frames if end - window_s - 1e-6 <= f.t <= end + 1e-6]
        if span:
            out.append(Window(max(first, end - window_s), end, subsample(span, max_frames)))
        end += interval_s
    # The tail of a clip is where the placement usually is, and a schedule that
    # stops one interval short would silently never look at it.
    if not out or out[-1].end < last - 1e-6:
        span = [f for f in frames if f.t >= last - window_s - 1e-6]
        out.append(Window(max(first, last - window_s), last, subsample(span, max_frames)))
    return out


def parse_action_tail(reply: str) -> dict[str, dict]:
    """The JSON array of {label, action, location}, keyed by lowercased label.

    Mirrors `reason/cosmos.py::_parse_action_tail`: the last balanced `[...]`
    in the reply, tolerating a reasoning preamble and the grounding tags before
    it. A malformed tail yields nothing, so the arm is scored as `unknown`
    rather than credited with an event it did not clearly state.
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
    out: dict[str, dict] = {}
    for item in parsed:
        if isinstance(item, dict) and "label" in item:
            out[str(item["label"]).strip().lower()] = item
    return out


def parse_labeled_box(reply: str, label: str, order: str) -> tuple | None:
    """The box tagged with `label`, or any box in the reply as a fallback.

    The fallback matters because the tail and the tags disagree on casing and
    plurality more often than they disagree on the object -- and the event axis
    is not trying to re-measure formatting compliance.
    """
    wanted = label.strip().lower()
    for match in _GROUNDING.finditer(reply):
        if match.group("label").strip().lower() == wanted:
            box = parse_box(match.group(0), order)
            if box is not None:
                return box
    return parse_box(reply, order)


def score_events(records: list[dict], truths: dict[str, ClipTruth]) -> dict:
    """Turn per-window replies into the numbers that decide the axis."""
    labels = list(ACTIONS)
    confusion = {truth: dict.fromkeys(labels, 0) for truth in labels}
    for record in records:
        confusion[record["truth_action"]][record["pred_action"]] += 1

    total = len(records)
    correct = sum(confusion[a][a] for a in labels)
    per_action: dict[str, dict] = {}
    f1s: list[float] = []
    for action in labels:
        tp = confusion[action][action]
        support = sum(confusion[action].values())
        predicted = sum(confusion[t][action] for t in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_action[action] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        if support:
            f1s.append(f1)

    # Quiet windows are the ones where the object was *there* and nothing
    # happened to it. Windows where it had left the frame are counted
    # separately: silence there is trivially correct and would dilute the
    # false-positive rate that decides the axis.
    quiet = [r for r in records if r["truth_action"] == "nothing_happened" and not r["absent"]]
    false_placed = sum(1 for r in quiet if r["pred_action"] == "placed")
    false_handling = sum(1 for r in quiet if r["pred_action"] in ("picked_up", "carried"))

    absent = [r for r in records if r["absent"]]
    phantom = sum(1 for r in absent if not r["no_box"])

    # Per-event recall: overlapping windows mean the pipeline needs any one of
    # them to see the placement, so this -- not the per-window rate -- is what
    # the product experiences.
    caught: list[dict] = []
    for clip, truth in truths.items():
        for event in truth.events:
            if event["action"] not in MEMORY_ACTIONS:
                continue
            covering = [
                r
                for r in records
                if r["clip"] == clip and r["start"] < float(event["t"]) <= r["end"]
            ]
            # A window can only report the event if the object is still in its
            # last frame -- that is where the box goes, and no box means no
            # event. An event no window could have caught is a corpus fact, not
            # a model failure, so it is counted apart from recall.
            catchable = [r for r in covering if not r["absent"]]
            hits = [r for r in catchable if r["pred_action"] == event["action"]]
            caught.append(
                {
                    "clip": clip,
                    "t": event["t"],
                    "action": event["action"],
                    "windows_covering": len(covering),
                    "windows_catchable": len(catchable),
                    "windows_hit": len(hits),
                    "detection_delay_s": (
                        round(min(r["end"] for r in hits) - float(event["t"]), 2) if hits else None
                    ),
                }
            )
    placements = [c for c in caught if c["action"] == "placed"]
    catchable_placements = [c for c in placements if c["windows_catchable"]]
    found = [c for c in catchable_placements if c["windows_hit"]]
    delays = [c["detection_delay_s"] for c in found if c["detection_delay_s"] is not None]

    ious = [r["window_iou"] for r in records if r["window_iou"] is not None]
    return {
        "windows": total,
        "window_accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(statistics.fmean(f1s), 4) if f1s else 0.0,
        "per_action": per_action,
        "confusion": confusion,
        "quiet_windows": len(quiet),
        "false_placed": false_placed,
        "false_placed_rate": round(false_placed / len(quiet), 4) if quiet else 0.0,
        "false_handling": false_handling,
        "false_handling_rate": round(false_handling / len(quiet), 4) if quiet else 0.0,
        "absent_windows": len(absent),
        "phantom_boxes": phantom,
        "phantom_rate": round(phantom / len(absent), 4) if absent else 0.0,
        "placements": len(placements),
        "placements_catchable": len(catchable_placements),
        "placements_found": len(found),
        "placement_recall": (
            round(len(found) / len(catchable_placements), 4) if catchable_placements else 0.0
        ),
        "detection_delay_p50_s": round(statistics.median(delays), 2) if delays else None,
        "events": caught,
        "no_box_windows": sum(1 for r in records if r["no_box"]),
        "mean_window_iou": round(statistics.fmean(ious), 4) if ious else 0.0,
    }


# --- The run -----------------------------------------------------------------

IMAGE_SUFFIXES = frozenset({".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"})
SPLITS = ("reference", "query_clean", "query_realistic")


def collect(dataset: Path, cache: Path, limit: int | None) -> list[tuple[str, Path, Path]]:
    """Every probe image that has a human-annotated box, in a stable order.

    Stable because the arms are run in separate processes, often on separate
    days, and a comparison across arms is only valid if they saw the same
    images in the same order.
    """
    items: list[tuple[str, Path, Path]] = []
    for instance in sorted(p for p in dataset.iterdir() if p.is_dir() and p.name != "all"):
        for split in SPLITS:
            folder = instance / split
            if not folder.is_dir():
                continue
            for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                cached = cache / f"{instance.name}__{split}__{path.stem}.npz"
                if cached.exists():
                    items.append((f"{instance.name}/{split}/{path.stem}", path, cached))
    return items[:limit] if limit else items


def score(replies: list[dict], order: str) -> dict:
    """Score already-collected replies under one coordinate convention.

    Separating the call from the scoring is what makes `--coord-order auto`
    honest: both readings are applied to the *same* replies, so the convention
    is measured rather than being a second, differently-sampled run.
    """
    ious: list[float] = []
    containments: list[float] = []
    ratios: list[float] = []
    no_box = 0
    for reply in replies:
        box = parse_box(reply["raw"], order)
        if box is None:
            no_box += 1
            continue
        truth = reply["truth"]
        ious.append(iou(truth, box))
        inside, ratio = containment(box, truth)
        containments.append(inside)
        ratios.append(ratio)
    scored = len(ious)
    return {
        "coord_order": order,
        "scored": scored,
        "no_box": no_box,
        "mean_iou": round(statistics.fmean(ious), 4) if ious else 0.0,
        "median_iou": round(statistics.median(ious), 4) if ious else 0.0,
        "iou_ge_0p5": sum(1 for v in ious if v >= 0.5),
        "iou_ge_0p5_rate": round(sum(1 for v in ious if v >= 0.5) / scored, 4) if scored else 0.0,
        "mean_containment": round(statistics.fmean(containments), 4) if containments else 0.0,
        "mean_area_ratio": round(statistics.fmean(ratios), 4) if ratios else 0.0,
    }


def run_events(args, arm: Arm, drift: str) -> int:
    """Score one arm on the event axis across every annotated recording."""
    truth_files = sorted(args.truth_dir.glob("*.json"))
    if not truth_files:
        print(
            f"no annotations under {args.truth_dir}.\n"
            "Annotate first:  uv run python scripts/annotate_placement.py annotate "
            "--clip ../../clips/recordings/<file>.mp4"
        )
        return 1

    print(f"arm      {arm.name}  ({arm.model})")
    print(f"licence  {arm.licence}{'' if arm.shippable else '   [CEILING ONLY -- not shippable]'}")
    print(f"clips    {len(truth_files)}   prompt drift: {drift}")
    print(
        f"windows  {args.window_seconds:.0f}s span every {args.interval_seconds:.0f}s, "
        f"{args.max_frames} frames each -- the production schedule"
    )
    print("-" * 72)

    truths: dict[str, ClipTruth] = {}
    records: list[dict] = []
    latencies: list[float] = []
    errors = 0
    for path in truth_files:
        truth = load_truth(path)
        clip = args.clips / truth.clip
        if not clip.is_file():
            print(f"  {truth.clip}: SKIPPED (not found under {args.clips})")
            continue
        truths[truth.clip] = truth
        frames = decode_clip(clip, truth.frame_stride, args.long_edge, truth.rotate)
        windows = build_windows(
            frames,
            window_s=args.window_seconds,
            interval_s=args.interval_seconds,
            max_frames=args.max_frames,
        )
        print(f"  {truth.clip}  {len(frames)} frames -> {len(windows)} windows")
        for window in windows:
            content = [text_block(window_prompt([truth.label], len(window.frames)))]
            content += [image_block(frame.data_url) for frame in window.frames]
            started = time.perf_counter()
            try:
                raw = ask_openai(args.base_url, arm.model, content, args.timeout, args.max_tokens)
            except (urllib.error.URLError, OSError, KeyError, TimeoutError) as exc:
                errors += 1
                print(f"    {window.end:6.1f}s  ERROR {exc}")
                continue
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            truth_action, absent = truth.expected(window.start, window.end, window.frames[-1].index)
            records.append(
                {
                    "clip": truth.clip,
                    "label": truth.label,
                    "start": round(window.start, 2),
                    "end": round(window.end, 2),
                    "frames": len(window.frames),
                    "last_index": window.frames[-1].index,
                    "truth_action": truth_action,
                    "absent": absent,
                    "raw": raw,
                    "latency_s": round(elapsed, 3),
                }
            )

    if not records:
        print(f"\nno replies at all ({errors} errors) -- is the server up at {args.base_url}?")
        return 1

    requested = args.coord_order or arm.coord_order
    orders = ("xyxy", "yxyx") if requested == "auto" else (requested,)
    scored = [(order, _resolve(records, truths, order)) for order in orders]
    order, resolved = max(scored, key=lambda pair: _mean_iou(pair[1]))
    for record, extra in zip(records, resolved, strict=True):
        record.update(extra)
    summary = score_events(records, truths)

    for record in records:
        mark = "ok " if record["truth_action"] == record["pred_action"] else "MISS"
        if record["window_iou"] is not None:
            grounding = f"IoU {record['window_iou']:.2f}"
        elif record["absent"]:
            # Truth says the object has left the frame, so no box is the right
            # answer and a box is a phantom.
            grounding = "absent, no box" if record["no_box"] else "PHANTOM BOX"
        else:
            grounding = "no box"
        print(
            f"  {record['clip'][:26]:<26} {record['end']:6.1f}s  {mark} "
            f"truth={record['truth_action']:<16} said={record['pred_action']:<16} {grounding}"
        )

    report = {
        "axis": "events",
        "arm": arm.name,
        "model": arm.model,
        "licence": arm.licence,
        "shippable": arm.shippable,
        "params_b": arm.params_b,
        "runtime": arm.runtime,
        "clips": sorted(truths),
        "errors": errors,
        "prompt_drift": drift,
        "long_edge": args.long_edge,
        "window_seconds": args.window_seconds,
        "interval_seconds": args.interval_seconds,
        "max_frames": args.max_frames,
        "coord_order_used": order,
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(
            sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3
        ),
        "latency_mean_s": round(statistics.fmean(latencies), 3),
        "gpu_used_mib": gpu_used_mib(),
        # Recorded, never scored -- judging a location phrase needs a human who
        # watched the clip. Read these when picking the winner; a model that
        # grounds well and describes the wrong room is not shippable either.
        "locations": [
            {
                "clip": r["clip"],
                "end": r["end"],
                "action": r["pred_action"],
                "location": r["location"],
            }
            for r in records
            if r["location"]
        ],
        **summary,
    }

    print("\n" + "=" * 72)
    print(
        f"{arm.name}: placement recall {report['placements_found']}"
        f"/{report['placements_catchable']} catchable "
        f"({report['placement_recall'] * 100:.0f}%)"
        + (
            f"   median delay {report['detection_delay_p50_s']:.1f}s after the event"
            if report["detection_delay_p50_s"] is not None
            else "   (no placement ever caught)"
        )
    )
    print(
        f"  false placements {report['false_placed']}/{report['quiet_windows']} quiet windows "
        f"({report['false_placed_rate'] * 100:.0f}%)  -- each one writes a wrong location"
    )
    print(
        f"  false handling   {report['false_handling']}/{report['quiet_windows']} "
        f"({report['false_handling_rate'] * 100:.0f}%)  -- this is what keeps "
        "promote_motion_events off"
    )
    if report["placements_catchable"] < report["placements"]:
        missing = report["placements"] - report["placements_catchable"]
        print(
            f"  NOTE: {missing} of {report['placements']} placements had left the frame by the "
            "end of every window covering them -- no arm could report those, and they are "
            "excluded from recall. That is a corpus/window-geometry result, not a model one."
        )
    print(
        f"  phantom boxes {report['phantom_boxes']}/{report['absent_windows']} windows where "
        f"the object had left the frame ({report['phantom_rate'] * 100:.0f}%)"
    )
    print(
        f"  window accuracy {report['window_accuracy']:.3f}  macro-F1 {report['macro_f1']:.3f}"
        f"  no box on {report['no_box_windows']}/{report['windows']}"
    )
    print(
        f"  last-frame grounding IoU {report['mean_window_iou']:.3f} "
        f"(against interpolated truth; the grounding axis is the real measurement)"
    )
    print(
        f"  latency p50 {report['latency_p50_s']:.2f}s  p95 {report['latency_p95_s']:.2f}s"
        f"   -- {args.max_frames} frames per call, fired every {args.interval_seconds:.0f}s"
    )
    if report["latency_p50_s"] > args.interval_seconds:
        print("  WARNING: p50 exceeds the firing interval; this arm backs the queue up.")
    if report["gpu_used_mib"]:
        print(f"  card in use: {report['gpu_used_mib']} MiB")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _resolve(records: list[dict], truths: dict[str, ClipTruth], order: str) -> list[dict]:
    """Read every reply under one coordinate convention.

    Only the box depends on the convention, but resolving actions here too
    keeps the two in one place -- and `no_box` changes the predicted action,
    because the pipeline drops a window with no grounding tag entirely
    (`CosmosReasoner._parse` returns nothing without a box), so an arm that
    forgets the tag has effectively said "nothing happened".
    """
    out: list[dict] = []
    for record in records:
        raw = record["raw"]
        entry = parse_action_tail(raw).get(record["label"].strip().lower(), {})
        claimed = str(entry.get("action", "unknown")).strip().lower()
        if claimed not in ACTIONS:
            claimed = "unknown"
        box = parse_labeled_box(raw, record["label"], order)
        truth = truths[record["clip"]]
        reference = truth.box_at(record["last_index"])
        out.append(
            {
                "claimed_action": claimed,
                # What the pipeline would actually have recorded.
                "pred_action": claimed if box is not None else "nothing_happened",
                "no_box": box is None,
                "box": None if box is None else [round(v, 4) for v in box],
                "window_iou": (
                    None if box is None or reference is None else round(iou(reference, box), 4)
                ),
                "location": entry.get("location"),
            }
        )
    return out


def _mean_iou(resolved: list[dict]) -> float:
    values = [r["window_iou"] for r in resolved if r["window_iou"] is not None]
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--arm", required=True, choices=sorted(ARMS), help="which contender")
    parser.add_argument(
        "--task",
        choices=("grounding", "events"),
        default="grounding",
        help="grounding: one box per still image. events: what happened across a window.",
    )
    parser.add_argument("--dataset", type=Path, help="identity-probe root (grounding)")
    parser.add_argument("--cache", type=Path, help="ground-truth boxes (grounding)")
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=repo_root / "docs/spikes/grounder-bakeoff/truth",
        help="annotations from annotate_placement.py (events)",
    )
    parser.add_argument(
        "--clips",
        type=Path,
        default=repo_root / "clips/recordings",
        help="where the annotated recordings live (events)",
    )
    # Defaults are the production values (config.py). Overridable because "is
    # the window wide enough" is itself an open question -- spike 5b only
    # bracketed the minimum between a 10 s window that failed and a 28 s clip
    # that passed -- and this harness is now the cheapest way to answer it.
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--interval-seconds", type=float, default=7.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--out", type=Path, help="write the run's JSON report here")
    parser.add_argument("--label", default="keys")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--device", default="cuda", help="transformers arms only")
    parser.add_argument("--long-edge", type=int, default=768, help="JPEG long edge sent")
    #: 320 is `reason_max_tokens`: a window reply is tags *plus* a JSON tail,
    #: and truncating the tail scores as `unknown` on every window.
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, help="stop after N images (smoke test)")
    parser.add_argument(
        "--coord-order",
        choices=("auto", "xyxy", "yxyx"),
        help="override the arm's declared convention",
    )
    parser.add_argument(
        "--check-prompt-drift",
        action="store_true",
        help="verify the inlined prompt still matches reason/cosmos.py",
    )
    args = parser.parse_args()

    arm = ARMS[args.arm]
    drift = check_prompt_drift(repo_root) if args.check_prompt_drift else "not requested"
    if drift.startswith("DRIFTED"):
        print(f"FATAL: {drift}")
        return 2

    if args.task == "events":
        if arm.transport != "openai":
            parser.error("the event axis sends several frames per call; openai transport only")
        return run_events(args, arm, drift)

    if args.dataset is None or args.cache is None:
        parser.error("--dataset and --cache are required for the grounding task")
    items = collect(args.dataset, args.cache, args.limit)
    if not items:
        parser.error(f"no annotated images found under {args.dataset} with {args.cache}")

    print(f"arm      {arm.name}  ({arm.model})")
    print(f"licence  {arm.licence}{'' if arm.shippable else '   [CEILING ONLY -- not shippable]'}")
    print(f"runtime  {arm.runtime}")
    print(f"images   {len(items)}   prompt drift: {drift}")
    print("-" * 72)

    prompt = localize_prompt(args.label)
    client: TransformersArm | None = None
    load_seconds = 0.0
    if arm.transport == "transformers":
        started = time.perf_counter()
        client = TransformersArm(arm.model, args.device)
        load_seconds = time.perf_counter() - started
        print(f"loaded weights in {load_seconds:.1f}s")

    import numpy as np

    replies: list[dict] = []
    latencies: list[float] = []
    errors = 0
    for name, image_path, cached in items:
        image = load_rgb(image_path)
        truth_px = np.load(cached)["box"]
        truth = (
            float(truth_px[0]) / image.width,
            float(truth_px[1]) / image.height,
            float(truth_px[2]) / image.width,
            float(truth_px[3]) / image.height,
        )
        started = time.perf_counter()
        try:
            if client is not None:
                raw = client.ask(image, prompt, args.max_tokens)
            else:
                raw = ask_openai(
                    args.base_url,
                    arm.model,
                    [text_block(prompt), image_block(encode_data_url(image, args.long_edge))],
                    args.timeout,
                    args.max_tokens,
                )
        except (urllib.error.URLError, OSError, KeyError, TimeoutError) as exc:
            # An arm that errors is a real result -- an unavailable server and a
            # model that cannot answer must not be averaged together silently.
            errors += 1
            print(f"  {name}: ERROR {exc}")
            continue
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        replies.append({"name": name, "raw": raw, "truth": truth})
        preview = parse_box(raw, arm.coord_order if arm.coord_order != "auto" else "xyxy")
        shown = f"IoU {iou(truth, preview):.3f}" if preview else f"NO BOX  raw={raw[:60]!r}"
        print(f"  {name}: {shown}  ({elapsed:.2f}s)")

    if not replies:
        print(f"\nno replies at all ({errors} errors) -- is the server up at {args.base_url}?")
        return 1

    requested = args.coord_order or arm.coord_order
    orders = ("xyxy", "yxyx") if requested == "auto" else (requested,)
    scored = [score(replies, order) for order in orders]
    best = max(scored, key=lambda s: s["mean_iou"])

    report = {
        "arm": arm.name,
        "model": arm.model,
        "licence": arm.licence,
        "shippable": arm.shippable,
        "params_b": arm.params_b,
        "runtime": arm.runtime,
        "notes": arm.notes,
        "dataset": str(args.dataset),
        "cache": str(args.cache),
        "images": len(items),
        "answered": len(replies),
        "errors": errors,
        "prompt_drift": drift,
        "long_edge": args.long_edge,
        "load_seconds": round(load_seconds, 2),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(
            sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 3
        ),
        "latency_mean_s": round(statistics.fmean(latencies), 3),
        "gpu_used_mib": gpu_used_mib(),
        "coord_order_used": best["coord_order"],
        "coord_order_scores": scored,
        **{k: v for k, v in best.items() if k != "coord_order"},
    }

    print("\n" + "=" * 72)
    print(f"{arm.name}: mean IoU {report['mean_iou']:.3f}  median {report['median_iou']:.3f}")
    print(
        f"  IoU>=0.5 {report['iou_ge_0p5']}/{report['scored']} "
        f"({report['iou_ge_0p5_rate'] * 100:.0f}%)   no box on {report['no_box']}"
    )
    print(
        f"  containment {report['mean_containment']:.3f}  "
        f"area ratio {report['mean_area_ratio']:.3f}  "
        "(containment ~1 with low IoU = right object, wrong extent)"
    )
    print(
        f"  latency p50 {report['latency_p50_s']:.2f}s  p95 {report['latency_p95_s']:.2f}s"
        "   -- the pipeline's window interval is 7s"
    )
    if report["latency_p50_s"] > 7.0:
        print("  WARNING: p50 exceeds reason_interval_seconds; this arm backs the queue up.")
    if len(orders) > 1:
        other = min(scored, key=lambda s: s["mean_iou"])
        print(
            f"  coordinate convention measured as {best['coord_order']} "
            f"({best['mean_iou']:.3f} vs {other['mean_iou']:.3f} under {other['coord_order']})"
        )
    if report["gpu_used_mib"]:
        print(f"  card in use: {report['gpu_used_mib']} MiB")
    if not arm.shippable:
        print("  NOTE: ceiling only -- this licence cannot ship in the product.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
