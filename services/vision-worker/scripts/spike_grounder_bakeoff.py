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

## The four axes

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


def localize_prompt(label: str) -> str:
    requirements = _LOCALIZE_REQUIREMENTS.get(
        label, f"{label.upper()} means the physical object itself, not a screen image of one."
    )
    return _LOCALIZE_PROMPT.format(
        label=label, requirements=requirements, extent=_EXTENT_RULE
    )


def check_prompt_drift(repo_root: Path) -> str:
    """Compare the inlined extent rule against the production one.

    Returns a status string rather than raising: on the rented box the service
    source may not be checked out beside the harness, and "could not check" must
    not read the same as "checked and matched".
    """
    source = (
        repo_root
        / "services/vision-worker/src/vision_worker/reason/cosmos.py"
    )
    if not source.is_file():
        return "unchecked (reason/cosmos.py not found)"
    text = source.read_text(encoding="utf-8")
    # The rule is a parenthesised implicit-concatenation of string literals, so
    # compare on collapsed whitespace and stripped quotes rather than exact text.
    match = re.search(r"_EXTENT_RULE\s*=\s*\((?P<body>.*?)\)\n", text, re.DOTALL)
    if not match:
        return "unchecked (_EXTENT_RULE not parseable)"
    literal = "".join(re.findall(r'"([^"]*)"', match.group("body")))
    if _collapse(literal) == _collapse(_EXTENT_RULE):
        return "ok (matches reason/cosmos.py)"
    return "DRIFTED -- the inlined prompt no longer matches reason/cosmos.py"


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
    copy.thumbnail((long_edge, long_edge), Image.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def ask_openai(
    base_url: str, model: str, data_url: str, prompt: str, timeout: float, max_tokens: int
) -> str:
    """One image, one prompt, plain chat -- and never `response_format`.

    Guided decoding against a box schema measured 0.05-0.16 IoU where free-form
    native `<box>` tokens measured 0.55 on the same images. The constraint is
    inherited, not rediscovered; see the cosmos-grounding-constraints note.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
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
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS), help="which contender")
    parser.add_argument("--dataset", type=Path, required=True, help="identity-probe root")
    parser.add_argument("--cache", type=Path, required=True, help="ground-truth boxes")
    parser.add_argument("--out", type=Path, help="write the run's JSON report here")
    parser.add_argument("--label", default="keys")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--device", default="cuda", help="transformers arms only")
    parser.add_argument("--long-edge", type=int, default=768, help="JPEG long edge sent")
    parser.add_argument("--max-tokens", type=int, default=256)
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
    repo_root = Path(__file__).resolve().parents[3]
    drift = check_prompt_drift(repo_root) if args.check_prompt_drift else "not requested"
    if drift.startswith("DRIFTED"):
        print(f"FATAL: {drift}")
        return 2

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
                    encode_data_url(image, args.long_edge),
                    prompt,
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
