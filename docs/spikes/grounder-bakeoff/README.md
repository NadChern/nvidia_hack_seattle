# Grounder bake-off — which open-source model should ship?

**Question.** The hackathon is over. In customer discovery the grounder choice is
no longer "what did we demo with" but "what can we ship, and is it fast enough
for realtime video". Those are two different filters and the incumbent passes
neither cleanly.

**The constraint that decides it:** the model must be **open source**, because
the privacy-first story is the product. That single requirement removes the two
best-grounding candidates from contention.

| Model | Params | Licence | Shippable |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | 4B | Apache-2.0 | ✅ |
| Qwen3-VL-8B-Instruct | 9B | Apache-2.0 | ✅ |
| MOSS-VL-Realtime | 11.3B | Apache-2.0 | ✅ |
| Cosmos3-Nano *(incumbent)* | 16B | NVIDIA Open Model License | ~ permissive, not Apache |
| LFM2.5-VL-3B | 3B | LFM Open License — **free only below $10M revenue** | ❌ ceiling only |
| ~~Moondream 3~~ | 9B MoE | **BSL 1.1**, "No Third-Party Service" | ❌ excluded |

LFM2.5-VL-3B is the current measured best (IoU 0.889, **0.92** with the extent
rule, spike 12c) and is exactly the model we would otherwise pick. Its revenue
cap bites precisely when the company starts working. It stays in the table as a
**ceiling**: without it we cannot tell whether the shippable winner is good or
merely the least bad of a bad set.

Moondream 3 is excluded outright rather than scored — BSL 1.1 forbids offering it
as a third-party service, which is the product, and [vLLM does not support the
architecture](https://github.com/vllm-project/vllm/issues/25215).

## The four axes

Grounding IoU alone already picked the wrong model once here. In spike 3c
containment sat at ~100% while identity F1 was 0.776: the box was on the right
object at the wrong *extent*, and IoU could not say so. Every arm reports:

| Axis | Measures | Gate |
|---|---|---|
| **IoU** | box vs 36 human annotations | ≥ 0.5 for a usable C-RADIO crop |
| **Containment + area ratio** | wrong object *vs* wrong extent | containment ≈ 1 with low IoU ⇒ prompt problem, not model problem |
| **Latency p50/p95** | per grounding call | **< 7 s** = `reason_interval_seconds`, or the pipeline backs up |
| **No-box rate** | silent declines | a decline reads downstream as "object absent" |

Latency is the axis "realtime" actually lives on, and it is the one the
incumbent is weakest on (~5 s warm, against a 7 s window interval).

## Data

- **Images:** `clips/identity-probe/` — 36 HEIC stills, keys_1 (15), keys_2 (14),
  keys_3 (7), across `reference` / `query_clean` / `query_realistic`.
- **Ground truth:** `clips/spike1-arms/_cache/` — 36 human-annotated boxes, 1:1
  with the images. Not model output; these are the annotations spike 1 built.

Do **not** score against `clips/extent-*/_boxes/` — those are *predicted* boxes
saved from an earlier run, and scoring a model against another model's output
measures agreement, not accuracy.

## Prompt parity

Every arm gets the **production** enrollment-localize prompt from
`reason/cosmos.py`, extent rule included — not a prompt tuned per model. Per-arm
tuning would measure prompt-engineering effort rather than model quality, and
spike 12c showed one sentence moves the worst noun from IoU 0.12 to 0.92, an
effect big enough to swamp the differences we are looking for.

The harness inlines that prompt (it must run under `uv --isolated`, where the
service package is not importable) and `--check-prompt-drift` re-reads
`reason/cosmos.py` to prove the copy is current. A drift is fatal, not a warning.

## Running it

One arm at a time, one server at a time — so `nvidia-smi` attribution is
unambiguous and no two models contend for the card.

```bash
# 0. once per box
pip install "vllm>=0.13"        # Qwen3-VL FP8 + Cosmos
pip install "sglang[all]"       # MOSS-VL only
```

### Qwen3-VL 4B / 8B — Apache-2.0, vLLM

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":1}'
```

### Cosmos3-Nano — the incumbent baseline

```bash
vllm serve nvidia/Cosmos3-Nano --port 8001 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

BF16 only — there is no official FP8/FP4 path, which is why this arm alone costs
~32 GiB of weights.

### MOSS-VL-Realtime — SGLang, not vLLM

```bash
python -m sglang.launch_server --model-path OpenMOSS-Team/MOSS-VL-Realtime \
  --port 8001 --mem-fraction-static 0.85
```

**Count the runtime against it.** Nemotron already runs on vLLM; an SGLang winner
means the deployment maintains two inference runtimes. Not disqualifying, but it
should have to win clearly, not narrowly.

### Score any HTTP arm

```bash
cd services/vision-worker && uv run --isolated \
  --with pillow --with pillow-heif --with numpy \
  python scripts/spike_grounder_bakeoff.py --arm qwen3-vl-8b --check-prompt-drift \
    --dataset ../../clips/identity-probe --cache ../../clips/spike1-arms/_cache \
    --out ../../docs/spikes/grounder-bakeoff/runs/qwen3-vl-8b.json
```

`pillow-heif` is required, not optional — all 36 probe images are HEIC.

### LFM2.5-VL-3B — the ceiling arm, local weights

```bash
cd services/vision-worker && uv run --isolated \
  --with 'transformers>=5.0' --with torch --with accelerate \
  --with pillow --with pillow-heif --with numpy \
  python scripts/spike_grounder_bakeoff.py --arm lfm2.5-vl-3b \
    --dataset ../../clips/identity-probe --cache ../../clips/spike1-arms/_cache \
    --out ../../docs/spikes/grounder-bakeoff/runs/lfm2.5-vl-3b.json
```

Isolated because LFM needs `transformers>=5.0` while the service pins 4.57.6 for
C-RADIOv4 — the same split that forced every grounding spike to run alone.

## Reading the result

The winner is **not** simply the highest IoU.

1. **Shippable arms only.** An unshippable ceiling arm winning tells you how much
   quality the licence constraint costs, not what to deploy.
2. **Latency p50 must be under 7 s.** An arm that grounds beautifully at 12 s
   cannot hold the window interval and is disqualified for realtime video.
3. **Containment ≈ 1 with low IoU is a prompt result, not a model result** — that
   arm is grounding correctly and cropping wrong, and the fix is wording.
4. **Then** VRAM, because it picks the card and the card picks the bill.

A per-arm JSON lands in `runs/`. Fill the summary table in `RESULTS.md` from
those files rather than from terminal scrollback.

## Coordinate conventions are measured, never assumed

Every arm defaults to `--coord-order auto`, which scores the *same* replies under
both xyxy and yxyx and reports which the model actually uses. Guessing this wrong
once already made Cosmos look far worse than it was. Cosmos is pinned to `xyxy`
because it self-reported the convention and it was confirmed.
