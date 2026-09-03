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
| LFM2.5-VL-3B | 3B | LFM Open License — free below $10M revenue | ✅ capped |
| ~~Moondream 3~~ | 9B MoE | **BSL 1.1**, "No Third-Party Service" | ❌ excluded |

LFM2.5-VL-3B is the current measured best (IoU 0.889, **0.92** with the extent
rule, spike 12c) and the smallest arm at 6.02 GiB resident — if it wins, the
deployment card drops with it, potentially to a 48 GB L40S at ~$1.55/hr.

Its licence caps commercial use above $10M annual revenue. **Owner decision,
2026-09-02: contend anyway.** The threshold is distant, and a grounder is a
replaceable component on roughly a six-month cycle — by the time the cap binds,
its replacement probably has not been published yet, and switching cost is a
bake-off re-run rather than a rewrite. What the choice costs is the *claim*, not
the code: "fully open source" becomes "open weights, commercially licensed above
$10M" if a customer asks. Due for review when revenue is within a year of the cap.

**Moondream 3 remains excluded, and it is not the same case.** BSL 1.1's "No
Third-Party Service" grant is not a revenue threshold to grow into — it forbids
offering the model as a service at *any* size, which is exactly what this product
is. No amount of growth makes it legal, so unlike LFM there is no trade to weigh.
[vLLM also does not support the architecture](https://github.com/vllm-project/vllm/issues/25215).

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
pip install "vllm>=0.23"        # >=0.13 covers Qwen3-VL FP8 + Cosmos; LFM needs >=0.23
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

### LFM2.5-VL-3B — vLLM, like the others

```bash
vllm serve LiquidAI/LFM2.5-VL-3B --port 8001 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":1}'
```

**Needs vLLM ≥ 0.23.0** — LFM2.5-VL is native (`Lfm2VlForConditionalGeneration`,
no `--trust-remote-code`) but landed later than the Qwen arms, which only need
≥0.13. Use one vLLM ≥0.23 for every arm so the latency axis is a comparison
rather than a mix of serving stacks.

Earlier grounding spikes ran LFM under `uv --isolated` with `transformers>=5.0`,
because the service pins 4.57.6 for C-RADIOv4. That is no longer necessary here:
serving it puts the dependency split behind an HTTP boundary, which is also how
the deployment would run it.

## Reading the result

The winner is **not** simply the highest IoU.

1. **Shippable arms only** — every arm in the table now qualifies except
   Moondream 3, which is not scored. If LFM wins, note in RESULTS.md what its
   licence costs the pitch, so the trade stays a decision rather than drifting
   into an unexamined default.
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
