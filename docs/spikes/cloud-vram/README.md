# Cloud VRAM spike — how big must the card be?

**Question.** Every VRAM number in this project was taken on the 8 GB laptop, one
model at a time. Two of them — the two that dominate the budget — were never
measured at all, because they never fit. Before renting a deployment card,
measure the whole stack **co-resident on one GPU, at its workload peak**, and get
one number.

**Status.** Harness ready, not yet run. `runs/` is empty until a card is rented.

## The stack

| Model | Role | Where it runs | Prior knowledge |
|---|---|---|---|
| Nemotron 3.5 Lightning 30B-A3B **NVFP4** | agent reasoning, tool routing | vLLM server | never measured; ~17–19 GiB weights derived |
| Cosmos3-Nano 16B **BF16** | grounder + event reasoner | vLLM server | never measured; **32 GiB** weights derived, BF16-only upstream |
| C-RADIOv4-H 653M | identity embedding | vision-worker | ~2.5 GiB (measured, alone) |
| SAM2.1-tiny | tracker | vision-worker | 0.2 GiB + **31.6 MiB/frame** VOS |
| Parakeet-TDT-0.6B v3 | ASR | speech | **activation-bound: tried 10.82 GiB on one 20 s utterance** |
| Kokoro-82M | TTS | speech | ~0.3 GiB expected |

Grounder candidates from the bake-off (`../grounder-bakeoff/`) are servable here
too — `qwen3vl4b`, `qwen3vl8b`, `mossvl` — so the card can be sized against the
winner rather than against the incumbent.

## The trap this probe exists to avoid

**vLLM's footprint is a policy, not a need.** `--gpu-memory-utilization` tells the
server what fraction of the card to take, and it takes it whether or not the
model needs it. Point vLLM at a 141 GiB card with the 0.9 default and it will
reserve ~127 GiB for a model whose weights are 32 GiB — then `nvidia-smi` will
report 127 GiB and you will buy a card sized for vLLM's appetite.

So "how much does Cosmos use" is not a well-formed question. Two things are:

- **`weights_mib`** — the floor. No setting tunes it away.
- **`kv_cache_mib`** — what the fraction bought at *this* utilization.

The probe parses both from each server's own startup log, and the driver defaults
to `--vllm-util 0.25`, **not** vLLM's 0.9, because several servers share the card.
Leave it at 0.9 and the first server started takes everything and the rest OOM.

## Why process-per-model

Two constraints make a single-process test both impossible and dishonest:

1. **Dependency split.** LFM2.5-VL needs `transformers>=5.0`, SAM2 video ships in
   4.57.6, Parakeet is NeMo. They cannot share an interpreter. Each worker runs
   under its own `--python` interpreter.
2. **Production is already multi-process.** vision-worker, speech, and two model
   servers are four processes, each paying its own **~1–2 GiB CUDA context**. That
   is real memory the deployment cannot avoid and a single-process test hides. The
   driver reports it as `implied_shared_or_unattributed_mib`.

Each worker loads, runs a representative workload, reports its peak, then
**holds** while the driver reads the card total from `nvidia-smi`.

## Running it

```bash
# venv A -- transformers 4.57.x: radio, sam2, parakeet (NeMo), kokoro
python -m venv .tf457 && .tf457/bin/pip install \
  "transformers==4.57.6" torch soundfile "nemo_toolkit[asr]" kokoro pillow numpy
# venv B -- transformers 5: lfm only
python -m venv .tf5 && .tf5/bin/pip install "transformers>=5.0" torch pillow numpy
# servers
pip install "vllm>=0.13" && pip install "sglang[all]"   # sglang only if mossvl is in play
```

Measure each in isolation first — it catches a bad install before a long
co-resident run:

```bash
.tf457/bin/python vram_probe.py worker --model parakeet --utterance-seconds 8
.tf5/bin/python   vram_probe.py worker --model lfm
```

The co-resident run — the answer:

```bash
.tf457/bin/python vram_probe.py driver \
  --models radio,sam2,parakeet,kokoro,nemotron,cosmos \
  --python lfm=$PWD/.tf5/bin/python \
  --vllm-util 0.25 --utterance-seconds 8 --frames 20 \
  --out runs/coresident-cosmos.json
```

Then swap the grounder to re-size against the bake-off winner:

```bash
.tf457/bin/python vram_probe.py driver \
  --models radio,sam2,parakeet,kokoro,nemotron,qwen3vl8b \
  --vllm-util 0.25 --out runs/coresident-qwen3vl8b.json
```

**Sweep Parakeet.** Its cost is activation, not weights, and the 10.82 GiB event
is the widest single uncertainty in the whole budget:

```bash
for s in 4 8 12 20; do
  .tf457/bin/python vram_probe.py worker --model parakeet --utterance-seconds $s
done
```

Servers download 50+ GiB on first run and `--server-timeout` defaults to 1800 s
for that reason. Seed the HF cache onto the persistent volume first and later
runs start in minutes.

## What the report answers

- `co_resident_used_mib` — **the number that picks the card**, with
  `fits_48gib_l40s` / `fits_80gib_h100` / `fits_96gib_rtx_pro_6000` /
  `fits_141gib_h200` computed from it.
- `sum_of_server_weights_mib` — the floor, ignoring KV policy. Read this
  *alongside* the total: the gap between them is what you can tune.
- `per_model[].workload_peak_mib` — Parakeet's is the one to watch.
- `implied_shared_or_unattributed_mib` — the CUDA-context tax of N processes,
  invisible in every prior single-model measurement.

## Nebius cards on the table

| Card | VRAM | On-demand | Note |
|---|---|---|---|
| L40S | 48 GB | from $1.55/hr | only reachable if Cosmos is replaced |
| HGX H100 | 80 GB | $3.85/hr | works with the current `torch==2.6.0`/cu126 pin |
| RTX PRO 6000 | 96 GB | $1.80/hr | **Blackwell sm_120 — needs torch ≥2.7/cu128**, the x86_64 pin tops out below it |
| HGX H200 | 141 GB | $4.50/hr | safe, and today's pins work unchanged |

Use **preemptible** for this measurement ($2.45/hr H200) and never for a customer
demo — it can be reclaimed mid-call.
