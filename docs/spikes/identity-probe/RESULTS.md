# Personal-object identity probe results

## Decision

**Adopt with constraints for the keys-only demo.** C-RADIOv4-SO400M clears the Phase-0
kill-switch and materially outperforms the CLIP baseline on same-keyring versus a different
keyring. Continue with the registry and fixture-backed identity integration. The measured
thresholds are exploratory defaults for keys, not general object-identity benchmark claims.
Masked object crops, a larger multi-class development set, a frozen held-out set, and physical
GN100 coexistence remain required before a release claim.

## Record

- Date: 2026-08-15
- Owner: Visual Memory Assistant maintainers
- Time box: Phase 0
- Source baseline: `8fdf61679f1ba1d8fd8415fa2d454f13cfbf2c27` on
  `feature/personal-object-identity` (the probe implementation is the Phase-0 commit following
  this baseline)
- Machine: `alex-laptop`, WSL2 Linux x86_64, NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB,
  driver 561.17
- Runtime: Python 3.11, Torch 2.6.0+cu126, CUDA 12.6, Transformers 4.57.6
- Primary checkpoint: `nvidia/C-RADIOv4-SO400M` revision
  `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`, fp32 weights with fp16 CUDA autocast,
  NVIDIA Open Model License
- Baseline checkpoint: `openai/clip-vit-base-patch32` revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, fp16 CUDA, MIT model license
- Inputs: local, gitignored `clips/identity-probe`; 3 physical keyrings, 11 reference images,
  8 clean queries, and 17 realistic queries. `keys_3` has no clean-query slice.
- Access: both checkpoints were downloadable without gated authentication; cache sizes were
  approximately 1.7 GB for C-RADIO and 1.2 GB for CLIP.

## Method

Each physical keyring is independently treated as the enrolled target. Its query is a positive;
a rotating query from another physical keyring is the same-class negative. Every positive has
exactly one negative, preserving the balanced IPLoc-ID protocol where always accepting produces
F1=0.667. C-RADIO's two distilled-teacher summary tokens are mean-pooled to the same 1,152
features used by the production adapter. A query scores as the maximum cosine against the
target's reference views. The selected threshold maximizes F1 subject to negative accuracy of
at least 0.80. Images are EXIF-corrected,
square-padded, and resized; this preliminary run does not have the masked crop path added in
Phase 2.

Reproduce from `services/vision-worker`:

```bash
uv sync --extra models
uv run --extra models python scripts/probe_identity.py \
  --dataset ../../clips/identity-probe \
  --backbone both \
  --device cuda \
  --batch-size 1
```

## Results

All rates include numerator/denominator. With only 25 positive and 25 negative trials, the set is
too small for a reliability claim.

| Metric | C-RADIOv4 | CLIP |
|---|---:|---:|
| Chosen cosine threshold | 0.8334 | 0.8482 |
| Accept/reject F1 | **0.917** | 0.750 |
| Positive accuracy | **0.880 (22/25)** | 0.720 (18/25) |
| Negative accuracy | **0.960 (24/25)** | 0.800 (20/25) |
| ROC-AUC | **0.968** | 0.762 |
| Same cosine, mean +/- std | 0.8787 +/- 0.0463 | 0.8573 +/- 0.0313 |
| Different cosine, mean +/- std | 0.7396 +/- 0.0512 | 0.8248 +/- 0.0386 |
| Suggested minimum margin | 0.0440 | 0.0000 |
| Empirical escalation band | 0.8216-0.8334 | 0.8169-0.8675 |
| Forward latency p50 / p95 | 61.3 / 80.8 ms, N=36 | 25.6 / 25.9 ms, N=36 |
| Peak CUDA allocation | 2,513.3 MiB | 1,852.5 MiB |

C-RADIO's F1 lift over CLIP is **+0.167**. At the initial PeKit threshold of 0.75, C-RADIO
accepts all positives (25/25) but rejects only 15/25 negatives, confirming that the threshold
cannot be copied unchanged onto these unmasked photographs.

Slice results at C-RADIO's chosen threshold:

| Slice | F1 | Positive accuracy | Negative accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| Clean | 1.000 | 1.000 (8/8) | 1.000 (8/8) | 1.000 |
| Realistic | 0.903 | 0.824 (14/17) | 1.000 (17/17) | 0.958 |
| Combined | 0.917 | 0.880 (22/25) | 0.960 (24/25) | 0.968 |

## Failures and constraints

C-RADIO has three false rejects (`keys_1/IMG_8279`, `keys_2/IMG_8253`, and
`keys_2/IMG_8254`) and one binary false accept (a `keys_1/IMG_8246` query against the
`keys_3` gallery). The later all-gallery resolver also requires the winning object to clear a
runner-up margin, so the binary false accept is expected to be safer than this target-at-a-time
probe suggests; Phase 2 must verify that rather than assume it.

The 0.88 positive accuracy misses the provisional 0.90 target by one trial. This does not trigger
the named kill-switch, which is negative accuracy below 0.80 even with escalation. It instead
supports the planned masked-crop parity and VLM escalation work. The current photographs also
contain substantial background and hands, making this a conservative test of the backbone but a
poor final threshold source.

## Follow-up

- Use `min_cosine=0.8334`, `min_margin=0.0440`, and escalation band
  `0.8216-0.8334` as keys-demo starting values, configurable rather than constants.
- Re-run after the shared masked crop path lands; do not freeze thresholds before that parity run.
- Expand to at least 10 physical objects and freeze a held-out set before any general claim.
- Measure C-RADIO beside detector, verifier, speech, and media on the physical GN100.
