# Model Landscape

Reviewed for the July 2026 hackathon stack. Freeze exact repository commits, model revisions, licenses, and container digests after workstation validation; model family names alone are not reproducible.

## Practical model roles

| Capability | Candidate | Role | Readiness and main caveat |
|---|---|---|---|
| Detection and tracking | [YOLOE](https://github.com/THU-MIG/yoloe) + a pure-numpy IoU tracker (`services/vision-worker/src/vision_worker/track/greedy_iou.py`) | Open-vocabulary, real-time, continuous detection on every frame | Integrated (`detect/yoloe.py`, `VMA_DETECTOR_KIND=yoloe`); verified end to end on `dev-wsl-cuda` (RTX 4070 Laptop, torch 2.6.0+cu126) -- see `services/vision-worker/model-manifest.toml` for the pinned checkpoint |
| Personal-object identity | [NVIDIA C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H) (653M) masked embeddings | Resolve *which physical instance* ("my keys", not any keys); store pooled vectors and reference crops | Selected for the GN100 profile at revision `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`. The earlier SO400M keys-only probe remains historical evidence, not an H-model threshold evaluation; H requires fresh enrollment vectors and held-out threshold measurement. |
| Geometry | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), preferably a refreshed `-1.1` checkpoint, MoGe-2 (`Ruicheng/moge-2-vitl-normal`), or YOLO26 depth | Metric depth, back-projection to a world point, and per-object Console ranges | MoGe-2 integrated (`depth/moge.py`, `VMA_DEPTH_KIND=moge`); YOLO26 metric depth integrated as the constrained-laptop overlay profile (`depth/yolo.py`, `VMA_DEPTH_KIND=yolo`) and measured live beside YOLOE and Speech on `dev-wsl-cuda`. Depth is only half the story: `depth_m` is a camera-space range, and turning it into a world position needs `domain/geometry.py`'s back-projection *and* a capture pose, which nothing produces until task #46's `DevicePose`. A `placed` observation today still carries a null room/surface -- honest, not incomplete, until that lands. Depth Anything 3 remains unevaluated (see below) |
| Primary event verifier | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Verify placement windows and return schema-constrained fields | Provisional default; benchmark latency, JSON validity, and action accuracy on the GN100 |
| GN100 event challenger | [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) (16B) | Compare physical and temporal reasoning on identical short clips through a selective verifier sidecar | Candidate for the 128 GB GN100, not the 8 GB laptop; BF16-only upstream validation, ARM64 runtime, latency, and complete-workload headroom remain physical deployment gates. `Cosmos3-Super` (64B) is excluded because BF16 weights leave no safe 128 GB operating margin |
| English speech-to-text | [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), using the exact validated checkpoint | Transcribe English questions and provide timestamps through the Speech Service | Use a platform-compatible local runtime for development; deployment must pass on Linux ARM64/CUDA |
| Local text-to-speech | [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), using the exact validated checkpoint and voice | Synthesize English answers through the Speech Service | Use MLX on compatible Macs and PyTorch/CUDA on Windows or GN100; GN100 execution remains the deployment gate |
| Persistent memory | Relational database + evidence files | Store trusted events, current status, last confirmed placement, and history | Required; more important than vector RAG for “where is X?” |
| Experimental spatial VLM | [Stream3D-VLM-4B](https://stream3d-vlm.github.io/) | Incremental spatial representation from streaming video | Research-only; new implementation and benchmark domain differ from the glasses demo |
| Experimental spatial prompting | [GPT4Scene](https://gpt4scene.github.io/) | Give a VLM consistent object markers across evidence frames and a bird's-eye-view scene image | Test the prompting method before considering its training pipeline |
| Experimental streaming map | [LingBot-Map](https://github.com/Robbyant/lingbot-map) | Produce camera pose, point-map, and reconstruction evidence from selected video frames | Research-only until Linux ARM64/CUDA, drift, latency, and coexistence pass on the GN100 |

## Recommended critical path

```text
YOLOE detection + pure-numpy IoU tracking, continuous at 24fps
  ↓
motion/rest interaction state machine (no hands -- see "Interaction state
  machine: rest, not hands" below)
  ↓
candidate event window + evidence ring
  ├── MoGe-2 geometry cues, low cadence (wired; Depth Anything 3 unevaluated)
  ├── C-RADIOv4 masked instance identity, once per track; Qwen3-VL near-threshold only
  ↓
Qwen3-VL-8B verifier through a schema-validating adapter
  (today: a deterministic rule-based verifier -- see docs/09-Spike-Plan.md
  S04's own stop condition, and `services/vision-worker/src/
  vision_worker/verify/rules.py`)
  ↓
deterministic memory reducer + evidence
```

Use Cosmos3-Nano on the same held-out clips as a challenger, not as a second always-resident VLM. Its 16B BF16 profile is the larger Cosmos3 tier expected to fit within the GN100's 128 GB unified-memory capacity while leaving room for the rest of the stack, but that expectation is not a release result until coexistence is measured on the physical workstation. Cosmos3-Super is 64B and only BF16 is officially tested, so its weights alone would consume approximately the full budget before KV caches, media, speech, database, and operating-system headroom. Freeze one primary verifier before full integration. If neither verifier meets the latency or precision gate, fall back to a conservative rule-based placement event and narrow the demo -- this is not a fallback still to be built; it is the current, working default.

## Role boundaries

### C-RADIOv4: masked personal-object identity

The GN100 profile uses the 653M `C-RADIOv4-H` checkpoint. The same mask-to-crop transform is used for enrollment and matching, then C-RADIOv4 produces a distilled-teacher summary vector and mask-weighted spatial vector. Matching takes the best reference view for each query frame and averages across query frames. The checkpoint revision is part of `embedder_id`, so SO400M vectors are intentionally incompatible and objects enrolled under the old model need fresh H-model views. Identity is a write gate in the current registered-object reasoner pipeline. See [Identity Probe Results](spikes/identity-probe/RESULTS.md) for the historical SO400M evaluation; its thresholds must not be represented as an H-model result. Tracker IDs remain scoped to a media epoch and never become stable identity.

### Interaction state machine: rest, not hands (decision superseding this section's original hand-based design)

Implemented in `services/vision-worker/src/vision_worker/domain/stability.py`. The original plan tracked hand/object contact explicitly; it does not. The decision that matters is not "is a hand touching the object" but "is the object at rest in the world, or moving through it" — keys carried from the kitchen to the front door and pocketed must never resolve to "the front hall" just because the last sighting was there. Deriving that from motion alone removes a model (hand detection), the flakiest inference in the original design (bounding-box overlap as a proxy for grasp), and maps directly onto the Memory Service's three trusted states.

```text
first sighting (never promotes — indistinguishable from "was always there")
  ↓
moving (screen motion diverges from background motion)
  ↓
settling → placed, once stability holds for a dwell period
  ↓
picked up (leaves a confirmed rest state) → moving again
```

A track's very first sample never promotes to `placed`, no matter how stable it looks: an object that has always been sitting somewhere is, for one frame, indistinguishable from an object that was just placed. A track that visibly moved and then settled needs a short dwell period; a track that is stable from its first sample needs a much longer sustained stillness, because motion-then-settle is the only strong evidence a placement genuinely happened.

Motion is read from the frame, not assumed: the state machine compares each object's own screen displacement against the frame's background motion, on the inversion that a held object stays roughly fixed relative to the camera while the background sweeps past, and a resting object's apparent motion tracks the background's, since both come from head movement alone. The default background-motion estimator (`pose/image_motion.py`) is phase correlation — an FFT-based global-translation estimate, pure numpy, no model — chosen specifically so this decision needs no GPU and runs identically on a laptop webcam, in CI, and on the glasses.

Thresholds — dwell frames, the passive-confirmation threshold for a first sighting, the motion-residual cutoff, occlusion tolerance — are configuration reported at `/v1/status`, per the evaluation-run requirement below.

### Plane detection is not used for geometry (prior-art finding)

A prior first-person AR project on the same glasses hardware (RayNeo X3 Pro) found that enabling SLAM plane detection wedges the RGB ShareCamera — it opens but delivers no frames, because plane detection keeps the VGA/Hexagon-DSP pipeline busy enough to starve RGB capture. Depth Anything 3 or a metric-depth model (e.g. MoGe-2, used successfully in that same prior project) is the geometry source instead — see the entry below — and remains firmware-independent, unlike a plane-detection feature that visibly contends with the camera this service depends on.

### YOLOE: detector candidate

[YOLOE](https://github.com/THU-MIG/yoloe) is an open-vocabulary detector at YOLO scale — text-prompted and prompt-free modes, real-time on modest hardware. It was proven on this exact glasses hardware and task by the prior-art project referenced above, which shipped two warm YOLOE checkpoints (a text-prompt variant for known targets, a prompt-free variant for open vocabulary). It is the default detector (`services/vision-worker/src/vision_worker/detect/yoloe.py`, selected via `VMA_DETECTOR_KIND=yoloe`), ported and verified running real inference on `dev-wsl-cuda`. Its `-seg` masks now feed C-RADIOv4 for personal-instance identity; they stay in-process and do not widen the frozen `Detection` wire shape.

Getting torch and ultralytics onto this dev-wsl-cuda profile took three compounding, non-obvious CUDA/packaging fixes, recorded in `services/vision-worker/pyproject.toml`'s comments since a future bump needs the same care: plain PyPI's `torch` pulls a `cuda-toolkit` metapackage targeting CUDA 13, newer than this driver supports, so `torch`/`torchvision` are pinned to PyTorch's own `cu126` index instead; that index only publishes `torchvision` up to `0.21.0`, which pins `torch` to the matching `2.6.0` rather than `>=`; and `torch`'s own declared `nvidia-cudnn-cu12` (and sibling CUDA runtime) dependencies get silently dropped by `uv`'s resolver — it flattens the dependency set from the aarch64 wheel's metadata (which expects a system-provided CUDA/cuDNN, the GN100/Jetson-style pattern) rather than the x86_64 wheel's — so they are re-declared explicitly, pinned to the exact versions `torch==2.6.0+cu126` expects.

### Depth Anything 3: geometry only

Use depth and pose outputs to support geometric tests such as whether an object is above a support plane or whether two detected entities are near each other. Labels such as `coffee_table`, `living_room`, or `beside the laptop` require configured zones, semantic detections, or VLM verification. Treat all derived geometry as uncertain evidence, not precise measurement.

### Qwen3-VL and Cosmos3-Nano: selective verification

Give the verifier a short temporal window, object and hand masks, timestamps, detected semantic candidates, and optional geometry metadata. Require output matching the canonical contract in [Data Contract and Memory Semantics](06-Data-Contract.md).

The adapter must:

- validate JSON and enums;
- reject extra prose or extract it outside the trusted payload;
- retry once with a repair prompt when appropriate;
- record the raw result, validated result, prompt version, and latency;
- send invalid or unsupported results to privacy-scoped candidate diagnostics without creating a canonical observation.

### Stream3D-VLM: research branch

Test offline or at low frequency on recorded glasses video. It may improve room/layout recall and viewpoint-independent reasoning, but its published benchmark uses different data and tasks. Do not treat published latency or accuracy as a GN100 or egocentric-glasses result.

### GPT4Scene: spatial prompting experiment

Borrow the global-local prompting method: mark consistent object IDs across selected evidence frames and a bird's-eye-view image, then give both to the existing verifier. Start with the current Qwen3-VL checkpoint; do not fine-tune GPT4Scene or add it to the live path unless the marked-frame experiment improves held-out room, surface, and relation accuracy.

GPT4Scene does not detect pickup or placement transitions. The hand/object state machine remains the event source, and the deterministic reducer remains the memory authority.

### LingBot-Map: optional geometry producer

Evaluate LingBot-Map as a source of camera pose, point maps, and a bird's-eye-view scaffold. It does not supply semantic labels, stable personal-object identity, or trusted events. Its output is optional evidence for the verifier and must never update memory directly.

The published setup recommends PyTorch/CUDA and FlashInfer. Treat Linux ARM64 support, long-sequence stability, window resets, peak unified-memory use, and coexistence with detection, speech, and the verifier as unproven until the GN100 spike passes.

## Speech runtime profiles

English is the only required speech language for the MVP. Parakeet and Kokoro have already been exercised on Windows and Apple-silicon macOS; those results establish developer-machine feasibility, not GN100 deployment compatibility.

| Profile | STT runtime | TTS runtime | Purpose |
|---|---|---|---|
| `dev-macos` | MLX-compatible Parakeet checkpoint | MLX-compatible Kokoro checkpoint | Local development on Apple silicon |
| `dev-windows-cuda` | Tested PyTorch/CUDA-compatible Parakeet checkpoint | Tested PyTorch/CUDA Kokoro runtime | Local development on Windows with an NVIDIA GPU |
| `dev-remote` | GN100 Speech Service | GN100 Speech Service | Development when a laptop cannot or should not host the models |
| `deploy-gn100` | Linux ARM64/CUDA-compatible Parakeet runtime | Linux ARM64/CUDA-compatible Kokoro runtime | Hackathon and production-like deployment |

The profiles may use runtime-specific converted weights, but they must preserve one Speech Service contract and the same English golden test set. Pin the logical model name, source checkpoint, converted artifact revision or hash, runtime version, precision, voice preset, and audio preprocessing for every profile. Do not infer Linux ARM64 support from a successful MLX or Windows CUDA test.

[Fish Audio `s2.1-pro-free`](https://fish.audio/blog/s2-1-pro-free-api/) is an optional, explicitly enabled cloud comparison backend. It is not a local model variant, is never an automatic fallback, and is excluded from the default privacy-preserving deployment path.

Artifact packaging, model-cache, image, and release requirements are defined in [Development and Deployment](08-Development-and-Deployment.md).

## GN100 compatibility gate

The Acer GN100 uses an ARM-based GB10 platform with 128 GB of coherent unified memory. Unified capacity does not guarantee that all models can run concurrently at the required latency.

Complete this matrix on the actual workstation:

| Item | Required record |
|---|---|
| Base system | DGX OS version, kernel, ARM64 architecture |
| GPU stack | Driver, CUDA, cuDNN, container toolkit |
| Python stack | Python, PyTorch, Transformers, vLLM, Flash Attention |
| Model | Exact checkpoint, revision, precision/quantization, license, access status |
| Cold start | Download complete, model load time, compilation time |
| Runtime | Peak unified memory, p50/p95 latency, input resolution/FPS |
| Coexistence | Detection + verifier + speech peak memory and latency |
| Media | WebRTC codec decode, FFmpeg/TorchCodec, reconnect behavior |

Known setup constraints to verify:

- SAM 3.1 currently requires a recent Python/PyTorch/CUDA stack and authenticated checkpoint access.
- Qwen3-VL requires a recent Transformers or serving stack; verify the selected 8B revision on ARM64.
- Cosmos3-Nano is a 16B omni model whose upstream vLLM path supports an OpenAI-compatible API and whose published runtime uses CUDA 12.8 or 13. Only BF16 is officially tested. Isolate it behind a verifier adapter, cap context and evidence windows, and measure its model cache, peak unified memory, latency, and coexistence on the physical GN100 before selecting it.
- Cosmos3-Edge is not a laptop fallback: its repository contains about 9.2 GB of artifacts, including about 9.1 GB of weights, which already exceeds the development laptop's 8,188 MiB VRAM before runtime allocations. The tested laptop profile uses the explicitly enabled MiniCPM API instead.
- Cosmos3-Super is 64B and is not a safe default for a 128 GB shared-memory deployment while upstream supports only BF16; do not select it without a measured configuration that preserves release headroom.
- Compiled optional packages may fail even when base PyTorch inference works.

The database and evidence store consume system memory and disk, not GPU compute. Budget the GN100's unified memory across model weights, KV caches, video buffers, speech, and ordinary processes, leaving operational headroom.

## Persistent identity

Tracker IDs are temporary. Maintain a user-object record with:

- stable `object_id` and aliases;
- reference images or embeddings;
- distinguishing visual attributes;
- previous trusted state;
- enrollment and last-confirmed timestamps;
- ambiguity and user-correction history.

Reject uncertain matches or ask the user rather than silently merging similar objects. Include similar-looking objects and reconnects in the evaluation set.

## Secondary candidates

- [CoTracker3](https://github.com/facebookresearch/co-tracker): point tracking and camera-motion cues, not semantic identity
- [VGGT](https://github.com/facebookresearch/vggt) or [MapAnything](https://github.com/facebookresearch/map-anything): offline multi-view reconstruction and room mapping
- [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) or [Depth Pro](https://github.com/apple/ml-depth-pro): alternative depth models
- [HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied), [Mage-VL](https://huggingface.co/microsoft/Mage-VL), and [VLM-3R](https://github.com/VITA-Group/VLM-3R): research candidates that require separate dependency, license, and hardware validation

## Principle

Use specialized models continuously or selectively, validate their outputs, and persist compact structured events. A VLM is a verifier and interpreter, not the database or the authority on current location.

## Primary references

- [Meta SAM 3 repository](https://github.com/facebookresearch/sam3)
- [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL)
- [NVIDIA Cosmos3-Nano model](https://huggingface.co/nvidia/Cosmos3-Nano)
- [NVIDIA Cosmos framework](https://github.com/NVIDIA/cosmos-framework)
- [NVIDIA Parakeet TDT 0.6B v2 English reference checkpoint](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [Kokoro-82M model](https://huggingface.co/hexgrad/Kokoro-82M)
- [Stream3D-VLM project and paper](https://stream3d-vlm.github.io/)
- [GPT4Scene project and paper](https://gpt4scene.github.io/)
- [LingBot-Map repository](https://github.com/Robbyant/lingbot-map)
- [Fish Audio S2.1 Pro free API announcement](https://fish.audio/blog/s2-1-pro-free-api/)
- [Acer Veriton GN100 specifications](https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip)

The execution order, acceptance criteria, and stop conditions for unresolved candidates are maintained in [Spike Plan](09-Spike-Plan.md).
