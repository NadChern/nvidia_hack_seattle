# Spike Plan

This document is the source of truth for short experiments that retire technical risk. A spike produces evidence and a decision; it does not become production code by default.

## Decision rules

- Use recorded glasses media and frozen fixtures whenever possible.
- Pin repository commit, model revision, runtime, precision, prompts, configuration, and input media.
- Record setup time, cold start, p50/p95 latency, peak unified memory, output quality, failures, and license constraints.
- Run deployment candidates on the physical Acer GN100. A Mac, Windows, x86 container, or ARM64 cross-build cannot close a GN100 compatibility question.
- End every spike with `adopt`, `adopt with constraints`, `defer`, or `reject`.
- Stop when the time box expires or a stop condition is met. Do not keep integrating an experiment because it is interesting.

## Portfolio

| ID | Spike | Priority | Status | Suggested owner | Time box | Decision unlocked |
|---|---|---|---|---|---|---|
| S01 | LiveKit media gateway | Required | SDK adopted; device/GN100 gates pending | Person 4 + Person 1 | 1 day for remaining gates | Media integration acceptance |
| S02 | GN100 platform and container preflight | Blocker | Pending | Person 4 + service owners | 1 day | Whether deployment work can rely on the selected CUDA/media stack |
| S03 | Parakeet and Kokoro on GN100 | Required | Pending | Person 4 | 1 day | Deployable English speech profile |
| S04 | SAM 3.1 and hand/object event pipeline | Required | Tracker-level evidence recorded (identity switches, latency); SAM 3.1 and real-footage recall/FP still pending | Person 1 | 1–2 days | Critical-path perception viability |
| S05 | Qwen3-VL versus Cosmos3-Nano | Required | Pending; MiniCPM Agent API compatibility passed on laptop | Person 2 | 1 day | Primary GN100 verifier and fallback |
| S06 | GPT4Scene-style spatial prompting | Optional | Pending | Person 2 | 1 day | Whether markers and a BEV improve spatial semantics |
| S07 | LingBot-Map streaming reconstruction | Optional | Pending | Person 2 | 1 day | Whether a persistent geometry branch is viable |
| S08 | LingBot-Map + GPT4Scene combination | Conditional | Blocked by S06 and S07 | Person 2 | 0.5 day | Whether map-assisted VLM verification is worth integration |
| S09 | Stream3D-VLM comparison | Optional | Pending | Person 2 | 1 day | Whether the research branch adds value beyond Path A |
| S10 | Glasses client (pure Kotlin) | Required | SG-A/A2/B/B2/C/D complete; SG-E/F need a session on the device | Person 4 | 0.5 day done, 1 day on device | Whether [docs/15](15-Glasses-Client-Plan.md) can be built as written |

S02-S05 can run in parallel with schema, reducer, API, UI, and prerecorded-pipeline development. They block freezing the final deployment models, not starting the project. S06-S09 do not block the MVP.

## Completed architecture decision: NVIDIA VSS

The [NVIDIA Video Search and Summarization Blueprint](https://docs.nvidia.com/vss/latest/index.html) has been reviewed. Adopt its separation of real-time perception, downstream verification, evidence-window retrieval, explicit verification outcomes, bounded query tools, and per-stage observability.

Do not deploy the complete blueprint. LiveKit, the project Vision Service, controlled evidence storage, and the relational Memory Service remain the selected components. No VSS runtime spike is required unless a future proposal names one component, the exact gap it fills, and a measurable acceptance test.

## S01 — LiveKit media gateway

**Decision already made:** Use self-hosted [LiveKit](https://livekit.io/) for the local WebRTC media boundary. S10 supersedes this spike's former Unity-client pin with a pure-Kotlin client using the LiveKit Android SDK. LiveKit remains a transport component and does not own inference, evidence, or memory.

**Existing result:** The local executable spike passed JWT authentication, audio/video subscription, bounded sampling, return audio, and deliberate disconnect/rejoin. See the [spike README](spikes/livekit-media-gateway/README.md) and [results](spikes/livekit-media-gateway/RESULTS.md).

**Superseded client evidence:** the Unity SDK maturity result remains historical evidence but no longer selects the client runtime. [S10](#s10--glasses-client) owns the native Android decision and device gate.

**Remaining question:** Does the native Android client and validated server stack pass the actual glasses and GN100 integration gates?

**Remaining gates:**

- run the pinned LiveKit Android SDK through `apps/glasses-x3` on the actual glasses and validate camera, microphone, HUD, return audio, permissions, and lifecycle behavior;
- validate the selected codecs and raw-track workers on Linux ARM64;
- run on the physical GN100 with external signaling and TURN disabled;
- confirm a reconnect starts a new media epoch and does not reuse tracker identity;
- measure latency, queue bounds, dropped frames, and 30-minute stability.

**Decision:** Keep the validated server boundary and use S10's native Android client. Release the live-glasses path only if the remaining gates pass; otherwise keep prerecorded video as the demo fallback and retain the Media Gateway interface so the client or transport implementation can be replaced without changing inference or memory.

## S10 — glasses client

**Decision already made:** the client is pure Kotlin on Android, no Unity, per
[Glasses Client Plan](15-Glasses-Client-Plan.md). This supersedes S01's Unity SDK pin for
the client half; S01's *server* boundary and its remaining device gates stand unchanged.

**Existing result:** five sub-spikes ran with no glasses, no Android toolchain, and no
GN100 — see [results](spikes/glasses-client/RESULTS.md). They found that the hands-free
wake matcher misses 6 of 10 realistic utterances because it anchors to the start of a
transcript, that a read-only console viewer joining a room collapsed the resolution
reaching the gateway from 720p to 320×180, and that LiveKit enforces token expiry with
roughly 60 s of clock-skew leeway.

**Remaining question:** does the LiveKit Android SDK publish acceptably from the X3 Pro?

**SG-C result:** explicit gateway-high/viewer-low subscription requests failed under the
demo topology (25/121 720p frames admitted after viewer join). Publisher-side simulcast
disabled passed 120/120, so the Android client must publish one 720p layer.

**SG-D result:** the ARGF20 runs Android 12 / API 32 on arm64-v8a only, a 1280×480
display, and an ordinary `cyttsp` multitouch transport for the temple pad, so plain Compose
gesture handling reaches the push-to-talk fallback. Two consequences fed straight back into
the client: it reports **two** back-facing cameras and does not support
`cmd media.camera get-camera-ids`, so the world camera is resolved by enumeration rather
than assumed to be `"0"`; and `dumpsys audio` reports **no hardware echo canceller**.

**Remaining gates:** SG-E (SDK on the device — the same gate S01 still carries) and SG-F
(software AEC in a real room). SG-F is now the sharper of the two: with no hardware
canceller, WebRTC's software APM is the only thing keeping the assistant's reply out of its
own microphone.

**Decision:** build against the plan now; SG-E gates the live-glasses demo path. The
prerecorded-video fallback in S01 remains the fallback here.

## S02 — GN100 platform and container preflight

**Question:** Does the physical GN100 support the exact Linux ARM64/CUDA, container, codec, and compiled-extension stack needed by the project?

**Method:**

1. Record OS, kernel, architecture, driver, CUDA, cuDNN, container toolkit, Docker, Compose, disk, and available unified memory.
2. Decode representative glasses audio/video with the intended FFmpeg/TorchCodec stack.
3. Start one minimal GPU container and verify PyTorch CUDA execution.
4. Exercise required compiled packages independently.
5. Build, start, health-check, and stop one representative `linux/arm64` service image.

**Pass gate:** All required base dependencies run natively, exact versions are recorded, and unresolved failures have an owned fallback.

**Stop condition:** A required dependency has no viable ARM64 build or consumes resources that invalidate the planned topology.

## S03 — Parakeet and Kokoro on GN100

**Question:** Which exact Linux ARM64/CUDA artifacts provide the deployment Speech Service while remaining behaviorally compatible with native developer profiles?

**Candidates:** [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for English STT and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for TTS. [Fish Audio S2.1 Pro](https://fish.audio/blog/s2-1-pro-free-api/) is an opt-in cloud comparison only and cannot satisfy the local deployment gate.

**Method:**

- run the shared English golden set on Mac, Windows, and GN100 profiles;
- compare transcript text, timestamps, audio format, pronunciation, and error behavior;
- record cold start, real-time factor, p50/p95 latency, peak memory, and coexistence impact;
- pin source and converted artifact hashes, runtime versions, voice, sample rate, and preprocessing.

**Pass gate:** Both local GN100 adapters pass the service contract and golden set at acceptable latency while leaving agreed memory headroom.

**Stop condition:** A model lacks a stable ARM64/CUDA path; switch only that adapter to an approved local alternative while preserving the Speech Service contract.

## S04 — SAM 3.1 and hand/object events

**Existing result:** The actual architecture diverged from this spike's original premise before S04 itself ran -- see docs/02-Model-Landscape.md's amended "Interaction state machine" section: no hand tracking (motion/rest replaces it), YOLOE plus a pure-numpy tracker as the continuous default, SAM 3.1 downgraded to a selective window-scoped identity verifier. The first real measurement against that actual pipeline is tracker-level: identity-switch rate and latency for both tracker implementations, against the shared golden scenarios. See the [tracker benchmark spike](spikes/tracker-benchmark/README.md) and [results](spikes/tracker-benchmark/RESULTS.md) -- zero identity switches across every scenario, both trackers. SAM 3.1 itself, and recall/false-positive rate against real footage, remain pending on task #47 and task #31 respectively.

**Question:** Can [SAM 3.1](https://github.com/facebookresearch/sam3), plus explicit hand tracking and a temporal state machine, produce reliable placement and pickup candidates from glasses video?

**Method:**

- use the frozen development clips, including occlusion, walking past, similar objects, reconnects, and pickup without placement;
- measure target recall, false positives, identity switches, placement/pickup precision and recall, and candidate-window correctness;
- benchmark selected resolution and sampling rates on the GN100;
- verify authenticated checkpoint access and offline cache behavior.

**Pass gate:** Meet the initial perception/event targets in the Evaluation Plan and produce bounded evidence windows suitable for the verifier.

**Stop condition:** If hand prompting or temporal events fail, keep SAM for objects and substitute a dedicated hand detector or conservative rule-based candidate generator.

## S05 — verifier selection

**Question:** Should the primary event verifier be [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano), a smaller variant, or conservative rules?

**Laptop evidence already recorded:** ModelBest's external `MiniCPM-V-4.5-9B` API returned the required `where_is` tool call and passed the Agent's fixture-backed real-model integration test in 7.35 seconds on 2026-08-11. A subsequent full ADK → Memory fixture → deterministic guard run called the tool, preserved `answer_status="confirmed"`, and passed the guard with the grounded reply. This establishes an explicit laptop Agent profile only; it does not select the GN100 event verifier and does not waive the external-egress warning.

**Method:** Serve Cosmos3-Nano as an isolated OpenAI-compatible vLLM sidecar on the physical GN100, add or exercise the provider adapter without changing the verifier or Memory contracts, and replay identical frozen candidate windows, masks, timestamps, semantic candidates, prompts, schema, and generation settings. Compare event precision/recall, room/surface/relation accuracy, raw and repaired JSON validity, abstention, cold start, p50/p95 latency, peak unified memory, and coexistence with detection, speech, media, and ordinary services. Record the exact checkpoint revision, BF16 runtime, container digest, CUDA stack, model-cache size, and ARM64 result.

Do not substitute Cosmos3-Super by assumption: at 64B with only BF16 officially tested it cannot preserve safe headroom in a shared 128 GB budget. Cosmos3-Nano is the planned 16B tier, but it still passes only after measured physical coexistence. Cosmos3-Edge is also not a laptop fallback because its approximately 9.1 GB of weights exceed the laptop's 8,188 MiB VRAM before runtime allocations.

If Cosmos3 is also proposed for the Agent, separately test OpenAI tool-call parsing, one-and-only-one `where_is` invocation for personal-memory questions, thinking suppression, guarded Memory rewrites, and bounded no-tool general-assistant answers.

**Pass gate:** Select one resident verifier that meets the precision, JSON-validity, latency, ARM64, and complete-workload memory gates while preserving agreed GN100 headroom.

**Stop condition:** If neither model passes, use conservative rules for placement and narrow the demo rather than keeping an unreliable VLM on the critical path.

## S06 — GPT4Scene-style spatial prompting

**Question:** Do [GPT4Scene](https://gpt4scene.github.io/)-style consistent markers and a bird's-eye-view image improve the existing verifier?

**Method:** Compare on the same held-out spatial questions:

1. baseline evidence frames;
2. frames with consistent object IDs and masks;
3. marked frames plus a BEV generated by an available offline geometry source.

Keep the verifier checkpoint and generation settings fixed. Measure room, surface, relation, grounding, JSON validity, p95 latency, and added preprocessing cost.

**Pass gate:** A statistically visible improvement on project clips without degrading event precision or exceeding the latency/memory budget.

**Stop condition:** If markers or BEV add no meaningful lift, retain ordinary evidence frames and do not reproduce the GPT4Scene training pipeline.

## S07 — LingBot-Map

**Question:** Can [LingBot-Map](https://github.com/Robbyant/lingbot-map) produce useful pose and geometry from egocentric glasses video on the GN100?

**Method:**

- first run a short indoor clip, then a 10–15 minute walkthrough with room revisits, blur, occlusion, and texture-poor surfaces;
- test the documented keyframe and windowed modes;
- measure reconstruction continuity, pose drift/collapse, revisit consistency, throughput, peak memory, reset behavior, and evidence export;
- verify PyTorch, CUDA, FlashInfer, and any compiled dependency on Linux ARM64;
- run beside the selected detection, verifier, and speech services.

**Pass gate:** It produces a stable enough map or BEV to improve a labeled project metric while preserving GN100 headroom.

**Stop condition:** Reject the live branch if it cannot build on ARM64, repeatedly collapses on glasses footage, or harms the critical path. Offline use may remain.

## S08 — combined map-assisted verification

Run only if S06 shows that BEV prompting helps and S07 produces a viable map.

Compare the selected verifier with and without LingBot-derived pose/BEV evidence on the same frozen clips. Adopt only if the combination improves last-confirmed-placement or spatial-relation accuracy beyond each individual branch and does not change deterministic pickup invalidation.

## S09 — Stream3D-VLM

**Question:** Does [Stream3D-VLM](https://stream3d-vlm.github.io/) improve room/layout recall or viewpoint-independent reasoning beyond the completed Path A?

Run offline or at low frequency on recorded glasses video. Compare only project-labeled spatial questions, latency, memory, and failure cases. Published benchmark results are context, not acceptance evidence.

**Pass gate:** Measurable project-specific improvement that cannot be obtained more cheaply with selected frames, markers, or ordinary geometry cues.

**Stop condition:** Defer if dependencies, licensing, ARM64 support, or resource use threaten the reliable demo.

## Required records

Each spike report must contain:

- date, owner, machine, and time box;
- source commit and model/checkpoint revisions;
- commands or reproducible harness entry point;
- input fixture identifiers and labels;
- quantitative results with denominators;
- latency and memory measurements;
- failure examples;
- license and access notes;
- decision, constraints, fallback, and follow-up owner.

Do not create a production dependency from a notebook result. Accepted experiments must still pass the contracts and release gates in [Development and Deployment](08-Development-and-Deployment.md).
