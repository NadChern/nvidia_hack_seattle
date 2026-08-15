# Hackathon Stack

## Critical path

### Path A — reliable demo

```text
Smart glasses or prerecorded video
        ↓
local WebRTC + Media Gateway
        ↓
SAM 3.1 detection and tracking
        ↓
hand/object state machine
        ↓ candidate event window
optional Depth Anything 3 geometry cues
        ↓
schema-validated Qwen3-VL verification
        ↓
deterministic memory reducer + evidence
        ↓
memory query → grounded text/TTS answer
```

`nvidia/Cosmos3-Nano` (16B, BF16) is the GN100 challenger for the verifier interface. It is the intended larger Cosmos3 tier for the 128 GB unified-memory workstation; `Cosmos3-Super` (64B) is not the default because BF16 weights would leave no safe margin for KV caches, detection, speech, media, database, and the operating system. Select one primary verifier before full integration rather than keeping both on the live path, and treat physical GN100 coexistence measurements—not parameter count—as the capacity decision.

### Path B — innovation experiment

```text
Recorded glasses stream → Stream3D-VLM-4B → incremental spatial representation
```

Use Path B to demonstrate research value only after Path A independently passes the demo acceptance test.

## Runtime topology

### Glasses

- pure-Kotlin Jetpack Compose HUD, capture, and playback
- camera and microphone permissions
- WebRTC publisher/subscriber using the pinned LiveKit Android SDK
- one non-simulcast 1280×720/15 FPS video layer
- recording indicator, pause/resume, and reconnect controls

### Acer GN100

- Media/Inference process group: self-hosted LiveKit signaling/SFU, raw-track workers, bounded buffers, speech, detection, tracking, geometry, verifier
- Application process group: API, memory reducer, query tools, database, controlled evidence store, retention worker

No external service is part of the default path. Record and review network connections before the demo.

The deployable speech target is the Acer GN100 running Linux ARM64 with CUDA. Parakeet STT and Kokoro TTS must use Linux ARM64/CUDA-compatible runtimes and pinned artifacts in this environment. Passing on macOS/MLX or Windows/CUDA does not satisfy the deployment gate.

Self-hosted LiveKit and its native Android SDK are the selected MVP media stack. S10 supersedes the earlier Unity-client choice while preserving S01's executable [gateway spike](spikes/livekit-media-gateway/README.md), which passed local media and rejoin tests. Release acceptance still requires the pinned Android dependency on the actual glasses and the server/workers on Linux ARM64 on the physical GN100.

## Local development profiles

Developers should run the compatible local implementation for their machine while preserving the same service boundary:

- Apple-silicon Mac: MLX-compatible Parakeet and Kokoro artifacts.
- Windows with NVIDIA GPU: the tested PyTorch/CUDA Parakeet and Kokoro implementations.
- Unsupported or resource-constrained laptop: connect to the GN100 Speech Service instead of silently selecting a cloud API.

For the temporary laptop Agent evaluation, ModelBest's `MiniCPM-V-4.5-9B`
OpenAI-compatible API is an explicit opt-in profile. It has passed the Agent's
fixture-backed ADK tool-call and deterministic-guard integration check and avoids loading another model into
the 8 GB GPU. The measured 8 GB launcher uses fixture Vision while retaining
real CUDA Parakeet STT and Kokoro TTS. MoGe warmup remains outside this laptop's
safe coexistence profile; the earlier attribution of a later failure to Kokoro
was superseded after the issue was traced to a Console reload bug. The external
Agent profile is never selected automatically: set
`VMA_ALLOW_EXTERNAL_LLM=true`, and expect the transcript plus complete Memory
tool response to leave the local trust boundary. The exact command and privacy
warning are in [`services/agent/README.md`](../services/agent/README.md).

For the GN100 verifier evaluation, serve Cosmos3-Nano from an isolated local
vLLM sidecar and keep the verifier contract unchanged. The current verifier
uses Ollama's `/api/chat` request shape, so switching to Cosmos requires an
OpenAI-compatible provider adapter; it is a planned configuration boundary,
not a claim that changing only `VMA_VLM_MODEL` works today. Pin the checkpoint,
container, runtime, precision, prompt, and adapter after the physical spike.

Select the speech backend through configuration. Code outside the Speech Service must not depend on MLX, PyTorch, CUDA, operating-system paths, or a particular checkpoint layout. All profiles must expose the same request/response schema, audio formats, timestamps, error behavior, and health checks.

Use one shared English golden set to compare developer and deployment profiles. Record the exact source checkpoint, conversion revision or hash, runtime and package versions, precision, voice preset, resampling settings, cold-start time, latency, and peak memory. Runtime-specific artifacts are acceptable; undocumented behavioral drift is not.

## Container and orchestration policy

Native Mac and Windows model workflows are supported and do not have to run in Docker. Every service deployed to the GN100 must provide a Linux ARM64 container path, locked dependencies, health behavior, fixtures, and a pinned model manifest when applicable.

Deploy the GN100 with Docker Compose using the logical services `livekit`, `media-worker`, `speech`, `vision`, `application-memory`, and `database`. Store database data, evidence, and model artifacts in controlled persistent volumes. Grant GPU access only to inference services and publish only the trusted-LAN ports required by LiveKit and the user-facing API.

Pull-request CI runs unit, reducer, contract, configuration, and mock-adapter tests and builds affected `linux/arm64` images. Deployment to the shared GN100 is manual, uses immutable image digests, runs the physical hardware acceptance suite, and preserves the previous known-good release for rollback.

Kubernetes is deferred for the single-node MVP. The complete policy and the conditions for revisiting Kubernetes are in [Development and Deployment](08-Development-and-Deployment.md).

## Runtime guidance

- Run detection at roughly 2–5 FPS and let tracking bridge frames when validated.
- Treat reconnects as new tracking epochs.
- Treat a changed LiveKit camera track SID as a new tracking epoch.
- Track hands explicitly and generate temporal event windows; a single-frame hand overlap is not a placement.
- Reject unexpected transition-frame dimensions before frames enter the inference queue.
- Use the VLM only for candidate events, ambiguity, or selected user questions.
- Run depth only when it improves held-out event or relation accuracy.
- Keep only the selected verifier resident during the live demo.
- Budget unified memory for model weights, KV caches, video buffers, speech, database, and operating-system headroom.
- Do not allocate GPU resources to the relational database.
- Validate ARM64, CUDA, Python, PyTorch, serving runtime, video codecs, and compiled dependencies on the GN100 before feature integration.

## Compatibility preflight

Record the result of every check:

1. Confirm DGX OS, ARM64, driver, CUDA, free disk, and available unified memory.
2. Decode a representative glasses recording with the exact runtime codec stack.
3. Run each selected checkpoint in isolation and record cold-start time, peak memory, and p50/p95 latency.
4. Run detection, verifier, and speech together under the planned input rate.
5. Confirm checkpoint access and cache all required weights before the event.
6. Pin repository commits, model revisions, Python packages, container digests, prompts, and licenses.
7. Test local signaling, ICE behavior, and reconnect with external network access disabled.
8. Execute ten representative clips end to end before adding Path B.

## Storage and APIs

Minimum endpoints:

```text
POST   /sessions
DELETE /sessions/{session_id}
POST   /observations
GET    /objects/{object_id}/state
GET    /objects/{object_id}/last-confirmed-placement
GET    /objects/{object_id}/history
GET    /objects/{object_id}/export
GET    /evidence/{evidence_id}
DELETE /objects/{object_id}/memories
POST   /agent/query
POST   /speech/synthesize
```

`POST /observations` requires an idempotency key and the canonical versioned payload. The server assigns evidence storage paths and ingestion timestamps. Object aliases are resolved to stable IDs before state promotion.

The object-state response distinguishes:

- current confirmed location;
- in-transit or unknown current status;
- last confirmed placement;
- last seen observation;
- ambiguous object identity.

Use a relational database as the source of truth. Store minimized evidence images or clips locally. Add embeddings only for enrollment, aliases, visual similarity, or semantic history retrieval. Follow [Data Contract and Memory Semantics](06-Data-Contract.md) and [Privacy and Security](07-Privacy-and-Security.md).

## VSS-inspired pipeline rules

Apply the selected [NVIDIA VSS](https://docs.nvidia.com/vss/latest/index.html) patterns inside the existing services:

- The media worker emits bounded, timestamped candidate windows and stores media through the controlled evidence API.
- The Vision Service retrieves the candidate window, returns `confirmed`, `rejected`, or `unverified`, and records model, prompt, latency, and failure metadata.
- The Memory Service receives validated observations only; `rejected` and `unverified` candidates remain diagnostic history and cannot silently promote state.
- The application exposes evidence and history retrieval as bounded tools with authorization and provenance.
- Queue depth, dropped frames, evidence retrieval, verifier outcomes, and latency are observable release metrics.

Do not deploy the VSS blueprint or add DeepStream, NIM, VIOS, Kafka, Redis, Elasticsearch, or the VSS Agent to Path A. Revisit an individual component only through a separately approved spike with a specific project gap and measurable acceptance criteria.

## Fallback levels

1. Live glasses + live inference
2. Prerecorded video + live inference
3. Precomputed observation candidates + live verifier and memory
4. Validated observation JSON + live memory and query
5. Fully scripted demo data, visibly labeled as scripted

Test every level before the presentation. A fallback must preserve the same data contract and truthful answer semantics.

## Suggested setup sequence

1. Confirm glasses connectivity and record test clips.
2. Verify one selected model on the GN100 and complete the compatibility matrix.
3. Freeze schema v1 and implement reducer tests.
4. Make the Agent answer from manually inserted placement and pickup records.
5. Integrate prerecorded video end to end.
6. Add live WebRTC and optimize latency.
7. Validate privacy controls, deletion, and network isolation.
8. Freeze features and rehearse every fallback mode.

## Nonfunctional targets

- Object-observation latency: p95 under 2 seconds
- Voice response after final transcript: p95 under 5 seconds, with a target under 3 seconds
- No current-location claim after a confirmed pickup without a later placement
- No confident answer without stored, loadable evidence
- Local-only default processing and configurable retention
- Recovery after WebRTC interruption without tracker-ID reuse
- Idempotent observation ingestion
- Session deletion removes metadata, evidence, embeddings, and caches in scope

## Feature-freeze gate

Do not add Path B or new object classes until:

- the prerecorded Path A passes the acceptance checklist;
- reducer and query tests pass;
- the selected verifier meets the precision and latency gate;
- at least two fallback levels work;
- the privacy checklist passes.

See [Recommended Architecture](01-Recommended-Architecture.md), [Evaluation Plan](04-Evaluation-Plan.md), [Team Split](05-Team-Split.md), [Development and Deployment](08-Development-and-Deployment.md), and [Spike Plan](09-Spike-Plan.md).
