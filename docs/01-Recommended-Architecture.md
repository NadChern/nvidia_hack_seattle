# Recommended Architecture

## Logical components

```text
Smart Glasses Client
  Pure Kotlin + Jetpack Compose + LiveKit Android SDK + camera + microphone
              ↓ encrypted audio/video
Media Gateway
  signaling, track routing, decoding, buffering
       ├── audio → local Speech Service
       └── video → local Vision Service

Speech Service → transcript → AI Agent
Vision Service → observations → Memory Service
AI Agent → Memory tools → answer → Speech Service → glasses
```

The Smart Glasses Client runs on the glasses. The remaining services run on the local workstation for the MVP. External signaling, TURN, speech, telemetry, and model APIs are disabled by default.

## Components

### 1. Smart Glasses Client

`apps/glasses-x3` captures camera and microphone streams, maintains the WebRTC session, plays synthesized audio, and renders a flat Compose HUD. It is a pure-Kotlin Android app with the LiveKit Android SDK and no Unity or proprietary RayNeo runtime. Keep it thin, pin the exact Android and LiveKit dependencies in Gradle, and publish one non-simulcast 1280×720/15 FPS video layer per SG-C. See [Glasses Client Plan](15-Glasses-Client-Plan.md).

### 2. Media Gateway

Owns WebRTC signaling, ICE/STUN/TURN configuration, codec normalization, audio/video extraction, routing, bounded buffering, and session state. For the local demo, prefer local signaling and direct connectivity; do not silently route media through a hosted TURN service. It can share a process with other services during the hackathon but should remain a clear logical boundary.

Use a self-hosted LiveKit server as the selected MVP implementation. S10 supersedes S01's former Unity client choice with the native LiveKit Android SDK; S01's validated server boundary remains. Capture, playback, permissions, codecs, reconnect behavior, and sustained operation must still pass on the actual glasses in SG-D/E/F.

A local spike validated JWT authentication, raw audio/video subscriptions, bounded frame sampling, return audio, and deliberate disconnect/rejoin cycles. LiveKit remains a transport boundary; vision inference, evidence, and memory stay in application services. See the [LiveKit spike results](spikes/livekit-media-gateway/RESULTS.md).

The Media Gateway holds the only inference LiveKit subscription and relays decoded, sampled, dimension-guarded media to the Vision and Speech services, so no inference service depends on WebRTC. A subscribe-only operator console viewer is the explicit carve-out. That transport and viewer boundary are defined in the [Media Relay Contract](12-Media-Relay-Contract.md); neither is a canonical observation format.

### 3. Speech Service

Treat speech-to-text (ASR/STT) and text-to-speech (TTS) as separate capabilities, even if they run in one service. Add voice activity detection and interruption handling if time permits.

The Speech Service has one stable API but platform-specific local implementations:

- Apple-silicon macOS development uses MLX-compatible Parakeet and Kokoro checkpoints.
- Windows development uses the tested Windows-compatible PyTorch/CUDA implementations, with CPU or the remote GN100 service as an explicit fallback.
- Deployment uses Linux ARM64 and CUDA-compatible Parakeet and Kokoro runtimes on the Acer GN100.

Application code must not import an MLX-, Windows-, or CUDA-specific model runtime directly. Select the backend through configuration and keep transcript, timestamp, synthesis, audio-format, error, and health-check contracts identical. A model that works on a developer laptop is not considered deployment-compatible until the exact Linux ARM64/CUDA artifact passes the GN100 compatibility gate.

### 4. Vision Service

Produces structured observations rather than conversational answers:

```text
frame sampling → object detection → tracking
               → temporal hand/object event candidates
               → semantic room/surface candidates
               → optional geometry cues
               → selective VLM verification
               → observation + evidence
```

Geometry models provide depth and pose cues; they do not supply semantic labels such as `coffee_table`. Semantic entities come from detections, configured zones, or VLM verification.

### 5. Memory Service

The trusted source of truth. It validates observations, resolves stable object identity, handles duplicates and late arrivals, maintains immutable timelines and reduced current state, stores evidence references, and applies confidence and retention rules. Its canonical contract and reducer are defined in [Data Contract and Memory Semantics](06-Data-Contract.md).

### 6. AI Agent / Query Service — implemented

`services/agent` is a concise personal assistant: Nemotron answers ordinary questions directly and selects bounded `where_is` or `start_registration` tools when a request needs personal visual memory. Memory's complete query result remains authoritative whenever a memory tool is used: the deterministic guard rejects unsupported locations, missing uncertainty, lost ambiguity, and empty or oversized replies, and a memory-answer veto returns Memory's canonical `spoken_answer` rather than failing the request. Every completed glasses transcript reaches the Agent without a wake/intent regex; a short content-independent post-reply cooldown prevents return-audio echo recursion. Barge-in remains explicitly out of scope.

The model endpoint is local by default. Any non-loopback language-model endpoint requires `VMA_ALLOW_EXTERNAL_LLM=true` at startup and is reported as external in status and the console. Only transcript text and the complete Memory query response may reach that endpoint—never audio, frames, or evidence media.

### 7. Data Storage

- Relational database: sessions, objects, observations, events, current state
- Local object storage: evidence frames and short clips
- Optional vector index: aliases, visual similarity, and semantic history search
- Retention worker: expiration, user-requested deletion, and derived-data cleanup
- Audit metadata: access, export, and deletion events without raw media

The database and object store use ordinary system memory and storage. They do not require GPU allocation.

## Adopted VSS architecture patterns

Use the [NVIDIA Video Search and Summarization Blueprint](https://docs.nvidia.com/vss/latest/index.html) as an architecture reference, not as the application runtime.

Adopt these patterns:

- Separate real-time perception, downstream verification, and query/search work so a slow VLM or search request cannot block media ingestion.
- Pass timestamped candidate-event metadata between stages; keep frames and clips in controlled evidence storage rather than embedding media in messages.
- Retrieve a bounded evidence window by `session_id`, media epoch, timestamps, and evidence ID before verification.
- Record every verifier outcome as `confirmed`, `rejected`, or `unverified`. Timeout, invalid JSON, unavailable evidence, and model failure produce `unverified`, never implicit confirmation.
- Persist raw candidate metadata, the verifier result, prompt/model versions, latency, and evidence digest so a decision can be reproduced.
- Expose bounded evidence, history, and semantic-search tools through the application API; tools return provenance and authorization-aware references rather than arbitrary filesystem paths.
- Measure per-stage queue depth, dropped frames, candidate counts, verifier outcomes, latency, evidence-store failures, and end-to-end answer status.

Do not add the full VSS stack to the MVP. In particular, VSS does not introduce requirements for DeepStream, NVIDIA NIM, VIOS, Kafka, Redis, Elasticsearch, the VSS Agent, or Kubernetes. LiveKit remains the media gateway, the project Vision Service owns perception and verification, the relational database remains the source of truth, and the Memory Service alone promotes observations into trusted state.

VSS documents that some blueprint connections assume a trusted isolated network and lack built-in TLS, authentication, or rate limiting. Those limitations are not copied into this project; the controls in [Privacy and Security](07-Privacy-and-Security.md) apply to every internal API.

## Memory semantics

The system keeps three distinct facts:

- `current_status`: `confirmed_at_location`, `in_transit`, or `unknown`
- `last_confirmed_placement`: the most recent supported placement event
- `last_seen`: the most recent supported visual observation

A confirmed `picked_up` or `carried` event clears the current location and changes the status to `in_transit`. It does not erase the historical placement. A weak observation is recorded but does not overwrite trusted state.

`in_transit` means the object is still tracked as being carried. If the track is lost or the session ends before a placement, the reducer changes the status to `unknown` while preserving the last confirmed placement as history.

Example:

```text
10:03 keys placed on coffee table
10:05 keys picked up
10:06 no later placement observed

current_status: in_transit
current_location: unknown
last_confirmed_placement: coffee table at 10:03
```

The safe answer is: “I last confirmed the keys on the coffee table at 10:03, but they were picked up afterward and I have not confirmed a new location.”

## Ordering and failure rules

- All timestamps are UTC ISO-8601; the UI renders local time.
- The ingestion API is idempotent.
- Late observations may recompute an object's timeline deterministically.
- A reconnect starts a new media epoch without silently reusing tracker IDs.
- Low-confidence or identity-ambiguous events remain observations and do not become current state.
- If evidence cannot be loaded or verified, the answer is downgraded to unknown or unavailable.
- The Agent never bypasses the Memory Service to answer from a VLM transcript.

## Physical hackathon deployment

Use three logical deployment groups:

1. On the glasses: pure-Kotlin Compose client with the LiveKit Android SDK, capture, WebRTC, HUD, playback, recording indicator, and reconnects
2. On the Acer GN100: containerized LiveKit, Media Gateway workers, Speech, and Vision
3. On the Acer GN100: containerized Agent, Memory, API, database, evidence storage, and retention worker

Docker Compose is the MVP orchestrator on the GN100. The initial logical services are `livekit`, `media-worker`, `speech`, `vision`, `application-memory`, and `database`; a pair may be packaged together when measured latency or hackathon simplicity justifies it. Persist database data, evidence, and the pinned model cache outside disposable container layers.

Native Mac and Windows model development remains supported. Containerized Linux ARM64/CUDA execution on the physical GN100 is the deployment and release gate; a local native test or cross-built image is not sufficient.

This keeps deployment manageable while preserving clean logical boundaries. See the [Architecture Diagram](10-Architecture-Diagram.md), [Hackathon Stack](03-Hackathon-Stack.md), [Team Split](05-Team-Split.md), [Privacy and Security](07-Privacy-and-Security.md), [Development and Deployment](08-Development-and-Deployment.md), and [Spike Plan](09-Spike-Plan.md).
