# Visual Memory Assistant

A local-first wearable assistant that remembers personal objects, answers everyday questions, and
connects a trusted person when human help is needed. RayNeo X3 Pro glasses stream first-person
camera and microphone media to the GN100, where perception, speech, reasoning, and memory run
locally by default.

> *“Where did I leave my keys?”*
>
> *“On the living-room coffee table at 10:42 — but they were picked up afterward and I have not
> confirmed a new location since.”*

That second sentence is the hard part. Being usefully uncertain matters more than sounding
confident.

Built for the NVIDIA Spark Hackathon, Seattle.

## Three assistant experiences

### 1. Personal visual memory

The wearer can register a personal object, such as keys or a wallet, by showing it to the glasses.
The system stores multiple C-RADIO gallery views, resolves that stable identity in later video,
and records confirmed placements in structured Application Memory.

When the wearer asks where an object is, Nemotron calls `where_is(label)`. Application Memory—not
the language model—is authoritative about its current or historical location. Registered objects
and their stable state survive glasses reconnects and new sessions.

### 2. Local personal assistant

Every completed non-empty glasses transcript reaches local NVIDIA Nemotron, which owns intent and
tool selection. It can answer concise general questions directly while using bounded tools for
personal visual memory and registration. Private model reasoning is removed before HUD display or
speech playback, and voice generation is bounded to avoid long apparent stalls.

Hermes profiles, `SOUL.md`, personal text memory, and web search are future improvements tracked in
the [Product Backlog](docs/17-Product-Backlog.md); they are not part of the current runtime.

### 3. Remote human assistant

The wearer can say, “Call my remote assistant.” The Agent calls `call_remote_assistant()` with a
trusted, non-model-visible session identity. A paired helper phone receives the request and joins
the same LiveKit room to see the wearer's camera and speak with them.

After the request succeeds, the Agent allows one fixed acknowledgement and immediately closes its
session audio gate. While assistance is `requested` or `accepted`:

- no new wearer transcript is submitted to Nemotron;
- the Agent's STT socket is closed;
- queued transcripts and in-flight ordinary replies are suppressed; and
- helper media remains outside Vision, Speech, Agent, and Application Memory inference.

Listening resumes with a fresh STT subscription after the helper disconnects or an unanswered
request expires.

## NVIDIA stack

| Role | Technology |
|---|---|
| Agent reasoning and tool routing | NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 through local vLLM |
| Multimodal temporal reasoning | NVIDIA Cosmos 3 Nano |
| Personal-object embeddings | NVIDIA C-RADIOv4-H 653M |
| Speech recognition | NVIDIA Parakeet TDT 0.6B v3 |
| Text-to-speech | Kokoro 82M |
| Glasses media | RayNeo X3 Pro + LiveKit/WebRTC |
| Trusted object state | FastAPI Application Memory with structured observations and evidence |
| Target deployment | Linux ARM64/CUDA on the Acer GN100 |

## Runtime architecture

```text
RayNeo X3 Pro glasses
  ├─ camera ──────> Media Gateway ─> Cosmos + C-RADIO ─> Application Memory
  ├─ microphone ──> Media Gateway ─> Parakeet ─> Nemotron ─> bounded tools
  │                                                        └─> Kokoro ─> glasses
  └─ assist request ────────────────────────────────> paired helper app
                                                        └─ LiveKit human call
```

The language model may decide whether to call a tool, but it cannot create a session identity,
overwrite visual memory, or turn a weak perception into trusted object state. Structured reducers
and deterministic guards decide what the system is allowed to claim.

## Start here

**New to the project? Open [`docs/onboarding.html`](docs/onboarding.html) in a browser.** It explains
the Media Gateway and starts the complete virtual-glasses development stack without requiring
physical glasses or a GN100.

Working with a coding agent? Start with [`docs/13-Dev-Onboarding.md`](docs/13-Dev-Onboarding.md) and
[`AGENTS.md`](AGENTS.md).

### Virtual glasses and console

```bash
./scripts/dev_stack.sh
```

This starts LiveKit, the five Python services, and the Console. Open the URL it prints and use the
Console's **Glasses** panel to publish the development machine's camera and microphone. The Console
also provides Vision, Memory, Speech, Assistant, Enrollment, and Remote Assist controls.

Model selection is platform-aware. CPU-constrained development uses fixtures or stubs rather than
silently choosing an external provider. Detailed macOS, Windows/WSL, and remote-GN100 profiles live
in [Dev Onboarding](docs/13-Dev-Onboarding.md).

### Physical glasses

The pure-Kotlin client is under `apps/glasses-x3`. Pair it with the target Gateway, then it publishes
camera/microphone media, renders transcripts and replies on the HUD, and plays the Gateway's return
audio track. See the [Glasses Client Plan](docs/15-Glasses-Client-Plan.md) and
[Development and Deployment](docs/08-Development-and-Deployment.md).

### Remote helper app

The Expo/React Native helper is under `apps/helper-app`:

```bash
cd apps/helper-app
npm ci
npm start
```

It pairs through the Gateway's existing QR flow, receives pending requests over WebSocket, and
joins an accepted session as the helper participant. A development build is required for the
native LiveKit integration.

## What exists today

| Component | State |
|---|---|
| **Glasses app** (`apps/glasses-x3`) | Working RayNeo client for pairing, camera/microphone publishing, HUD events, and return audio. |
| **Helper app** (`apps/helper-app`) | Working paired mobile client for incoming assist requests and LiveKit human calls. |
| **Console** (`apps/console`) | Working virtual glasses, live video, memory review/reset, enrollment, speech, Agent, and Assist UI. |
| **Media Gateway** (`services/media-gateway`) | Working LiveKit session authority, bounded media relay, return audio, HUD events, and remote-assist lifecycle. |
| **Media contract** (`packages/media-contract`) | Working relay models, client, framing, and provider/consumer fixtures. |
| **Vision** (`services/vision-worker`) | Working Cosmos window reasoner, C-RADIO identity gallery, registration capture, and placed-event promotion. |
| **Application Memory** (`services/application-memory`) | Working durable object registry, evidence/state reducer, cross-session lookup, and honest location answers. |
| **Speech** (`services/speech`) | Working Parakeet STT and Kokoro TTS with real CUDA backends on the GN100 and explicit stubs elsewhere. |
| **Agent** (`services/agent`) | Working local Nemotron/Google ADK orchestration with `where_is`, `start_registration`, and `call_remote_assistant`. |

## Repository layout

```text
apps/
  glasses-x3/       pure-Kotlin RayNeo application
  helper-app/       Expo/React Native remote-helper application
  console/          browser development and operator console
services/           five independently locked FastAPI services
packages/           shared contracts and bounded clients
tools/dev-livekit/  local LiveKit configuration and tooling
docs/               architecture, contracts, privacy, plans, and onboarding
deploy/             release configuration
compose.dev.yaml    local development dependencies
compose.yaml        GN100 release topology
compose.gpu.yaml    real-model GPU overlay
```

## Privacy and safety boundaries

Raw media, transcripts, inference, object memory, and evidence remain on the glasses and trusted
workstation by default. Enabling an external model or search provider is explicit opt-in and must
be disclosed. Remote-assist mode intentionally shares the live room with a paired helper device.

The helper app never enables its camera, and helper tracks are excluded from inference ingest.
However, the tested LiveKit server rejected a server-enforced microphone-only source grant, so the
current helper token temporarily permits any publish source. Client-side camera disabling is not a
security boundary; the strict expected-failure test tracks restoration of server enforcement.

The MVP is a memory aid, not a guarantee of an object's current location and not a safety-critical
medication-management system. It distinguishes current confirmation, historical placement,
ambiguity, and unknown state. See [Privacy and Security](docs/07-Privacy-and-Security.md).

## Documentation

Start with [Overview](docs/00-Overview.md), which indexes the complete documentation set.

- [Recommended Architecture](docs/01-Recommended-Architecture.md)
- [Data Contract and Memory Semantics](docs/06-Data-Contract.md)
- [Privacy and Security](docs/07-Privacy-and-Security.md)
- [Development and Deployment](docs/08-Development-and-Deployment.md)
- [Media Relay Contract](docs/12-Media-Relay-Contract.md)
- [Dev Onboarding](docs/13-Dev-Onboarding.md)
- [Glasses Client Plan](docs/15-Glasses-Client-Plan.md)
- [Main Integration and Remote Assist Plan](docs/16-Main-Integration-and-Remote-Assist-Plan.md)
- [Product Backlog](docs/17-Product-Backlog.md)

## Engineering rules

- Python services use CPython 3.11, uv, FastAPI, Pydantic v2, Ruff, Pyright, and pytest.
- Every deployable service owns its `pyproject.toml`, `uv.lock`, tests, health endpoints, and
  Dockerfile.
- No secrets, raw media, transcripts, or tokens belong in source or logs.
- External model endpoints are rejected unless explicitly enabled.
- Physical GN100 and glasses testing is the final release gate.

```bash
python3 .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
```

## Event

[NVIDIA Spark Hack: Seattle](https://luma.com/spark-hack-seattle?tk=8XBLkn) ·
[DGX Spark](https://build.nvidia.com/spark?utm_source=luma) ·
[NVIDIA Build models](https://build.nvidia.com/models?utm_source=luma) ·
[VSS Spark Playbook](https://build.nvidia.com/spark/vss?utm_source=luma)
