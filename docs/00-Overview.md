# Visual Memory Assistant

## Purpose

Build a privacy-friendly assistant for people with memory difficulties. Smart glasses capture first-person audio/video; a local workstation turns important visual events into searchable memories.

The core demo is:

1. The user places keys on a table.
2. The system detects the object, action, room, surface, time, and evidence frame.
3. The user asks, “Where did I leave my keys?”
4. The system returns the last confirmed location with supporting visual evidence.

If the keys were later picked up without a confirmed placement, the system must say that the earlier location is only the last confirmed placement. It must not present that stale location as current.

## Core conclusion

Do not rely on one large VLM to watch and remember the entire stream. Use a layered pipeline:

```text
WebRTC video/audio
        ↓
Fast perception: detection, tracking, hand/object motion
        ↓
Selective geometry and VLM verification
        ↓
Structured event memory + evidence files
        ↓
Deterministic retrieval and conversational answer
```

Models perceive and propose; structured memory decides what the system claims to remember.

## Recommended notes

**New here? Start with [Dev Onboarding](13-Dev-Onboarding.md)** — a fresh clone to media flowing on your machine, with no glasses and no GN100.

- [Recommended Architecture](01-Recommended-Architecture.md) — logical and deployable system design
- [Model Landscape](02-Model-Landscape.md) — model roles, maturity, and risks
- [Hackathon Stack](03-Hackathon-Stack.md) — practical critical path and fallbacks
- [Evaluation Plan](04-Evaluation-Plan.md) — dataset, metrics, and acceptance targets
- [Team Split](05-Team-Split.md) — ownership for a four-person team
- [Data Contract and Memory Semantics](06-Data-Contract.md) — canonical observations, state transitions, and query responses
- [Privacy and Security](07-Privacy-and-Security.md) — trust boundary, retention, access, and user controls
- [Development and Deployment](08-Development-and-Deployment.md) — native development, containers, CI/CD, GN100 releases, and rollback
- [Spike Plan](09-Spike-Plan.md) — required compatibility gates, research experiments, dependencies, and stop conditions
- [Architecture Diagram](10-Architecture-Diagram.md) — runtime services, evidence flow, trust boundary, storage, and optional spatial branch
- [Engineering Standards](11-Engineering-Standards.md) — mandatory Python service stack, repository skill, scaffolding, validation, and exceptions
- [Media Relay Contract](12-Media-Relay-Contract.md) — the Media Gateway to Vision/Speech transport: framing, media epochs, sampling, and lifecycle signals
- [Agent Laptop Testing](14-Agent-Laptop-Testing.md) — safe MiniCPM and OpenRouter external-LLM evaluation profiles
- [Glasses Client Plan](15-Glasses-Client-Plan.md) — the pure-Kotlin RayNeo X3 Pro client: scope, gaps, milestones, and risks

## MVP boundary

- 3–5 personal objects: keys, wallet, phone, medication, remote
- 2–3 rooms or zones
- Perception events: observed, picked up, carried, placed
- Derived memory: current status, last seen, and last confirmed placement
- One main query: “Where is my ___?”
- Local-only inference, signaling, speech, and evidence storage by default
- Native Mac and Windows development with platform-compatible model adapters
- Docker Compose deployment on the GN100 using tested Linux ARM64/CUDA images
- Answer includes status, location when confirmed, timestamp, confidence, and an evidence frame
- One authorized user and one trusted workstation for the demo

## Safety boundary

The MVP is a memory aid, not a guarantee of an object's current location and not a safety-critical medication-management system. It reports stored evidence, distinguishes confirmed from unknown state, and abstains when object identity or location is ambiguous.

## Privacy boundary

“Local” means that raw media, transcripts, observations, models, memory, and evidence stay on the glasses and the trusted workstation. Any cloud speech, signaling, TURN relay, telemetry, or model API is outside that boundary and must be disabled by default and disclosed when enabled. See [Privacy and Security](07-Privacy-and-Security.md).

## Success criteria

The system should recognize a target object, capture placement and invalidation events, store its last confirmed placement and current status, answer a natural-language query, and avoid confident or stale answers without supporting evidence.
