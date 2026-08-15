# Four-Person Team Split

Split ownership by subsystem, keep one versioned contract, and make every subsystem runnable against mocks.

## Person 1 — Media-to-observation vision

Owns decoded video ingestion, frame sampling, SAM 3.1 detection/segmentation, short-term tracking, hand masks, interaction state machine, timestamped candidate-window generation, candidate metadata, and evidence-frame selection.

Also owns tracker-epoch behavior after media reconnect, but not WebRTC signaling.

**Deliverable:** observation candidates and minimized evidence using schema v1.

## Person 2 — Spatial verification and evaluation

Owns configured room/zone labels, surface candidates, optional Depth Anything 3 cues, spatial relations, Qwen/Cosmos verifier adapters, explicit `confirmed`/`rejected`/`unverified` results, schema validation, prompt versions, and the shared evaluation harness.

**Deliverable:** validated event fields with per-field confidence, provenance, benchmark results, and failure examples.

## Person 3 — Memory, API, and grounded query

Owns the canonical schema, stable object identity, idempotent ingestion, deterministic reducer, current status, last confirmed placement, history, controlled evidence-window retrieval, evidence metadata, bounded query tools, and the thin Agent layer that verbalizes memory responses. The Agent layer is implemented in `services/agent`; remaining ownership covers physical glasses/GN100 acceptance and model-profile measurement rather than core service construction.

Also owns export, deletion, and retention APIs with Person 4 integrating their user controls.

**Deliverable:** stable APIs and reducer tests, including pickup invalidation and late/duplicate events.

## Person 4 — Glasses, WebRTC, speech, UX, and release

Owns the pure-Kotlin `apps/glasses-x3` client, its pinned LiveKit Android SDK dependency, self-hosted LiveKit configuration, Media Gateway signaling and codec flow, pairing, recording controls, STT/TTS, HUD and response playback, reconnect UX, top-level Compose release orchestration, fallback selection, and demo rehearsal. Person 4 records the exact Android/Gradle dependency versions and proves camera, microphone, display, playback, permissions, codec, and reconnect behavior on the actual glasses.

Person 4 is the integration lead, but each owner supplies and tests their adapter; integration bugs are not delegated to one person.

**Deliverable:** end-to-end demo that communicates confirmed, last-confirmed-only, unknown, and ambiguous answers with time, confidence, and evidence.

## Shared contract

[Data Contract and Memory Semantics](06-Data-Contract.md) is the only canonical event and response format. Do not copy a smaller JSON example into subsystem code or notes.

Every service must:

- declare the schema versions it accepts and emits;
- preserve `observation_id`, `object_id`, timestamps, confidence, and provenance;
- use mocked inputs and contract fixtures in automated tests;
- return explicit unavailable, invalid, or ambiguous results;
- avoid accepting arbitrary evidence paths.

## Integration ownership

| Interface | Provider | Consumer | Contract test owner |
|---|---|---|---|
| WebRTC tracks → decoded frames/audio | Person 4 | Persons 1 and 4 | Persons 1 + 4 |
| Candidate event → verified observation | Person 1 | Person 2 | Persons 1 + 2 |
| Verified observation → trusted state | Person 2 | Person 3 | Persons 2 + 3 |
| Memory response → speech/UI | Person 3 | Person 4 | Persons 3 + 4 |

An interface is complete only when its provider fixture passes in the consumer's test harness.

## Packaging and release ownership

Each subsystem owner owns:

- the service code, Dockerfile, and dependency lock;
- startup, liveness, readiness, and graceful shutdown behavior;
- model manifests and cache expectations where applicable;
- unit, fixture, and contract tests;
- the service's Linux ARM64 build and GN100 defects.

Person 4, as release owner, owns:

- the top-level `compose.yaml`;
- shared CI workflow structure and registry conventions;
- environment configuration and secret injection;
- release manifests, immutable image-digest records, and deployment coordination;
- aggregate readiness, rollback, and demo fallback selection.
- aggregate queue, dropped-frame, evidence-retrieval, verifier-outcome, and latency dashboards or release reports.

The release owner coordinates failures but does not inherit defects inside another owner's service image. Changes to a shared interface, base image, Compose service, persistent volume, port, or secret require review by the affected owners.

## Integration milestones

### First hours

- Connect to the GN100 and glasses.
- Verify codec decode and one selected model.
- Freeze schema v1, enums, example fixtures, and reducer rules.
- Make the Agent answer manually inserted placement and pickup events.
- Confirm the local-only network boundary.

### Middle phase

Integrate the complete prerecorded flow:

```text
video → candidate → verified observation
→ trusted state → query → truthful answer
```

Pass the pickup-without-placement case before adding live streaming.

### Final phase

- Integrate live WebRTC and reconnect behavior.
- Run the held-out test set and choose the primary verifier.
- Verify evidence, deletion, retention, and privacy controls.
- Rehearse at least two fallback levels.
- Freeze features several hours before submission.

## Definition of done

A subsystem is done only when:

- its contract fixtures pass;
- failure and ambiguity paths are demonstrated;
- p50/p95 latency and peak memory are recorded where applicable;
- its configuration, checkpoint, prompt, or schema versions are pinned;
- it works in the prerecorded end-to-end flow.
- if it deploys to the GN100, its Linux ARM64 image builds, becomes ready, and passes its fixtures on the physical workstation.

## Team rule

Vision proposes observations. Verification adds bounded interpretation. Memory turns validated observations into trusted state. The Agent retrieves and explains that state without upgrading uncertainty.

See [Recommended Architecture](01-Recommended-Architecture.md), [Hackathon Stack](03-Hackathon-Stack.md), [Evaluation Plan](04-Evaluation-Plan.md), [Privacy and Security](07-Privacy-and-Security.md), [Development and Deployment](08-Development-and-Deployment.md), and [Spike Plan](09-Spike-Plan.md).
