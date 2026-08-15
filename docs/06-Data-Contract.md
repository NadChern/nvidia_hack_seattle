# Data Contract and Memory Semantics

This document is the source of truth for observations, trusted memory state, and query responses. Other notes should link here instead of copying a smaller event schema.

## Design rules

1. Perception produces immutable observations with evidence and provenance.
2. The Memory Service validates observations and reduces them into trusted state.
3. A model confidence score is evidence, not permission to overwrite current state.
4. A confirmed pickup invalidates the previous current location even when no later placement is observed.
5. `last_confirmed_placement`, `last_seen`, and `current_status` are different facts.
6. `last_seen` is derived metadata, not an action emitted by perception.
7. Reprocessing the same observation must be idempotent.
8. A candidate event is not a trusted observation until its required verification succeeds.

## Candidate verification boundary

> **Status: accepted and implemented.** `services/vision-worker` produces `CandidateEvent`s from its interaction/rest state machine and verifies them with a deterministic rule-based verifier (`verify/rules.py`) by default -- itself [S04](09-Spike-Plan.md)'s own named fallback, built first rather than as a contingency. Person 2 replaces that module with a VLM adapter (Qwen3-VL) behind the same `Verifier` interface; nothing else in the pipeline changes. `hand` candidates are part of the contract below but are never populated by this implementation -- see [Model Landscape](02-Model-Landscape.md)'s "Interaction state machine: rest, not hands" for why the decision that matters turned out not to need one.

The candidate-event record is an internal Vision Service contract, not the canonical Memory Service payload. It identifies:

- `session_id`, media epoch, and session-scoped track IDs;
- candidate action and bounded start/end timestamps;
- object, hand, room, and surface candidates;
- controlled evidence IDs and digests;
- detector, state-machine, geometry, and pipeline versions.

The verifier returns one explicit result:

- `confirmed`: the evidence supports the candidate and it may be converted into a canonical observation, subject to schema, identity, confidence, and evidence validation;
- `rejected`: the evidence contradicts the candidate;
- `unverified`: evidence is missing, the verifier timed out or failed, JSON is invalid after the allowed repair, or the result is inconclusive.

Rejected and unverified candidates may be retained as privacy-scoped diagnostic records, but they must not be translated into a trusted `placed`, `picked_up`, or `carried` observation. Candidate and verifier records include prompt/model revisions, latency, reason code, and evidence digest. This is the VSS-inspired verification boundary; it does not add a VSS runtime dependency.

The executable form of this boundary is `packages/vision-contract` (`CandidateEvent`, `EvidenceWindow`, `VerifierResult`). It is a separate package from this document's `Observation` envelope below, deliberately: a candidate is Vision-internal and never reaches the Memory Service directly. Only a `confirmed` `VerifierResult` authorizes the Vision Service to construct an `Observation` from the corresponding candidate.

The candidate-event record:

```json
{
  "schema_version": "1.3",
  "candidate_id": "cand_01JABCEXAMPLE00000000000",
  "session_id": "sess_01JAB...",
  "device_id": "glasses-01",
  "media_epoch_id": "TR_VCabc123",
  "track_id": "track-42",
  "label": "keys",
  "action": "placed",
  "window": {
    "window_started_at": "2026-07-29T17:42:11.240Z",
    "window_ended_at": "2026-07-29T17:42:14.240Z",
    "frame_count": 72,
    "evidence_ids": []
  },
  "object_candidate": {
    "label": "keys",
    "confidence": 0.91,
    "box": {"x_min": 0.41, "y_min": 0.52, "x_max": 0.49, "y_max": 0.58},
    "centroid": {"x": 0.45, "y": 0.55},
    "depth_m": 0.62
  },
  "hand_candidate": null,
  "room_candidate": "living_room",
  "surface_candidate": "coffee_table",
  "world_point": {"x": 1.12, "y": -0.34, "z": 2.05},
  "detector": {"name": "yoloe-11s-seg", "checkpoint": "yoloe-11s-seg.pt", "revision": "rev-1"},
  "tracker": {"name": "botsort", "checkpoint": "botsort.yaml", "revision": "rev-1"},
  "depth_model": {"name": "moge-2", "checkpoint": "Ruicheng/moge-2-vitl-normal", "revision": "rev-1"},
  "state_machine_version": "vision-stability-v1",
  "pipeline_version": "vision-pipeline-v1"
}
```

`hand_candidate` is nullable and stays null in the current implementation -- the interaction state machine decides `placed`/`picked_up`/`carried` from whether the object itself is at rest or moving through the world, not from hand contact. The field is kept rather than removed so a future owner who adds hand detection has a contract-compatible slot to fill.

The verifier result:

```json
{
  "candidate_id": "cand_01JABCEXAMPLE00000000000",
  "outcome": "confirmed",
  "reason_code": "stable_for_dwell_period",
  "latency_ms": 4.2,
  "verifier": {"name": "rules", "checkpoint": "n/a", "revision": "rev-1"},
  "prompt_version": null,
  "occurred_at": "2026-07-29T17:42:14.310Z"
}
```

## Observation envelope

The canonical ingestion payload is:

```json
{
  "schema_version": "1.2",
  "observation_id": "obs_01JABC...",
  "idempotency_key": "glasses-01/session-17/event-42",
  "session_id": "sess_01JAB...",
  "device_id": "glasses-01",
  "media_epoch_id": "TR_VCabc123",
  "object": {
    "object_id": "object-keys-01",
    "label": "keys",
    "track_id": "track-42"
  },
  "event": {
    "action": "placed",
    "source": "vision_pipeline",
    "occurred_at": "2026-07-29T17:42:11.240Z",
    "window_started_at": "2026-07-29T17:42:09.900Z",
    "window_ended_at": "2026-07-29T17:42:12.100Z"
  },
  "location": {
    "room": "living_room",
    "surface": "coffee_table",
    "relation": "on",
    "description": "beside the laptop"
  },
  "confidence": {
    "event": 0.91,
    "identity": 0.94,
    "room": 0.88,
    "surface": 0.90,
    "relation": 0.82
  },
  "evidence": [
    {
      "evidence_id": "evidence-obs-01JABC",
      "captured_at": "2026-07-29T17:42:11.100Z",
      "frame_index": 1242,
      "media_type": "image/jpeg",
      "sha256": "hex-encoded-sha256"
    }
  ],
  "provenance": {
    "detector": {
      "name": "sam-3.1",
      "checkpoint": "exact-checkpoint-id",
      "revision": "repository-commit-or-model-revision"
    },
    "geometry": {
      "name": "depth-anything-3",
      "checkpoint": "exact-checkpoint-id",
      "revision": "repository-commit-or-model-revision"
    },
    "verifier": {
      "name": "qwen3-vl-8b-instruct",
      "checkpoint": "Qwen/Qwen3-VL-8B-Instruct",
      "revision": "model-revision"
    },
    "prompt_version": "placement-verifier-v1",
    "pipeline_version": "vision-pipeline-v1"
  }
}
```

The server stamps `ingested_at`, assigns the stored evidence path, and records validation results. Clients must not submit arbitrary local file paths.

## Media epochs and the scope of `track_id`

`media_epoch_id` is a new optional, nullable, top-level field carrying the LiveKit **track SID** the observation was derived from. It is additive, so this is schema **1.1** and a consumer pinned to 1.0 is unaffected.

**`object.track_id` is unique only within `(session_id, media_epoch_id)`. Consumers must not join tracker IDs across epochs.**

The [S01 spike](spikes/livekit-media-gateway/RESULTS.md) established why. Across three deliberate rejoin cycles with an unchanged participant identity, every rejoin produced new track SIDs. Identity is therefore not a usable boundary and the SID is. A Wi-Fi blip mid-session silently restarts the vision tracker's numbering, so `track-42` before the blip and `track-42` after it are different physical objects. Without this field, joining them is indistinguishable from a correct join, and the resulting memory is wrong in a way no confidence score reflects.

This makes [Recommended Architecture](01-Recommended-Architecture.md)'s *"a reconnect starts a new media epoch without silently reusing tracker IDs"* and [Model Landscape](02-Model-Landscape.md)'s *"tracker IDs are scoped to a media epoch and must not survive reconnects"* mechanically checkable rather than advisory.

The Media Gateway announces each epoch on the relay so consumers know when to reset; see [Media Relay Contract](12-Media-Relay-Contract.md).

`media_epoch_id` may be `null` for an observation with no media provenance — a manual correction, a backfill, or a test fixture. It must not be null for an observation derived from a video frame.

## Session identity

`session_id` is minted by the **Media Gateway**, in the format `sess_<ULID>`.

The gateway is the only component present at session start: a session begins when a device requests a token, which is a gateway operation. Nothing else can observe that moment, so nothing else can name it without inventing a second identifier that must then be reconciled.

The Memory Service owns session **persistence and deletion**. Minting and owning are deliberately split: the gateway holds no durable state and must not be the authority on what a session *was*, only on when one started.

When `VMA_SESSION_REGISTRY_URL` is configured, the gateway registers the session with Memory first and adopts the identifier Memory returns. Until then it uses its own. That hands authority over with a configuration change and no code change on either side.

ULIDs are lexicographically sortable by creation time, so a session listing sorts correctly without a join.

## Required enums and null behavior

### Event actions

- `observed`
- `picked_up`
- `carried`
- `placed`

The Media Gateway or Memory Service may also emit lifecycle signals:

- `track_lost`
- `session_ended`

An unknown action is rejected. `last_seen` and `current_location` are derived fields, not event actions. The `source` field distinguishes perception events from lifecycle signals.

Lifecycle signals do not use the observation envelope; see [Lifecycle signals](#lifecycle-signals) below.

### Current status

- `confirmed_at_location`
- `in_transit`
- `unknown`

### Location relations

- `on`
- `in`
- `under`
- `beside`
- `in_front_of`
- `behind`
- `unknown`

Unknown room, surface, or relation fields must be `null` or the explicit `unknown` enum where allowed. They must never be filled with a guessed label to satisfy the schema.

`object.object_id` may be `null` at initial ingestion when identity is unresolved. The label and session-scoped `track_id` remain required. Only observations resolved to one stable object may be promoted to trusted state.

## Validation and promotion

The ingestion API must:

- validate the schema version, enums, UTC timestamps, confidence range, and evidence metadata;
- reject duplicate `observation_id` values with different content;
- return the original result for a repeated `idempotency_key`;
- resolve aliases to a stable `object_id`, or mark the observation ambiguous;
- validate verifier JSON before it reaches the reducer;
- retain low-confidence observations for history without promoting them to trusted current state;
- store evidence through a controlled media API and verify its digest.

Promotion thresholds are configuration, not model constants. Record the threshold set used for every evaluation run.

## Deterministic state reducer

Process accepted observations by `occurred_at`, with a deterministic tie-breaker. Late events may recompute the affected object's timeline but must not silently reorder history.

| Accepted observation | State transition |
|---|---|
| High-confidence `placed` with identity and evidence | Set `current_status=confirmed_at_location`; set current location and `last_confirmed_placement` |
| High-confidence `picked_up` | Set `current_status=in_transit`; clear current location; preserve the previous placement only as history |
| High-confidence `carried` | Keep `current_status=in_transit`; clear current location |
| `observed` without interaction | Update `last_seen`; do not create a placement event |
| `track_lost` or `session_ended` while in transit | Set `current_status=unknown`; preserve the last confirmed placement as history |
| Weak, ambiguous, or unsupported observation | Preserve trusted state; record the observation and ambiguity |
| Conflicting high-confidence observation | Apply configured precedence, retain both records, and flag the conflict for evaluation |

The reducer must be covered by table-driven tests for duplicates, late arrivals, reconnects, identity ambiguity, pickup without placement, and conflicting observations.

## Lifecycle signals

> **Status: accepted and implemented.** The Memory Service accepts this envelope at `POST /v1/lifecycle` and performs the fan-out described below. The three questions this section originally posed are answered at the end.

### The contradiction

This document names the Media Gateway as an authorized emitter of `track_lost` and `session_ended`, and the reducer table treats `track_lost` as **per-object**: *"`track_lost` or `session_ended` while in transit → set `current_status=unknown`"*.

**A transport component cannot know about objects.** The gateway relays decoded frames. It has never run a detector, holds no `object_id`, and by design must not — the whole point of the relay boundary is that inference lives elsewhere. It can report only that a track went away. As written, the only emitter of `track_lost` cannot produce a well-formed one.

### Proposed envelope

A lifecycle signal is **not an observation**. It has no `object`, `location`, `confidence`, or `evidence`, because the emitter observes none of those. `scope` carries the blast radius.

```json
{
  "schema_version": "1.0",
  "signal_id": "lc_01JABC...",
  "idempotency_key": "glasses-01/sess_01JAB/TR_VCabc123/track_lost",
  "session_id": "sess_01JAB...",
  "device_id": "glasses-01",
  "signal": {
    "action": "track_lost",
    "source": "media_gateway",
    "occurred_at": "2026-07-29T17:45:02.100Z",
    "reason": "track_unsubscribed"
  },
  "scope": {
    "media_epoch_id": "TR_VCabc123",
    "object_id": null,
    "track_id": null
  },
  "provenance": {
    "component": "media-gateway",
    "version": "0.1.0",
    "protocol_version": "media-relay/1.0"
  }
}
```

### Scope semantics

| `scope` | Applies to |
|---|---|
| `object_id` set | that one object |
| `media_epoch_id` set, `object_id` null | every object whose `in_transit` state originated in that epoch |
| neither set, `session_id` only | every `in_transit` object in the session |

Memory performs the fan-out, because Memory is the only component that knows which objects were in transit. The reducer transition itself is unchanged — `in_transit` becomes `unknown`, the last confirmed placement is preserved as history. Only the addressing changes.

`reason` is constrained by `action`:

| `action` | Permitted `reason` |
|---|---|
| `track_lost` | `track_unsubscribed`, `participant_disconnected`, `room_disconnected`, `session_ended`, `gateway_shutdown` |
| `session_ended` | `participant_disconnected`, `room_disconnected`, `session_deleted`, `session_ttl_expired`, `gateway_shutdown` |

These are the executable enums in `packages/media-contract`; when this document and that package disagree, one of them is a bug.

The `idempotency_key` is deterministic — `{device_id}/{session_id}/{media_epoch_id}/{action}` — so a gateway that restarts mid-teardown and re-sends cannot double-apply.

### The decisions, as taken

1. **A separate endpoint.** `POST /v1/lifecycle`, not an observation with null fields. An observation carrying no object would fail the promotion rules, and widening those rules to admit it would weaken them for every real observation.
2. **`media_epoch_id`-scoped fan-out, with session scope available.** An epoch-scoped signal reaches every object whose identity was established in that epoch; omitting the epoch applies it to the whole session. Memory performs the fan-out, because Memory is the only component that knows which objects those are.
3. **The reducer's table-driven suite owns the fan-out test**, and it lives in Memory's suite as `tests/test_reducer.py` and `tests/test_repository.py`.

One rule worth stating explicitly, because it is not obvious from the reducer table: a `track_lost` for an object that is **confirmed at a location is ignored**. A camera disconnect is not evidence that an object moved, and downgrading a good answer because the network blinked discards correct memory for nothing. The transition only applies while an object is `in_transit`.

## Object registry contract

Personal-object registration versions independently from observations under
`OBJECT_REGISTRY_SCHEMA_VERSION = "1.0"`. It does not bump the observation schema:
`object.object_id` was already nullable and a resolved producer already had a compatible place
to carry a stable id.

Memory mints `object_id`. Registered objects and views are durable, have no foreign key to a
session, and therefore survive reconnects and ordinary session retention. Each selected view
stores its controlled crop plus two little-endian float32 pooled vectors. `embedder_id`,
`pooling`, and `dim` make a stale or incompatible gallery fail loudly rather than compare vectors
from different domains.

```json
{
  "schema_version": "1.0",
  "registry_version": 2,
  "unchanged": false,
  "objects": [
    {
      "schema_version": "1.0",
      "object_id": "object_01JABC",
      "label": "keys",
      "idempotency_key": "registration/sess_01JAB/keys/1",
      "created_at": "2026-07-29T17:42:11.240Z",
      "updated_at": "2026-07-29T17:42:17.240Z",
      "registry_version": 2
    }
  ],
  "views": [
    {
      "schema_version": "1.0",
      "view_id": "view_01JABC",
      "object_id": "object_01JABC",
      "view_index": 0,
      "quality": {
        "detection_confidence": 0.94,
        "box_area_fraction": 0.31,
        "sharpness_score": 1.12,
        "mask_box_ratio": 0.76,
        "quality_score": 0.91,
        "angular_velocity_rad_s": null
      },
      "embedder_id": "nvidia/C-RADIOv4-SO400M@c0457f5d",
      "pooling": "summary+mask-weighted-spatial-v1",
      "dim": 2,
      "summary": [0.25, -0.5],
      "pooled_spatial": [0.125, -0.25],
      "crop_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "crop_media_type": "image/jpeg",
      "crop_reference": "/v1/objects/object_01JABC/views/view_01JABC/crop",
      "created_at": "2026-07-29T17:42:17.240Z",
      "registry_version": 2
    }
  ]
}
```

`registry_version` is monotonic across creates, view writes, and deletions. A gallery request with
`since_version` equal to the current version returns `unchanged=true` and empty object/view lists;
an older version receives a full snapshot. Serving a last-known-good snapshot while Memory is
unreachable is a Vision policy, not permission for Memory to return stale data.

View insertion is idempotent on `(object_id, view_index, crop_sha256)`, bounded by configured crop,
dimension, and view-count limits. Deleting an enrolled object removes its registry rows,
embeddings, and durable crop subtree. Session deletion removes only that session's event history;
it must not erase a registered object or another session's state for that stable id.

## Trusted object state

```json
{
  "object_id": "object-keys-01",
  "current_status": "unknown",
  "current_location": null,
  "current_event_id": "event-track-lost-44",
  "state_reason": "picked_up_then_track_lost",
  "invalidated_at": "2026-07-29T17:45:02.100Z",
  "last_confirmed_placement": {
    "event_id": "event-placement-42",
    "occurred_at": "2026-07-29T17:42:11.240Z",
    "room": "living_room",
    "surface": "coffee_table",
    "relation": "on",
    "evidence_id": "evidence-obs-01JABC"
  },
  "last_seen": {
    "occurred_at": "2026-07-29T17:45:01.900Z",
    "room": "living_room",
    "evidence_id": "evidence-obs-01JABD"
  },
  "updated_at": "2026-07-29T17:45:02.300Z"
}
```

## Query response contract

```json
{
  "object_id": "object-keys-01",
  "answer_status": "last_confirmed_only",
  "current_status": "unknown",
  "current_location": null,
  "last_confirmed_placement": {
    "occurred_at": "2026-07-29T17:42:11.240Z",
    "room": "living_room",
    "surface": "coffee_table",
    "relation": "on",
    "evidence_id": "evidence-obs-01JABC",
    "evidence_url": "/v1/evidence/evidence-obs-01JABC",
    "evidence_media_type": "image/jpeg"
  },
  "last_confirmed_placement_confidence": 0.91,
  "spoken_answer": "I last confirmed the keys on the living-room coffee table at 10:42, but they were picked up afterward and I have not confirmed a new location."
}
```

`evidence_url` is present **only when the bytes are actually retrievable**. Retention deletes evidence files while the rows that reference them survive, and a row pointing at a deleted file is indistinguishable from a valid one — so the field is populated from a filesystem check, not from the reference. A URL that 404s is worse than no URL: it looks like evidence right up until someone follows it.

The field is absent from stored trusted state on purpose. A URL is not durable — it outlives neither the file it points at nor the route that served it — so it is assembled per answer rather than persisted.

`evidence_media_type` lets a client choose between an `<img>` and a `<video>` without sniffing the response. Evidence is media-type agnostic: a placement may be supported by a frame or by a short clip, and nothing in this contract distinguishes them.

`answer_status` is one of:

- `confirmed`
- `last_confirmed_only`
- `unknown`
- `ambiguous_object`

The conversational layer may shorten wording, but it must preserve `answer_status`, uncertainty, and invalidation information.

## Versioning

Breaking changes increment the major schema version. Additive optional fields increment the minor version. Store schemas, prompts, reducer configuration, model revisions, and evaluation results together so a memory decision can be reproduced.

| Version | Change |
|---|---|
| 1.0 | Initial contract |
| 1.1 | Added optional `media_epoch_id`; scoped `object.track_id` to `(session_id, media_epoch_id)`; recorded the Media Gateway as the `session_id` minter |
| 1.2 | Added optional `evidence_url` and `evidence_media_type` to an answered placement |

## Related

- [Media Relay Contract](12-Media-Relay-Contract.md) — the transport this document is *not*; it carries decoded media, never an observation
- [Recommended Architecture](01-Recommended-Architecture.md) — where the components sit
- [Team Split](05-Team-Split.md) — who owns which side of these interfaces
