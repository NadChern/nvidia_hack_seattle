# Evaluation Plan

## Dataset and split

Target 60 first-person clips recorded with the actual glasses; use at least 40 if hackathon time is constrained. Label before inspecting final model results.

- Development set: approximately 60%, used for prompts, thresholds, and debugging
- Held-out test set: approximately 40%, frozen before the final comparison

Keep clips from the same continuous recording, room setup, and repeated action sequence in the same split to reduce leakage. Record participant, object instance, room, lighting, and capture conditions so results can be sliced by scenario.

### Personal-object identity protocol

The identity set adds **same-instance positives and different-instance, same-class negatives**.
For each enrolled reference set, pair one query of the same physical object with one query of a
different physical object carrying the same label. Keep the protocol balanced: always accepting
then produces F1=0.667 and cannot masquerade as instance recognition. Report accept/reject F1,
positive accuracy, negative accuracy, ROC-AUC, cosine distributions, identity margin, coverage,
and VLM escalation rate. Accuracy is always paired with embed latency and peak memory.

Thresholds move only on the development set. The frozen held-out set reports the selected
threshold once. Every rate includes its numerator/denominator, and a set below approximately 30
physical objects is explicitly labeled too small for a reliability claim. The preliminary
keys-only set (3 keyrings, 25 balanced positive/negative trials) is a demo risk check, not the
frozen set; its record is in [Identity Probe Results](spikes/identity-probe/RESULTS.md).
`services/vision-worker/scripts/eval_identity.py` prints the fixed end-to-end table from labeled
predictions and reads the running service's resolved/ambiguous/unmatched/escalated, latency, and
gallery counters from `/v1/status`, so offline evaluation and the demo report the same runtime
measurements.

## Ground-truth labels

For every clip, label:

- stable object identity and aliases;
- event action and temporal window;
- room, surface, and relation when observable;
- pickup invalidation and current status;
- last confirmed placement;
- last seen observation;
- acceptable evidence-frame interval;
- whether a safe query answer is confirmed, last-confirmed-only, unknown, or ambiguous.

Use two reviewers for ambiguous event and evidence labels when possible. Record disagreements instead of forcing certainty.

## Required scenarios

- keys placed on a table and moved between rooms;
- pickup without a later placement;
- object carried through a room change;
- object visible without interaction;
- walking past an object without touching it;
- wallet placed in an open drawer, then the drawer closed;
- similar-looking objects and ambiguous identity;
- brief hand occlusion and partial frame visibility;
- dim lighting, glare, and motion blur;
- camera disconnect and reconnect;
- duplicate and out-of-order observation delivery;
- low-confidence evidence after a strong prior placement;
- evidence file unavailable at query time;
- spoken alias, correction, and unknown-object query.

## Metric definitions

### Perception and identity

- **Target-object recall:** ground-truth target appearances with at least one matched detection divided by all target appearances.
- **False-positive rate:** unmatched target detections divided by all target detections.
- **Time to first detection:** elapsed time from first label-visible frame to first matched detection.
- **Identity switch rate:** incorrect stable-object assignments after occlusion, track loss, or reconnect.
- **Ambiguous merge rate:** similar objects silently merged into one stable object.

Define mask/box matching thresholds and object-visibility rules before the held-out run.

### Event and spatial understanding

- Placement and pickup precision/recall
- False placement events per clip
- Event timestamp error against the labeled event window
- Current-status transition accuracy
- Normalized room and surface accuracy
- Spatial-relation accuracy only when both semantic entities are correct
- Evidence-frame correctness: target and support are visible and the frame falls inside the accepted interval
- Verifier JSON validity before and after a single repair attempt

### End-to-end memory

- **Last-confirmed-placement accuracy:** stable object, room, surface, and relation match the labeled placement.
- **Current-status accuracy:** confirmed-at-location, in-transit, or unknown matches the label.
- **Stale-location claim rate:** system claims a current location after that location was invalidated.
- **Unsupported confident answer rate:** a confirmed answer lacks a qualifying stored event and loadable evidence.
- **Answer coverage:** queries receiving a confirmed current-location answer divided by all queries.
- **Safe abstention accuracy:** unknown or ambiguous cases handled without a location claim.
- Correct timestamp and evidence returned
- Query-to-answer latency at p50 and p95

Report coverage beside error rates. A system can achieve zero unsupported answers by refusing every query, so abstention and correctness must be interpreted together.

## Initial engineering targets

These are internal gates, not published benchmark claims:

```text
Target-object recall:             ≥ 90%
Placement-event precision:        ≥ 85%
Pickup-event precision:           ≥ 90%
Current-status accuracy:          ≥ 85%
Last-confirmed-placement accuracy:≥ 80%
Correct evidence frame:           ≥ 90%
Stale current-location claims:      0
Unsupported confident answers:      0
Safe abstention accuracy:         ≥ 90%
Query response p95:                < 5 seconds after transcript
Query response target:             < 3 seconds after transcript
```

For every zero-error result, report the numerator and denominator, for example `0/24`, and a confidence interval or an explicit note that the test set is too small for a reliability claim.

## Model comparison protocol

Compare the provisional Qwen3-VL verifier and the GN100-hosted `nvidia/Cosmos3-Nano` challenger on identical frozen clips, frames, masks, metadata, prompts, JSON schema, and generation settings. MiniCPM's external API may be measured as a laptop Agent profile, but it is not part of this event-verifier comparison. Record:

- exact checkpoint and repository revision;
- precision or quantization;
- prompt and schema version;
- event classification precision/recall;
- room, surface, and relation accuracy;
- raw and repaired JSON validity;
- p50/p95 latency and cold-start time;
- peak unified-memory use, including the complete detection, speech, media, and verifier workload;
- failure examples and abstentions;
- tool-call and guard behavior separately if a verifier checkpoint is also proposed for the Agent.

Choose one primary verifier before final integration. Evaluate GPT4Scene-style markers, LingBot-Map, their conditional combination, and Stream3D-VLM only through the frozen experiments in [Spike Plan](09-Spike-Plan.md). Compare them on explicitly labeled room/layout questions from recorded streams; do not mix published benchmark scores with project results.

## Memory reducer tests

Run deterministic table-driven tests independent of the models:

- duplicate delivery is idempotent;
- late observations produce the same final timeline;
- pickup invalidates current location;
- observed-without-interaction does not create a placement;
- weak evidence does not overwrite a strong placement;
- reconnect does not reuse tracker identity;
- ambiguous identity does not merge objects;
- missing evidence downgrades the answer;
- deletion removes state and evidence in scope.

## Demo acceptance checklist

- [ ] Object recognized and stable identity resolved
- [ ] Placement event captured
- [ ] Pickup invalidation handled
- [ ] Room, surface, and relation recorded when supported
- [ ] Evidence frame saved and loadable
- [ ] Natural-language query resolved through memory tools
- [ ] Confirmed, last-confirmed-only, unknown, and ambiguous responses are truthful
- [ ] Low-confidence case handled without overwriting trusted state
- [ ] Pure-Kotlin client with the pinned LiveKit Android SDK passes camera, microphone, HUD, and return-audio tests on the actual glasses
- [ ] WebRTC reconnect handled without tracker-ID reuse
- [ ] Session deletion and retention tested
- [ ] At least two fallback modes rehearsed
- [ ] Held-out metrics recorded with denominators and latency percentiles

## Deployment release gate

The demo checklist validates product behavior. A release must also pass the deployment path:

- [ ] Every required deployment image builds for Linux ARM64 and is recorded by immutable digest
- [ ] Docker Compose starts successfully on the physical GN100
- [ ] All required liveness and readiness checks pass
- [ ] Pinned LiveKit Android SDK camera, microphone, return-audio, and reconnect paths pass on the actual glasses against the GN100 deployment
- [ ] Parakeet and Kokoro pass the shared English golden set
- [ ] Representative vision, memory, query, evidence, and deletion scenarios pass together
- [ ] A 30-minute complete-workload soak has no crash, unbounded queue, or audio underrun
- [ ] Peak unified-memory use preserves the agreed operating-system and recovery headroom
- [ ] Persistent database and evidence data survive a service restart
- [ ] Network exposure matches the trusted-LAN policy with no unexpected external dependency
- [ ] The previous known-good Compose manifest and image digests can be restored

Cross-architecture image construction, CPU emulation, and mocked CI do not satisfy the physical GN100 gate.

See [Hackathon Stack](03-Hackathon-Stack.md), [Data Contract and Memory Semantics](06-Data-Contract.md), [Team Split](05-Team-Split.md), [Development and Deployment](08-Development-and-Deployment.md), and [Spike Plan](09-Spike-Plan.md).
