# Personal-object identity pipeline Phase-2 results

## Decision

**Adopt the fixture-backed integration; real-footage quality remains pending.** Identity now
segments and embeds a bounded frame set once per track, matches a versioned last-known-good
gallery, escalates only the overlap band, and annotates rather than vetoes candidates. Strong
events carry the cached stable id. A registered non-resting track emits one weak `observed`
last-seen when it retires; a confirmed placement disappearance keeps the stronger `vanished`
path exclusively.

## Implemented gates

- Vision contract 1.4 distinguishes `identity=None` (not run) from an abstaining
  `IdentityMatch(object_id=None, ...)`.
- `test_strong_event_reads_the_cached_track_identity`: pass; one resolution per track.
- `test_identity_never_vetoes_candidate_availability`: pass; candidate actions identical with
  identity disabled and identity abstaining.
- `test_registered_track_end_emits_one_weak_last_seen`: pass; exactly 1 write.
- `test_unregistered_track_end_never_writes_last_seen`: pass; 0 writes.
- `test_last_seen_only_never_overclaims_a_current_location`: pass; wording says "last saw" and
  never claims a current location.
- `test_registered_identity_maps_cosine_to_the_memory_policy_floor`: pass; resolved confidence is
  at least the configured reducer threshold rather than raw cosine.
- `test_unregistered_identity_keeps_pre_feature_detection_confidence`: pass; promotion behavior is
  unchanged for the general, unregistered world.
- C-RADIO model smoke: summary and mask-weighted spatial vectors both have dimension 1,152 and
  unit norm on dev-wsl-cuda.

## Current quantitative evidence

The corrected C-RADIO distilled-summary pooling re-run on the keys-only Phase-0 set produced F1
0.917, positive accuracy 22/25, negative accuracy 24/25, ROC-AUC 0.968, and forward p95 80.8 ms.
The keys-demo starting gate is cosine 0.8334, margin 0.0440, with escalation from 0.8216. See
[Identity Probe Results](../identity-probe/RESULTS.md).

These are backbone results, not end-to-end held-out pipeline rates. Identity precision/recall,
false-identity rate, switch rate, escalation rate, and p95 with Qwen escalation remain pending the
frozen labeled set and real verifier. `scripts/eval_identity.py` prints the fixed table with
numerator/denominator and can append the live `/v1/status` identity counters.

## Physical gates still open

- Real YOLOE mask quality on the glasses clips.
- Qwen3-VL same-instance escalation quality and p95.
- Detector + C-RADIO + Qwen + speech coexistence and peak unified memory on the physical GN100.
