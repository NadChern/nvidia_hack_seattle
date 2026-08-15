# Tracker Benchmark Results

Test date: 2026-08-05

## Decision

**Keep `GreedyIoUTracker` as the default tracker.** Zero identity switches
across all 7 shared golden scenarios plus both halves of the reconnect
scenario, at sub-6µs/frame — pure-Python arithmetic on bounding boxes is not
the bottleneck anywhere in this pipeline. `WorldProximityTracker` is real,
tested, and ready, but stays unwired until task #46's `DevicePose` gives it
something to run on — there is no regression risk in shipping it now, since
nothing calls it.

**This does not close the BoT-SORT question, and must not be read as
closing it** — see "What this benchmark cannot see" below. Every scenario
here is a stationary-camera simulation, and camera motion is the one thing
BoT-SORT was selected for.

Detector-level comparison (YOLOE vs. a second real detector, recall and
false-positive rate) is **deferred** — see below.

## What ran

`services/vision-worker/scripts/benchmark_detectors.py`, no GPU, no
`models` extra, no glasses. `GreedyIoUTracker` against the real scenarios in
`packages/vision-contract`'s `fixtures.py`; `WorldProximityTracker` against
a script-local synthetic 3D dataset (see the script's own docstring for
why). Three consecutive runs, no differences in identity-switch count run
to run — only latency jitters, as expected of a microbenchmark timing
microsecond-scale work.

| tracker | scenario | frames | detections | identity switches | latency (ms) | µs/frame |
|---|---|---:|---:|---:|---:|---:|
| GreedyIoUTracker | keys_placed_on_table | 18 | 18 | 0 | 0.061–0.063 | 3.4–3.5 |
| GreedyIoUTracker | keys_carried_out_never_replaced | 25 | 25 | 0 | 0.123–0.142 | 4.9–5.7 |
| GreedyIoUTracker | keys_carried_to_another_room_and_set_down | 36 | 36 | 0 | 0.062–0.071 | 1.7–2.0 |
| GreedyIoUTracker | object_visible_never_touched | 40 | 40 | 0 | 0.067–0.069 | 1.7 |
| GreedyIoUTracker | walking_past_without_touching | 1 | 1 | 0 | 0.002 | 1.9–2.1 |
| GreedyIoUTracker | brief_hand_occlusion | 39 | 20 | 0 | 0.045–0.047 | 1.2 |
| GreedyIoUTracker | two_similar_objects | 18 | 36 | 0 | 0.060–0.062 | 3.3–3.4 |
| GreedyIoUTracker | reconnect_before_disconnect | 18 | 18 | 0 | 0.028–0.030 | 1.6–1.7 |
| GreedyIoUTracker | reconnect_after_reconnect | 18 | 18 | 0 | 0.028–0.057 | 1.6–3.2 |
| WorldProximityTracker | single_object_settles_3d | 18 | 18 | 0 | 0.056–0.079 | 3.1–4.4 |
| WorldProximityTracker | two_similar_objects_3d | 18 | 36 | 0 | 0.050–0.052 | 2.8–2.9 |

**Zero identity switches, every scenario, every run.** `two_similar_objects`
and its 3D counterpart `two_similar_objects_3d` are the scenarios that
actually contest identity (two same-label objects tracked simultaneously);
both trackers keep them separate throughout. `reconnect_after_reconnect`'s
one noisy run (3.2µs/frame vs. 1.6-1.7 elsewhere) is measurement jitter, not
a correctness signal — the identity-switch count for that run was still 0.

## Findings

### What this benchmark cannot see: camera motion

Every scenario in `fixtures.py` fixes `background_motion` at `(0, 0)` — a
stationary camera, by construction (the module docstring says so). That is
the right shape for testing the *stability machine*, which is what those
fixtures were written for. It is the wrong shape for concluding anything
about a tracker on a head-worn camera.

The original plan chose BoT-SORT for one specific reason, and it is exactly
the reason this benchmark cannot test: **global motion compensation.** On
glasses, a stationary object can jump most of the frame width because the
head moved. `GreedyIoUTracker` has no motion model at all — a head turn that
displaces a box past its own width drops IoU to zero and mints a new id, an
identity switch on an object that never moved. BoT-SORT's GMC exists to
absorb precisely that, and its two-stage association recovers the
low-confidence detections a fast pan produces.

So the honest reading of the table below is: **zero identity switches on the
scenarios that contest identity while the camera is still.** The egocentric
case is not measured here and cannot be until either an ego-motion fixture
or real X3 Pro footage (task #31) exists. `track/botsort.py` remains worth
building for the YOLOE path — where `ultralytics` is loaded anyway, so it
costs no new dependency — rather than being considered displaced by this
result. `GreedyIoUTracker` remains the right default for the no-GPU path,
which has no alternative.

### The reconnect scenario validates the reset contract, not just the tracker

`reconnect_before_disconnect` and `reconnect_after_reconnect` are scored
independently, with a fresh `GreedyIoUTracker()` for the second half —
matching what `Pipeline.epoch_started` actually does
(`self._tracker.reset()`). Both halves reuse ground-truth id `track-1` by
construction (see `fixtures.reconnect_reuses_a_track_id`'s docstring); a
benchmark that scored them against one continuous identity map would either
need to special-case that reuse or produce a meaningless number. Scoring
them as two independent epochs is the correct comparison, and it is also
exactly what `tests/test_pipeline.py::test_epoch_reset_treats_a_reused_
track_id_as_a_new_sighting` already asserts one layer up, at the `Pipeline`
level instead of the tracker level.

### `WorldProximityTracker` has nothing to fail on yet

Its two scenarios are synthetic by necessity — no capture pose exists
anywhere in this service (task #46 is what produces one), so there is no
real `Point3D` per detection to benchmark against. The two scenarios here
exercise the reconciliation algorithm's own logic (same-label gate, nearest-
match-wins) correctly, but they cannot substitute for real-world 6DoF pose
data once that exists — this result should be re-run once task #46 lands
and real world points are available from a live epoch.

### Both trackers are far from the frame budget

At 24fps the frame budget is ~41.7ms. Both trackers finish in single-digit
microseconds — three orders of magnitude of headroom. Detection and depth
inference, not tracking, are what will determine this pipeline's real
frame-rate ceiling.

## Deferred

**Detector recall and false-positive rate.** This needs real footage with
labeled ground-truth boxes — `media/clips/manifest.json`, from task #31,
gated on the RayNeo X3 Pro (task #30's uplink gate). Neither exists yet.
There is also only one real detector in this service today (YOLOE); SAM 3.1
(task #47) is deferred, so there is no second detector to compare it
against even once footage exists. Re-run this benchmark's detector-facing
half once both land — the script's own docstring documents exactly this gap
so it isn't rediscovered.

**BoT-SORT under real camera motion.** See "What this benchmark cannot see"
above. Needs either an ego-motion fixture (synthetic, cheap, and worth doing
before the footage exists) or real X3 Pro clips.

**MOTChallenge-standard identity-switch scoring.** This benchmark's
identity-switch counter is simplified — see `benchmark_detectors.py`'s
module docstring for precisely what it does and does not measure (no
fragmentation weighting, no bipartite IoU-overlap match against ground-truth
boxes). Sufficient to catch what these scenarios are built to catch; not a
substitute for `TrackEval` or an equivalent library against real footage.
