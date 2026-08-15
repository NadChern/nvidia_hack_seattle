# Tracker Benchmark

Measures identity-switch rate and per-frame latency for both tracker
implementations `services/vision-worker` has, against the shared golden
scenarios — the algorithm-level evidence docs/09-Spike-Plan.md's S04 entry
calls for, ahead of anything that needs real recorded footage.

## What it runs

The actual script is `services/vision-worker/scripts/benchmark_detectors.py`
(the name is inherited from the plan's original task table, written before
BoT-SORT was descoped — see the script's own docstring for why it
benchmarks trackers, not detectors).

```text
packages/vision-contract's SCENARIOS (7 named scenarios)
  + reconnect_reuses_a_track_id (2 halves, tracker reset between them)
    → GreedyIoUTracker.update() every frame, scored against ground truth

script-local synthetic 3D scenarios (2 scenarios)
    → WorldProximityTracker.update() every frame, scored against ground truth
```

`GreedyIoUTracker` is scored against the real, committed scenarios in
`packages/vision-contract/src/visual_memory_vision_contract/fixtures.py` —
the same ones `tests/test_fixture_scenarios.py` holds the stability machine
accountable to, so a tracker regression that let two objects merge would
also start breaking those tests. `WorldProximityTracker` is not wired into
`Pipeline` — nothing produces a capture pose yet, see `track/world.py`'s own
docstring — so it has no real footage to run against; the benchmark instead
authors a small 3D dataset directly, mirroring the shared `two_similar_
objects` scenario's shape.

## Run it

```bash
cd services/vision-worker
uv run python scripts/benchmark_detectors.py
uv run python scripts/benchmark_detectors.py --json results.json
```

No GPU, no `models` extra, no glasses — both trackers are pure Python/numpy.
Exit code is non-zero if any identity switch is detected, so this doubles as
a regression check.

## What this does not measure

Detector recall and false-positive rate, which need real footage with
ground-truth boxes (task #31, gated on the RayNeo X3 Pro — not done), and a
second real detector to compare YOLOE against (SAM 3.1, task #47, deferred).
See [RESULTS.md](RESULTS.md)'s Decision and Deferred sections.
