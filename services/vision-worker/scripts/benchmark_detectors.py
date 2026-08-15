#!/usr/bin/env python3
"""Measure tracker identity-switch rate and per-frame latency against the
shared golden scenarios -- the algorithm-level S04 evidence docs/09-Spike-
Plan.md asks for ("first real S04 evidence, at the algorithm level only",
per the plan).

    uv run python scripts/benchmark_detectors.py
    uv run python scripts/benchmark_detectors.py --json results.json

Despite the name -- inherited from the plan's own task table, written before
BoT-SORT was descoped in favor of a pure-numpy default (see
`track/greedy_iou.py`) -- this benchmarks **trackers**, not detectors.
There is only one real detector in this service (YOLOE) and no second one
to compare it against; SAM 3.1 is deferred (task #47), and a real recall /
false-positive measurement needs recorded footage, which needs the X3 Pro
(task #31, not done). See `docs/spikes/tracker-benchmark/RESULTS.md`'s
"Deferred" section for exactly what that leaves unmeasured.

What IS measurable today, with no GPU and no glasses: identity-switch rate
and latency for both tracker implementations this service has.
`GreedyIoUTracker` -- the one actually wired into `Pipeline` -- is scored
against the real, committed scenarios in
`packages/vision-contract/src/visual_memory_vision_contract/fixtures.py`,
the same ones `tests/test_fixture_scenarios.py` holds the stability machine
accountable to. `WorldProximityTracker` -- not wired into `Pipeline`, since
nothing produces a capture pose yet (see `track/world.py`'s own docstring)
-- is scored against a small script-local synthetic 3D dataset instead: the
shared fixtures are deliberately image-space-only (no `world_point`), so
inventing 3D ground truth for them belongs here, not in the shared package.

The identity-switch counter here is simplified, not the official
MOTChallenge IDSW metric (which also weighs fragmentations and requires a
bipartite IoU-overlap match against ground-truth boxes) -- it counts a
switch whenever a ground-truth object's assigned predicted id changes, or a
predicted id gets reassigned to a different ground-truth object (a merge).
That is enough to catch what these scenarios are built to catch: two
similar objects merging into one track, or an object's identity flipping
mid-approach.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from visual_memory_vision_contract.fixtures import SCENARIOS, reconnect_reuses_a_track_id
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    Detection,
    Point2D,
    Point3D,
    TrackSample,
)

from vision_worker.track.greedy_iou import GreedyIoUTracker
from vision_worker.track.world import WorldProximityTracker


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    tracker: str
    scenario: str
    frame_count: int
    detection_count: int
    identity_switches: int
    total_latency_ms: float
    mean_latency_us_per_frame: float


class _IdentityScorer:
    """See the module docstring for what this does and does not measure."""

    def __init__(self) -> None:
        self._gt_to_pred: dict[str, str] = {}
        self._pred_to_gt: dict[str, str] = {}
        self.switches = 0

    def observe(self, gt_id: str, pred_id: str) -> None:
        prior_pred = self._gt_to_pred.get(gt_id)
        if prior_pred is not None and prior_pred != pred_id:
            self.switches += 1
        prior_gt = self._pred_to_gt.get(pred_id)
        if prior_gt is not None and prior_gt != gt_id:
            self.switches += 1
        self._gt_to_pred[gt_id] = pred_id
        self._pred_to_gt[pred_id] = gt_id


def _group_by_frame(samples: Sequence[TrackSample]) -> list[list[TrackSample]]:
    """One entry per frame index from the sequence's first to its last,
    including empty frames -- a tracker must be called every frame, even
    with zero detections, for its own staleness/max-age bookkeeping to run
    the way it does inside `Pipeline`."""
    by_frame: dict[int, list[TrackSample]] = {}
    for sample in samples:
        by_frame.setdefault(sample.frame_index, []).append(sample)
    lo, hi = min(by_frame), max(by_frame)
    return [by_frame.get(i, []) for i in range(lo, hi + 1)]


def _benchmark_image_space(
    tracker: GreedyIoUTracker, *, tracker_name: str, scenario: str, samples: Sequence[TrackSample]
) -> BenchmarkResult:
    scorer = _IdentityScorer()
    frame_count = 0
    detection_count = 0
    elapsed_s = 0.0
    for frame_samples in _group_by_frame(samples):
        detections = [s.detection for s in frame_samples]
        gt_ids = [s.track_id for s in frame_samples]

        started = time.perf_counter()
        matches = tracker.update(detections)
        elapsed_s += time.perf_counter() - started

        frame_count += 1
        detection_count += len(detections)
        for gt_id, (pred_id, _detection) in zip(gt_ids, matches, strict=True):
            scorer.observe(gt_id, pred_id)

    return BenchmarkResult(
        tracker=tracker_name,
        scenario=scenario,
        frame_count=frame_count,
        detection_count=detection_count,
        identity_switches=scorer.switches,
        total_latency_ms=elapsed_s * 1000.0,
        mean_latency_us_per_frame=(elapsed_s * 1_000_000.0 / frame_count) if frame_count else 0.0,
    )


def _benchmark_world_space(
    tracker: WorldProximityTracker,
    *,
    tracker_name: str,
    scenario: str,
    frames: Sequence[Sequence[tuple[str, Detection, Point3D]]],
) -> BenchmarkResult:
    scorer = _IdentityScorer()
    frame_count = 0
    detection_count = 0
    elapsed_s = 0.0
    for frame in frames:
        samples = [(detection, world_point) for _gt_id, detection, world_point in frame]
        gt_ids = [gt_id for gt_id, _detection, _world_point in frame]

        started = time.perf_counter()
        matches = tracker.update(samples)
        elapsed_s += time.perf_counter() - started

        frame_count += 1
        detection_count += len(samples)
        for gt_id, (pred_id, _detection) in zip(gt_ids, matches, strict=True):
            scorer.observe(gt_id, pred_id)

    return BenchmarkResult(
        tracker=tracker_name,
        scenario=scenario,
        frame_count=frame_count,
        detection_count=detection_count,
        identity_switches=scorer.switches,
        total_latency_ms=elapsed_s * 1000.0,
        mean_latency_us_per_frame=(elapsed_s * 1_000_000.0 / frame_count) if frame_count else 0.0,
    )


def _detection(label: str, x: float, y: float = 0.5) -> Detection:
    half_width = 0.08
    return Detection(
        label=label,
        confidence=0.9,
        box=BoundingBox(x_min=x - half_width, y_min=y - 0.08, x_max=x + half_width, y_max=y + 0.08),
        centroid=Point2D(x=x, y=y),
    )


def _moved_then_settled(start: float, step: float, *, settle_frames: int = 15) -> list[float]:
    moving = [start, start + step, start + 2 * step]
    return moving + [moving[-1]] * settle_frames


def _world_scenarios() -> dict[str, list[list[tuple[str, Detection, Point3D]]]]:
    """Script-local synthetic 3D ground truth for `WorldProximityTracker`.

    Not part of `visual_memory_vision_contract.fixtures` -- those are
    deliberately image-space-only (no `world_point`), per that module's own
    docstring, since no capture pose exists anywhere in this service yet.
    These positions are authored directly as `Point3D`, not produced by
    `domain/geometry.py`'s back-projection, for the same reason
    `tests/test_track_world.py` hand-builds them: there is nothing today
    that could compute them from a real frame.
    """

    def track(
        obj_id: str, label: str, positions: Sequence[tuple[float, float, float]]
    ) -> list[tuple[str, Detection, Point3D]]:
        return [(obj_id, _detection(label, 0.5), Point3D(x=x, y=y, z=z)) for x, y, z in positions]

    single = track(
        "obj-a",
        "keys",
        [(x, 0.0, 1.0) for x in _moved_then_settled(0.0, 0.15)],
    )
    single_frames = [[sample] for sample in single]

    # Two objects, far enough apart in world space (2m) that no frame ever
    # puts them within `match_distance_m` of each other -- the scenario
    # `two_similar_objects` exercises image-space-only, translated to 3D.
    a = track("obj-a", "keys", [(x, 0.0, 1.0) for x in _moved_then_settled(0.0, 0.15)])
    b = track("obj-b", "keys", [(x, 0.0, 1.0) for x in _moved_then_settled(2.0, 0.15)])
    two_objects_frames = [[a[i], b[i]] for i in range(len(a))]

    return {
        "single_object_settles_3d": single_frames,
        "two_similar_objects_3d": two_objects_frames,
    }


def _print_table(results: Sequence[BenchmarkResult]) -> None:
    header = (
        f"{'tracker':<22} {'scenario':<43} {'frames':>7} {'dets':>6} "
        f"{'idsw':>5} {'latency (ms)':>13} {'us/frame':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            f"{r.tracker:<22} {r.scenario:<43} {r.frame_count:>7} {r.detection_count:>6} "
            f"{r.identity_switches:>5} {r.total_latency_ms:>13.3f} "
            f"{r.mean_latency_us_per_frame:>9.2f}"
        )
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the full result table as JSON"
    )
    args = parser.parse_args(argv)

    results: list[BenchmarkResult] = []

    for name, build in SCENARIOS.items():
        results.append(
            _benchmark_image_space(
                GreedyIoUTracker(), tracker_name="GreedyIoUTracker", scenario=name, samples=build()
            )
        )

    before, after = reconnect_reuses_a_track_id()
    results.append(
        _benchmark_image_space(
            GreedyIoUTracker(),
            tracker_name="GreedyIoUTracker",
            scenario="reconnect_before_disconnect",
            samples=before,
        )
    )
    results.append(
        _benchmark_image_space(
            # A fresh tracker, matching `Pipeline.epoch_started`'s
            # `self._tracker.reset()` -- track_id is only ever meaningful
            # within one epoch, so scoring the second half against the
            # first would penalize a reset that is supposed to happen.
            GreedyIoUTracker(),
            tracker_name="GreedyIoUTracker",
            scenario="reconnect_after_reconnect",
            samples=after,
        )
    )

    for name, frames in _world_scenarios().items():
        results.append(
            _benchmark_world_space(
                WorldProximityTracker(),
                tracker_name="WorldProximityTracker",
                scenario=name,
                frames=frames,
            )
        )

    _print_table(results)

    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote {args.json}")

    total_switches = sum(r.identity_switches for r in results)
    if total_switches:
        print(f"\n{total_switches} identity switch(es) detected -- see the table above.")
        return 1
    print("\nzero identity switches across every scenario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
