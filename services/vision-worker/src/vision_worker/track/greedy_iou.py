"""A tracker that needs no model: greedy IoU matching across consecutive
frames.

The default, not a placeholder. The Roboflow multi-object-tracking survey
this project's tracking approach was chosen against notes that "well-designed
heuristic strategies can outperform complex global optimization approaches
while maintaining computational efficiency" -- TrackTrack's own justification
for greedy, track-perspective association over a full Hungarian solve. This
is that same idea at the simplest useful scale: plain Python arithmetic on
bounding boxes, no `numpy`, no `scipy`, no `torch`.

This also closes a gap the plan's first draft left open: `track/botsort.py`
couples tracking to `ultralytics`' `model.track()`, which only runs when a
real detector is loaded. Without a dependency-free default here, the `ci` and
`dev-macos` no-GPU paths would have a detector but no tracker, and the
stability machine has nothing to consume without one.

**This does not replace BoT-SORT for the on-glasses path.** BoT-SORT was
chosen for global motion compensation: on a head-worn camera a stationary
object can jump most of the frame because the head moved, and pure IoU
association has no motion model at all to absorb that -- a fast head turn
displaces a box past its own width, IoU falls to zero, and the id is lost.
Every fixture this tracker is benchmarked against is a stationary-camera
simulation (`background_motion` fixed at zero), so that failure mode is
untested here by construction. See
`docs/spikes/tracker-benchmark/RESULTS.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from visual_memory_vision_contract.protocol import BoundingBox, Detection


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x_min = max(a.x_min, b.x_min)
    y_min = max(a.y_min, b.y_min)
    x_max = min(a.x_max, b.x_max)
    y_max = min(a.y_max, b.y_max)
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    intersection = (x_max - x_min) * (y_max - y_min)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(slots=True)
class _Track:
    box: BoundingBox
    last_seen_frame: int


class GreedyIoUTracker:
    """Matches each frame's detections against the previous frames' boxes by
    IoU, greedily, highest-overlap pairs first.

    `max_age_frames` is how long a track may go completely unmatched before
    its id is retired and a later reappearance mints a new one. Defaulted to
    `vision_worker.domain.stability.StabilityConfig.reacquire_within_frames`'s
    default (45): if this tracker forgot an id *before* the stability machine
    was willing to tolerate the gap, identity would already be lost at this
    layer and the physics state machine would never get a chance to apply its
    own occlusion tolerance. Keep this value at least as large as whatever
    `reacquire_within_frames` is configured to.
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_age_frames: int = 45) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_frames = max_age_frames
        self._tracks: dict[str, _Track] = {}
        self._next_id = 0
        self._frame_index = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        self._frame_index = 0

    def update(self, detections: Sequence[Detection]) -> Sequence[tuple[str, Detection]]:
        candidates: list[tuple[float, int, str]] = [
            (iou, index, track_id)
            for index, detection in enumerate(detections)
            for track_id, track in self._tracks.items()
            if (iou := _iou(detection.box, track.box)) >= self._iou_threshold
        ]
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        matched_detection: dict[int, str] = {}
        claimed_tracks: set[str] = set()
        for _, index, track_id in candidates:
            if index in matched_detection or track_id in claimed_tracks:
                continue
            matched_detection[index] = track_id
            claimed_tracks.add(track_id)

        results: list[tuple[str, Detection]] = []
        for index, detection in enumerate(detections):
            track_id = matched_detection[index] if index in matched_detection else self._mint_id()
            self._tracks[track_id] = _Track(box=detection.box, last_seen_frame=self._frame_index)
            results.append((track_id, detection))

        self._expire_stale_tracks()
        self._frame_index += 1
        return tuple(results)

    def _mint_id(self) -> str:
        self._next_id += 1
        return f"track-{self._next_id}"

    def _expire_stale_tracks(self) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if self._frame_index - track.last_seen_frame > self._max_age_frames
        ]
        for track_id in stale:
            del self._tracks[track_id]


__all__ = ["GreedyIoUTracker"]
