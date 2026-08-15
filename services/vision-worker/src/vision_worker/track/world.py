"""A tracker that matches identity by 3D world-position proximity instead of
2D image-space IoU -- the prior-art project's proven reconciliation rule
(`DetectionAnchorRegistry.FindBestMatch`, `worldMatchDistanceMeters = 0.3`),
ported as the "authority when depth and pose are available" tier the plan
describes: a resting object has a constant world position regardless of
head motion, which beats any image-space tracker once that position exists.

**Not wired into `Pipeline` yet, and does not implement `track.base.Tracker`.**
`Tracker.update()` takes only `Sequence[Detection]` -- there is nowhere in
this service today that can hand it a `Point3D` per detection, because that
requires both a depth adapter (`depth/moge.py`, wired) and a capture pose
(`domain/geometry.py`'s `CapturePose`, which nothing produces until task
#46's `DevicePose` reads real ARDK pose off the relay's data channel). This
class takes `(Detection, Point3D)` pairs directly instead of inventing a
pose to satisfy the existing interface -- built and tested now so task #46
only has to supply real world points, not design this reconciliation from
scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from visual_memory_vision_contract.protocol import Detection, Point3D


def _distance(a: Point3D, b: Point3D) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


@dataclass(slots=True)
class _Track:
    label: str
    world_point: Point3D
    last_seen_frame: int


class WorldProximityTracker:
    """Matches each frame's `(Detection, Point3D)` pairs against the
    previous frames' world positions, greedily, nearest same-label pairs
    first -- the same "score all candidate pairs, claim best-first" shape
    `track/greedy_iou.py` uses, with 3D distance in place of IoU and an
    added same-label constraint (ported from `FindBestMatch`, which never
    matches a "keys" detection to a "wallet" track regardless of distance).

    `max_age_frames` defaults to `GreedyIoUTracker`'s own default for the
    same reason documented there: this must stay at least as large as
    `StabilityConfig.reacquire_within_frames`, or identity is lost here
    before the stability machine's own occlusion tolerance gets a chance to
    apply.
    """

    def __init__(self, *, match_distance_m: float = 0.3, max_age_frames: int = 45) -> None:
        self._match_distance_m = match_distance_m
        self._max_age_frames = max_age_frames
        self._tracks: dict[str, _Track] = {}
        self._next_id = 0
        self._frame_index = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        self._frame_index = 0

    def update(
        self, samples: Sequence[tuple[Detection, Point3D]]
    ) -> Sequence[tuple[str, Detection]]:
        candidates: list[tuple[float, int, str]] = [
            (distance, index, track_id)
            for index, (detection, world_point) in enumerate(samples)
            for track_id, track in self._tracks.items()
            if track.label == detection.label
            and (distance := _distance(world_point, track.world_point)) <= self._match_distance_m
        ]
        candidates.sort(key=lambda candidate: candidate[0])

        matched_sample: dict[int, str] = {}
        claimed_tracks: set[str] = set()
        for _, index, track_id in candidates:
            if index in matched_sample or track_id in claimed_tracks:
                continue
            matched_sample[index] = track_id
            claimed_tracks.add(track_id)

        results: list[tuple[str, Detection]] = []
        for index, (detection, world_point) in enumerate(samples):
            track_id = matched_sample[index] if index in matched_sample else self._mint_id()
            self._tracks[track_id] = _Track(
                label=detection.label, world_point=world_point, last_seen_frame=self._frame_index
            )
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


__all__ = ["WorldProximityTracker"]
