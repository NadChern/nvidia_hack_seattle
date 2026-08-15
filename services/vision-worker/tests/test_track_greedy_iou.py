"""GreedyIoUTracker: identity assignment, the no-model default."""

from __future__ import annotations

from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.track.base import Tracker
from vision_worker.track.greedy_iou import GreedyIoUTracker


def box(x: float, y: float, *, size: float = 0.05) -> BoundingBox:
    return BoundingBox(x_min=x - size, y_min=y - size, x_max=x + size, y_max=y + size)


def a_detection(label: str = "keys", *, x: float = 0.5, y: float = 0.5) -> Detection:
    return Detection(label=label, confidence=0.9, box=box(x, y), centroid=Point2D(x=x, y=y))


def test_the_first_sighting_of_anything_mints_a_fresh_id() -> None:
    tracker = GreedyIoUTracker()

    (track_id, detection) = tracker.update([a_detection()])[0]

    assert track_id == "track-1"
    assert detection.label == "keys"


def test_an_overlapping_detection_next_frame_keeps_the_same_id() -> None:
    tracker = GreedyIoUTracker()
    tracker.update([a_detection(x=0.5, y=0.5)])

    [(track_id, _)] = tracker.update([a_detection(x=0.51, y=0.5)])

    assert track_id == "track-1"


def test_a_detection_with_no_overlap_gets_a_new_id() -> None:
    tracker = GreedyIoUTracker()
    tracker.update([a_detection(x=0.1, y=0.1)])

    [(track_id, _)] = tracker.update([a_detection(x=0.9, y=0.9)])

    assert track_id == "track-2"


def test_two_simultaneous_detections_get_different_ids() -> None:
    tracker = GreedyIoUTracker()

    results = tracker.update(
        [a_detection(label="keys", x=0.1, y=0.1), a_detection(label="wallet", x=0.9, y=0.9)]
    )

    ids = [track_id for track_id, _ in results]
    assert len(set(ids)) == 2


def test_the_closer_overlap_wins_a_contested_match() -> None:
    """Two detections both overlap one existing track; the greedy assignment
    must give it to whichever overlaps more, not whichever comes first."""
    tracker = GreedyIoUTracker(iou_threshold=0.05)
    tracker.update([a_detection(x=0.5, y=0.5, label="keys")])

    results = tracker.update(
        [
            a_detection(x=0.56, y=0.5, label="a"),  # weaker overlap
            a_detection(x=0.51, y=0.5, label="b"),  # stronger overlap
        ]
    )

    by_label = {detection.label: track_id for track_id, detection in results}
    assert by_label["b"] == "track-1"
    assert by_label["a"] != "track-1"


def test_a_track_survives_a_gap_shorter_than_max_age() -> None:
    tracker = GreedyIoUTracker(max_age_frames=3)
    tracker.update([a_detection(x=0.5, y=0.5)])

    # Two frames with nothing detected at all.
    tracker.update([])
    tracker.update([])

    [(track_id, _)] = tracker.update([a_detection(x=0.5, y=0.5)])
    assert track_id == "track-1"


def test_a_track_older_than_max_age_is_retired_and_reappearance_is_new() -> None:
    tracker = GreedyIoUTracker(max_age_frames=2)
    tracker.update([a_detection(x=0.5, y=0.5)])

    tracker.update([])
    tracker.update([])
    tracker.update([])

    [(track_id, _)] = tracker.update([a_detection(x=0.5, y=0.5)])
    assert track_id == "track-2"


def test_reset_clears_state_and_the_id_counter() -> None:
    tracker = GreedyIoUTracker()
    tracker.update([a_detection()])

    tracker.reset()
    [(track_id, _)] = tracker.update([a_detection()])

    assert track_id == "track-1"


def test_satisfies_the_tracker_protocol() -> None:
    tracker: Tracker = GreedyIoUTracker()
    assert tracker is not None
