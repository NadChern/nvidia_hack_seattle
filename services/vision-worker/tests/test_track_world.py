"""WorldProximityTracker: identity by 3D world-position proximity, the
authority tier task #46's real ARDK poses will eventually feed. Every test
here hand-builds the `(Detection, Point3D)` pairs a live pipeline would only
be able to produce once a capture pose exists -- exactly like
`test_stability.py` hand-builds `TrackSample`s instead of running a real
detector.
"""

from __future__ import annotations

from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D, Point3D

from vision_worker.track.world import WorldProximityTracker

_BOX = BoundingBox(x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5)


def a_detection(label: str = "keys") -> Detection:
    return Detection(label=label, confidence=0.9, box=_BOX, centroid=Point2D(x=0.45, y=0.45))


def at(x: float, y: float, z: float) -> Point3D:
    return Point3D(x=x, y=y, z=z)


def test_the_first_sighting_of_anything_mints_a_fresh_id() -> None:
    tracker = WorldProximityTracker()

    [(track_id, detection)] = tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    assert track_id == "track-1"
    assert detection.label == "keys"


def test_a_nearby_reappearance_keeps_the_same_id() -> None:
    tracker = WorldProximityTracker(match_distance_m=0.3)
    tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    [(track_id, _)] = tracker.update([(a_detection(), at(0.05, 0.0, 1.0))])

    assert track_id == "track-1"


def test_a_detection_far_away_gets_a_new_id() -> None:
    tracker = WorldProximityTracker(match_distance_m=0.3)
    tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    [(track_id, _)] = tracker.update([(a_detection(), at(5.0, 0.0, 1.0))])

    assert track_id == "track-2"


def test_the_same_label_at_the_same_place_is_still_required_to_match() -> None:
    """Ported from `FindBestMatch`: same-label is a hard gate, not just a
    tiebreaker -- a "keys" detection never reconciles onto a "wallet" track
    no matter how close, since that would silently merge two different
    physical objects."""
    tracker = WorldProximityTracker(match_distance_m=0.3)
    tracker.update([(a_detection("keys"), at(0.0, 0.0, 1.0))])

    [(track_id, _)] = tracker.update([(a_detection("wallet"), at(0.0, 0.0, 1.0))])

    assert track_id == "track-2"


def test_two_simultaneous_detections_get_different_ids() -> None:
    tracker = WorldProximityTracker()

    results = tracker.update(
        [
            (a_detection("keys"), at(0.0, 0.0, 1.0)),
            (a_detection("wallet"), at(5.0, 0.0, 1.0)),
        ]
    )

    ids = [track_id for track_id, _ in results]
    assert len(set(ids)) == 2


def test_the_closer_point_wins_a_contested_match() -> None:
    """Both candidates share the existing track's label -- same-label is
    already covered by its own test above, so this isolates distance as the
    tiebreaker."""
    tracker = WorldProximityTracker(match_distance_m=1.0)
    tracker.update([(a_detection("keys"), at(0.0, 0.0, 1.0))])

    results = tracker.update(
        [
            (a_detection("keys"), at(0.6, 0.0, 1.0)),  # farther
            (a_detection("keys"), at(0.1, 0.0, 1.0)),  # closer
        ]
    )

    farther_id, closer_id = results[0][0], results[1][0]
    assert closer_id == "track-1"
    assert farther_id != "track-1"


def test_a_track_survives_a_gap_shorter_than_max_age() -> None:
    tracker = WorldProximityTracker(max_age_frames=3)
    tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    tracker.update([])
    tracker.update([])

    [(track_id, _)] = tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])
    assert track_id == "track-1"


def test_a_track_older_than_max_age_is_retired_and_reappearance_is_new() -> None:
    tracker = WorldProximityTracker(max_age_frames=2)
    tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    tracker.update([])
    tracker.update([])
    tracker.update([])

    [(track_id, _)] = tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])
    assert track_id == "track-2"


def test_reset_clears_state_and_the_id_counter() -> None:
    tracker = WorldProximityTracker()
    tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    tracker.reset()
    [(track_id, _)] = tracker.update([(a_detection(), at(0.0, 0.0, 1.0))])

    assert track_id == "track-1"
