"""Golden scenarios, shared by producer and consumer.

Person 1 (Vision) checks that the interaction/rest state machine transitions
correctly on these; Person 2 (spatial verification) checks that a verifier
judges the resulting candidates correctly. Per docs/05-Team-Split.md an
interface is complete only when the provider fixture passes inside the
consumer's harness, and shared fixtures are what makes that mechanical
rather than aspirational -- `packages/memory-contract`'s `fixtures.py` plays
the same role for the reducer, and this mirrors its structure directly.

Everything here is deterministic -- fixed identifiers, a fixed origin
timestamp, and a fixed 24fps frame interval -- so a byte comparison is
meaningful and a diff points at a real change.

Every scenario uses the image-space-only path (`background_motion` fixed at
`(0, 0)`, no `world_point`) -- the stationary-camera simulation that is also
the no-glasses path anyone can replay on a laptop. Motion and settle phases
are sized with margin above `vision_worker.domain.stability.StabilityConfig`'s
*default* thresholds (`dwell_frames=12`, `passive_confirmation_frames=90`),
not a shortened test config, so a fixture behaves the same way replayed
against real production defaults as it does against a tuned-down test
config.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D, TrackSample

#: A fixed wall-clock origin. 10:42 local in the demo narrative -- the same
#: moment `packages/memory-contract`'s fixtures use.
T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
FRAME_INTERVAL = dt.timedelta(seconds=1 / 24)

SESSION = "sess_01JABDEMO0000000000000000"
DEVICE = "glasses-01"
EPOCH = "TR_VCabc123"
#: A second epoch, produced by a reconnect. Same wearer, new track SIDs.
EPOCH_AFTER_RECONNECT = "TR_VCdef456"

#: Comfortably above StabilityConfig's default dwell_frames=12, so a settle
#: phase promotes with margin rather than riding the exact threshold.
_SETTLE_FRAMES = 15
#: Comfortably below the default passive_confirmation_frames=90, so a first
#: sighting that stays still for this long must still not promote.
_BRIEF_SIGHTING_FRAMES = 40


def _detection(label: str, x: float, *, y: float = 0.5, confidence: float = 0.9) -> Detection:
    half_width = 0.08
    return Detection(
        label=label,
        confidence=confidence,
        box=BoundingBox(x_min=x - half_width, y_min=y - 0.08, x_max=x + half_width, y_max=y + 0.08),
        centroid=Point2D(x=x, y=y),
    )


def _sample(
    *, track_id: str, frame_index: int, label: str, x: float, y: float = 0.5
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        captured_at=T0 + frame_index * FRAME_INTERVAL,
        detection=_detection(label, x, y=y),
        # Stationary-camera simulation throughout -- see the module docstring.
        background_motion=Point2D(x=0.0, y=0.0),
    )


def _track(track_id: str, label: str, positions: Sequence[float]) -> tuple[TrackSample, ...]:
    return tuple(
        _sample(track_id=track_id, frame_index=i, label=label, x=x) for i, x in enumerate(positions)
    )


def _moved_then_settled(
    start: float, step: float, *, settle_at: float | None = None
) -> list[float]:
    """Three moving frames, then `_SETTLE_FRAMES` still frames at the final
    (or `settle_at`) position -- the "carried in and set down" shape every
    `placed`-producing scenario below shares."""
    moving = [start, start + step, start + 2 * step]
    final = settle_at if settle_at is not None else moving[-1]
    return moving + [final] * _SETTLE_FRAMES


def keys_placed_on_table() -> Sequence[TrackSample]:
    """Clip 1: carried in, then settles. Expected: "observed" on the first
    sample, then "placed" once the settle phase crosses `dwell_frames`."""
    return _track("track-1", "keys", _moved_then_settled(0.10, 0.06))


def keys_carried_out_never_replaced() -> Sequence[TrackSample]:
    """Clip 2, the demo case. Placed, then picked up and carried away, and
    never settles again. Expected: "observed", "placed", "picked_up", and
    no further "placed" -- a system that answers with the last sighting
    here is wrong in the way this whole project exists to avoid.
    """
    settle = _moved_then_settled(0.10, 0.06)
    last = settle[-1]
    leave = [last + 0.06 * i for i in range(1, 8)]
    return _track("track-1", "keys", settle + leave)


def keys_carried_to_another_room_and_set_down() -> Sequence[TrackSample]:
    """Clip 3: carried, then a new "placed" at a different position.
    Expected: "observed", "placed", "picked_up", "placed" again."""
    settle = _moved_then_settled(0.10, 0.06)
    move = [settle[-1] + 0.06 * i for i in range(1, 4)]
    second_settle = [move[-1]] * _SETTLE_FRAMES
    return _track("track-1", "keys", settle + move + second_settle)


def object_visible_never_touched() -> Sequence[TrackSample]:
    """Clip 4: stable from the very first sighting, for `_BRIEF_SIGHTING_
    FRAMES` -- deliberately shorter than `passive_confirmation_frames`.
    Expected: exactly one "observed", never a "placed". Looks exactly like
    an object that was always there; see `domain.stability`'s module
    docstring for why that distinction cannot be made from a single frame.
    """
    return _track("track-1", "keys", [0.5] * _BRIEF_SIGHTING_FRAMES)


def walking_past_without_touching() -> Sequence[TrackSample]:
    """Clip 5: one fleeting sighting. Expected: exactly one "observed",
    nothing else -- not even a repeat of "observed"."""
    return _track("track-1", "keys", [0.5])


def brief_hand_occlusion() -> Sequence[TrackSample]:
    """Clip 6: settles, then a gap in `frame_index` (not represented as
    samples -- a consumer replaying this must feed `TrackRegistry.observe
    (track_id, None)` for the missing frames, matching `domain.stability.
    step`'s `sample=None` contract), then reappears at the same position.
    Expected: "observed", "placed", then nothing on reappearance -- the
    gap is well inside `reacquire_within_frames` (default 45), so the
    reappearance must not re-announce "observed".
    """
    before = _track("track-1", "keys", _moved_then_settled(0.10, 0.06))
    gap = 20
    resume_at = before[-1].frame_index + gap
    after = tuple(
        _sample(
            track_id="track-1",
            frame_index=resume_at + i,
            label="keys",
            x=before[-1].detection.centroid.x,
        )
        for i in range(2)
    )
    return before + after


def two_similar_objects() -> Sequence[TrackSample]:
    """Clip 7: two keys-labeled objects settle in different places around
    the same time. Expected: both independently reach "placed" as separate
    tracks -- a naive label-only lookup afterward would be ambiguous
    between them, which is exactly the `ambiguous_object` case docs/06
    requires an honest answer for rather than a guessed merge. Samples are
    interleaved by `frame_index`, matching what a live pipeline sees when
    two objects share the same frames.
    """
    track_a = _track("track-1", "keys", _moved_then_settled(0.10, 0.06))
    track_b = _track("track-2", "keys", _moved_then_settled(0.70, 0.06))
    return tuple(
        sorted(track_a + track_b, key=lambda sample: (sample.frame_index, sample.track_id))
    )


def reconnect_reuses_a_track_id() -> tuple[Sequence[TrackSample], Sequence[TrackSample]]:
    """Clip 8. Two independent sequences, both using `track-1` -- a
    consumer MUST reset all per-track state between them (`TrackRegistry.
    reset()`, `Tracker.reset()`, `PoseSource.reset()`, all called on
    `epoch_started`), or the second sequence's object gets merged into the
    first's. That merge is the exact failure `docs/06-Data-Contract.md`:110
    describes: a tracker's numbering restarts after a dropout, so
    `track-1` before and after are different physical objects.

    Returns a pair rather than one sequence -- deliberately not in
    `SCENARIOS`, since its entire point is the reset boundary between the
    two halves, not a single replay.
    """
    before = _track("track-1", "keys", _moved_then_settled(0.10, 0.06))
    after = _track("track-1", "phone", _moved_then_settled(0.60, 0.06))
    return before, after


SCENARIOS = {
    "keys_placed_on_table": keys_placed_on_table,
    "keys_carried_out_never_replaced": keys_carried_out_never_replaced,
    "keys_carried_to_another_room_and_set_down": keys_carried_to_another_room_and_set_down,
    "object_visible_never_touched": object_visible_never_touched,
    "walking_past_without_touching": walking_past_without_touching,
    "brief_hand_occlusion": brief_hand_occlusion,
    "two_similar_objects": two_similar_objects,
}


def scenario(name: str) -> Sequence[TrackSample]:
    """Look up a single-sequence scenario by name, failing loudly on a typo.

    `reconnect_reuses_a_track_id` is not reachable here -- it returns a pair
    of sequences, not one; call it directly.
    """
    try:
        return SCENARIOS[name]()
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(
            f"unknown scenario {name!r}; known scenarios: {known}, or call "
            "reconnect_reuses_a_track_id() directly (it returns a pair)"
        ) from None


__all__ = [
    "DEVICE",
    "EPOCH",
    "EPOCH_AFTER_RECONNECT",
    "FRAME_INTERVAL",
    "SCENARIOS",
    "SESSION",
    "T0",
    "brief_hand_occlusion",
    "keys_carried_out_never_replaced",
    "keys_carried_to_another_room_and_set_down",
    "keys_placed_on_table",
    "object_visible_never_touched",
    "reconnect_reuses_a_track_id",
    "scenario",
    "two_similar_objects",
    "walking_past_without_touching",
]
