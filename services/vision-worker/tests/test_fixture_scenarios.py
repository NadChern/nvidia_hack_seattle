"""Replays every golden scenario from `packages/vision-contract/fixtures.py`
through the real stability machine at its *default* thresholds -- not the
shortened `FAST` config `test_stability.py` uses for speed.

This is what actually proves the fixtures are correct: `fixtures.py` cannot
import `domain.stability` (it must stay dependency-light for Person 2's
harness), so its docstrings only *claim* an expected outcome. This is where
that claim is checked against the real state machine, at production
defaults, closing the loop docs/05-Team-Split.md requires: "an interface is
complete only when the provider fixture passes in the consumer's harness."
"""

from __future__ import annotations

from collections.abc import Sequence

from visual_memory_vision_contract import fixtures
from visual_memory_vision_contract.protocol import TrackSample

from vision_worker.domain.stability import TrackRegistry


def replay_into(registry: TrackRegistry, samples: Sequence[TrackSample]) -> list[tuple[str, str]]:
    """Return `(track_id, action)` pairs in emission order."""
    emitted: list[tuple[str, str]] = []
    for sample in samples:
        result = registry.observe(sample.track_id, sample)
        if result.action is not None:
            emitted.append((sample.track_id, result.action))
    return emitted


def replay(samples: Sequence[TrackSample]) -> list[tuple[str, str]]:
    """`replay_into` with a fresh registry at default thresholds."""
    return replay_into(TrackRegistry(), samples)


def test_keys_placed_on_table() -> None:
    actions = [action for _, action in replay(fixtures.keys_placed_on_table())]
    assert actions == ["observed", "placed"]


def test_keys_carried_out_never_replaced() -> None:
    """The demo case, at real production thresholds."""
    actions = [action for _, action in replay(fixtures.keys_carried_out_never_replaced())]
    assert actions == ["observed", "placed", "picked_up"]
    assert "placed" not in actions[actions.index("picked_up") :]


def test_keys_carried_to_another_room_and_set_down() -> None:
    actions = [action for _, action in replay(fixtures.keys_carried_to_another_room_and_set_down())]
    assert actions == ["observed", "placed", "picked_up", "placed"]


def test_object_visible_never_touched() -> None:
    """Shorter than passive_confirmation_frames -- must never promote."""
    actions = [action for _, action in replay(fixtures.object_visible_never_touched())]
    assert actions == ["observed"]


def test_walking_past_without_touching() -> None:
    actions = [action for _, action in replay(fixtures.walking_past_without_touching())]
    assert actions == ["observed"]


def test_brief_hand_occlusion_does_not_reset_on_reappearance() -> None:
    """The gap (missing frame_index values) must be fed as sample=None to
    the registry -- fixtures.py documents this; this test performs it."""
    samples = fixtures.brief_hand_occlusion()
    registry = TrackRegistry()
    emitted: list[str] = []

    previous_index: int | None = None
    for sample in samples:
        if previous_index is not None:
            for _ in range(sample.frame_index - previous_index - 1):
                registry.observe("track-1", None)
        result = registry.observe("track-1", sample)
        if result.action is not None:
            emitted.append(result.action)
        previous_index = sample.frame_index

    assert emitted == ["observed", "placed"]


def test_two_similar_objects_both_independently_settle() -> None:
    emitted = replay(fixtures.two_similar_objects())
    by_track: dict[str, list[str]] = {}
    for track_id, action in emitted:
        by_track.setdefault(track_id, []).append(action)

    assert by_track["track-1"] == ["observed", "placed"]
    assert by_track["track-2"] == ["observed", "placed"]


def test_reconnect_reuses_a_track_id_without_merging_state() -> None:
    before, after = fixtures.reconnect_reuses_a_track_id()

    registry = TrackRegistry()
    before_actions = [action for _, action in replay_into(registry, before)]

    # The epoch boundary: a live pipeline calls this on every epoch_started.
    registry.reset()

    after_actions = [action for _, action in replay_into(registry, after)]

    assert before_actions == ["observed", "placed"]
    # If state had leaked across the reset, the second sequence's first
    # sample would not be "observed" -- it would inherit "at_rest" and
    # never re-announce, or worse, its motion would look like a pickup of
    # the *first* object.
    assert after_actions == ["observed", "placed"]
