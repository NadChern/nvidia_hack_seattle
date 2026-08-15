"""The nine reducer cases docs/04-Evaluation-Plan.md requires, plus fan-out.

These run with no database, no API, and no models, because the reducer is a
pure function. That is the whole reason it is one: these are the rules that
decide what the assistant is allowed to claim, and they should be cheap enough
to run on every keystroke.

Each test names the failure it prevents rather than the code path it covers.
"""

from __future__ import annotations

import datetime as dt

from visual_memory_memory_contract.fixtures import (
    COFFEE_TABLE,
    T0,
    keys_placed_and_left,
    keys_placed_then_picked_up,
    placement_without_evidence,
    weak_placement_after_a_strong_one,
)
from visual_memory_memory_contract.protocol import Observation

from application_memory.domain.reducer import (
    LifecycleEvent,
    PromotionPolicy,
    reduce,
)

OBJECT = "object-keys-01"


def resolved(observations: tuple[Observation, ...] | list[Observation]) -> list[Observation]:
    """Stamp a resolved identity, as ingestion does before the reducer runs.

    Fixtures ship with `object_id` null because that is what Vision emits;
    identity resolution happens on the way in.
    """
    return [
        observation.model_copy(
            update={"object": observation.object.model_copy(update={"object_id": OBJECT})}
        )
        for observation in observations
    ]


def test_the_demo_scenario_never_claims_a_current_location() -> None:
    """The single most important assertion in this repository.

    Keys placed, seen, then picked up and never put down. Answering "on the
    coffee table" here is the failure the entire product exists to avoid.
    """
    result = reduce(OBJECT, resolved(keys_placed_then_picked_up()))

    assert result.state is not None
    assert result.state.current_status == "in_transit"
    assert result.state.current_location is None
    # The placement survives as history, which is what makes a truthful
    # "I last confirmed them there, but..." possible.
    assert result.state.last_confirmed_placement is not None
    assert result.state.last_confirmed_placement.surface == "coffee_table"
    assert result.state.invalidated_at is not None


def test_an_undisturbed_placement_stays_confirmed() -> None:
    result = reduce(OBJECT, resolved(keys_placed_and_left()))

    assert result.state is not None
    assert result.state.current_status == "confirmed_at_location"
    assert result.state.current_location is not None
    assert result.state.current_location.surface == "coffee_table"
    assert result.state.invalidated_at is None


# --- The nine cases from docs/04 ------------------------------------------


def test_duplicate_delivery_is_idempotent() -> None:
    """A retried POST must not apply a transition twice."""
    timeline = resolved(keys_placed_then_picked_up())

    once = reduce(OBJECT, timeline)
    twice = reduce(OBJECT, [*timeline, *timeline])

    assert once.state == twice.state


def test_late_observations_produce_the_same_final_timeline() -> None:
    """Arrival order must not change history.

    A network hiccup that delivers the pickup before the placement must not
    leave the system thinking the keys are still on the table.
    """
    timeline = resolved(keys_placed_then_picked_up())

    in_order = reduce(OBJECT, timeline)
    reversed_order = reduce(OBJECT, list(reversed(timeline)))

    assert in_order.state == reversed_order.state


def test_pickup_invalidates_the_current_location() -> None:
    timeline = resolved(keys_placed_and_left())
    confirmed = reduce(OBJECT, timeline)
    assert confirmed.state is not None
    assert confirmed.state.current_location is not None

    pickup = resolved(keys_placed_then_picked_up())[2]
    after = reduce(OBJECT, [*timeline, pickup])

    assert after.state is not None
    assert after.state.current_location is None
    assert after.state.current_status == "in_transit"


def test_observed_without_interaction_creates_no_placement() -> None:
    """Seeing something is not evidence that someone put it there."""
    sighting = resolved(keys_placed_then_picked_up())[1]
    assert sighting.event.action == "observed"

    result = reduce(OBJECT, [sighting])

    assert result.state is not None
    assert result.state.last_confirmed_placement is None
    assert result.state.current_status == "unknown"
    # It still counts as having seen the object.
    assert result.state.last_seen is not None


def test_weak_evidence_does_not_overwrite_a_strong_placement() -> None:
    """A hesitant guess must not displace a confident observation."""
    result = reduce(OBJECT, resolved(weak_placement_after_a_strong_one()))

    assert result.state is not None
    assert result.state.current_location is not None
    assert result.state.current_location.surface == "coffee_table"
    assert result.state.current_location.room == "living_room"
    assert len(result.rejected_ids) == 1


def test_a_reconnect_does_not_reuse_tracker_identity() -> None:
    """Two epochs both containing `track-1` are two different objects.

    The reducer works on one object at a time, so this asserts the guarantee it
    depends on: identity resolution must not have merged them. Feeding both
    into one timeline is exactly the bug -- the keys would appear to teleport
    to the kitchen.
    """
    from visual_memory_memory_contract.fixtures import reconnect_reuses_a_track_id

    before, after = reconnect_reuses_a_track_id()

    assert before.object.track_id == after.object.track_id
    assert before.media_epoch_id != after.media_epoch_id
    # Different labels, so identity resolution keyed on
    # (label, track_id, session_id, media_epoch_id) cannot collapse them.
    assert before.object.label != after.object.label


def test_ambiguous_identity_does_not_merge_objects() -> None:
    """An unresolved identity is stored as history and promotes nothing."""
    unresolved = keys_placed_then_picked_up()[0]
    assert unresolved.object.object_id is None

    result = reduce(OBJECT, [unresolved])

    assert result.state is None
    assert result.rejected_ids == (unresolved.observation_id,)


def test_missing_evidence_can_be_refused_outright() -> None:
    """With the strict policy, an uncorroborated placement promotes nothing.

    With the default policy it is promoted but the query layer refuses to call
    it `confirmed`, which is the softer half of the same rule.
    """
    timeline = resolved(placement_without_evidence())

    lenient = reduce(OBJECT, timeline)
    strict = reduce(OBJECT, timeline, policy=PromotionPolicy(require_evidence_for_placement=True))

    assert lenient.state is not None
    assert lenient.state.last_confirmed_placement is not None
    assert lenient.state.last_confirmed_placement.evidence_id is None
    assert strict.state is None


def test_deletion_removes_state_by_removing_history() -> None:
    """State is derived, so deleting the timeline deletes the claim.

    This is why observations are immutable and state is never stored as the
    source of truth: there is no second place a deleted memory can survive.
    """
    assert reduce(OBJECT, []).state is None


# --- Lifecycle fan-out, the docs/06 sign-off ------------------------------


def test_track_lost_while_in_transit_becomes_unknown() -> None:
    timeline = resolved(keys_placed_then_picked_up())
    lost = LifecycleEvent(
        signal_id="lc_01JABC",
        action="track_lost",
        reason="track_unsubscribed",
        occurred_at=T0 + dt.timedelta(minutes=4),
    )

    result = reduce(OBJECT, [*timeline, lost])

    assert result.state is not None
    assert result.state.current_status == "unknown"
    assert result.state.state_reason == "picked_up_then_track_lost"
    # The placement is still there to be reported as last-confirmed.
    assert result.state.last_confirmed_placement is not None


def test_track_lost_does_not_disturb_a_confirmed_object() -> None:
    """A camera disconnect is not evidence that the keys moved.

    Downgrading a good answer because the network blinked would throw away
    correct memory for nothing.
    """
    timeline = resolved(keys_placed_and_left())
    lost = LifecycleEvent(
        signal_id="lc_01JABD",
        action="track_lost",
        reason="room_disconnected",
        occurred_at=T0 + dt.timedelta(minutes=4),
    )

    result = reduce(OBJECT, [*timeline, lost])

    assert result.state is not None
    assert result.state.current_status == "confirmed_at_location"
    assert result.state.current_location is not None


def test_a_repeated_lifecycle_signal_applies_once() -> None:
    """A gateway that restarts mid-teardown re-sends the same signal."""
    timeline = resolved(keys_placed_then_picked_up())
    lost = LifecycleEvent(
        signal_id="lc_01JABC",
        action="track_lost",
        reason="track_unsubscribed",
        occurred_at=T0 + dt.timedelta(minutes=4),
    )

    once = reduce(OBJECT, [*timeline, lost])
    twice = reduce(OBJECT, [*timeline, lost, lost])

    assert once.state == twice.state


# --- Conflicts -------------------------------------------------------------


def test_simultaneous_contradictory_placements_are_flagged_not_guessed() -> None:
    """Two confident placements at the same instant in different rooms.

    One is wrong and the reducer cannot know which, so it keeps the first,
    records both, and flags it rather than silently picking a winner.
    """
    first = resolved(keys_placed_and_left())[0]
    elsewhere = first.model_copy(
        update={
            "observation_id": "obs_01JABDEMO_conflict",
            "location": COFFEE_TABLE.model_copy(update={"room": "kitchen", "surface": "counter"}),
        }
    )

    result = reduce(OBJECT, [first, elsewhere])

    assert result.conflicts
    assert set(result.conflicts[0].observation_ids) == {
        first.observation_id,
        elsewhere.observation_id,
    }
    assert result.state is not None
    assert result.state.current_location is not None
    assert result.state.current_location.room == "living_room"


def test_a_later_placement_is_ordinary_history_not_a_conflict() -> None:
    """Moving an object is normal. Only a same-instant disagreement is a conflict."""
    first = resolved(keys_placed_and_left())[0]
    later = first.model_copy(
        update={
            "observation_id": "obs_01JABDEMO_later",
            "event": first.event.model_copy(update={"occurred_at": T0 + dt.timedelta(minutes=10)}),
            "location": COFFEE_TABLE.model_copy(update={"room": "kitchen", "surface": "counter"}),
        }
    )

    result = reduce(OBJECT, [first, later])

    assert not result.conflicts
    assert result.state is not None
    assert result.state.current_location is not None
    assert result.state.current_location.room == "kitchen"
