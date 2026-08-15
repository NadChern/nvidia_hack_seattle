"""The deterministic state reducer.

This is the product. Everything else in this service moves bytes around so that
this function can decide what the assistant is allowed to claim.

It is a pure function of a timeline: no database, no clock, no I/O, no
network. `tests/test_domain_isolation.py` asserts that mechanically, because
one convenient import would cost the ability to test the rules that matter in
milliseconds.

**Recompute, never mutate.** `docs/06-Data-Contract.md` requires that late
events recompute the affected object's timeline without silently reordering
history, so state is always derived by replaying every entry in
`(occurred_at, id)` order. Duplicate delivery, out-of-order arrival, and replay
after a restart then produce identical state by construction rather than by a
special case for each.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeAlias

from visual_memory_memory_contract.protocol import (
    LastSeen,
    LifecycleAction,
    LifecycleReason,
    Location,
    ObjectState,
    Observation,
    Placement,
)


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Thresholds an observation must clear to touch trusted state.

    Configuration, not model constants: docs/04 requires the threshold set used
    for an evaluation run to be recorded, which is impossible if the numbers
    are baked into the reducer.
    """

    min_event_confidence: float = 0.7
    min_identity_confidence: float = 0.7
    #: When true a placement with no evidence is not promoted at all. When
    #: false it is promoted but the query layer refuses to call it `confirmed`,
    #: because an answer whose evidence cannot be loaded is exactly the
    #: "unsupported confident answer" docs/04 measures.
    require_evidence_for_placement: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """A track or session ending, already resolved to one object.

    The gateway cannot name objects, so it scopes signals by media epoch and
    Memory fans them out. By the time a signal reaches the reducer that
    fan-out has happened and this is about exactly one object.
    """

    signal_id: str
    action: LifecycleAction
    reason: LifecycleReason
    occurred_at: dt.datetime
    media_epoch_id: str | None = None


TimelineEntry: TypeAlias = Observation | LifecycleEvent


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two promotable observations that cannot both be true.

    Retained rather than resolved silently: docs/06 requires conflicting
    high-confidence observations to keep both records and be flagged for
    evaluation.
    """

    reason: str
    observation_ids: tuple[str, ...]
    occurred_at: dt.datetime


@dataclass(frozen=True, slots=True)
class ReduceResult:
    """The derived state, plus what the reducer chose not to trust.

    `state` matches the documented `ObjectState` shape exactly. The rejected and
    conflicting records live beside it rather than inside it, because docs/06's
    trusted-state contract has no field for them and widening it would put
    untrusted data in the structure whose entire job is to be trusted.
    """

    state: ObjectState | None
    promoted_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    conflicts: tuple[Conflict, ...] = ()


@dataclass(slots=True)
class _Accumulator:
    """Mutable scratch space while folding. Never escapes this module."""

    status: str = "unknown"
    location: Location | None = None
    current_event_id: str | None = None
    state_reason: str | None = None
    invalidated_at: dt.datetime | None = None
    last_confirmed_placement: Placement | None = None
    last_seen: LastSeen | None = None
    updated_at: dt.datetime | None = None
    last_promoted_action: str | None = None
    promoted: list[str] = field(default_factory=list[str])
    rejected: list[str] = field(default_factory=list[str])
    conflicts: list[Conflict] = field(default_factory=list[Conflict])


def _entry_sort_key(entry: TimelineEntry) -> tuple[dt.datetime, str]:
    """Order by event time, breaking ties on identifier.

    Identifiers are ULIDs, so a tie-break on id is a tie-break on creation
    order. Without a total order two observations sharing an `occurred_at`
    could reduce differently between runs, and a memory that changes on replay
    is not a memory.
    """
    if isinstance(entry, LifecycleEvent):
        return (entry.occurred_at, entry.signal_id)
    return (entry.event.occurred_at, entry.observation_id)


def is_promotable(observation: Observation, policy: PromotionPolicy) -> bool:
    """Whether an observation may touch trusted state.

    Failing this is not an error. The observation is still stored as history --
    docs/06 requires low-confidence observations to be retained without
    promotion -- it simply does not get to change what the assistant claims.
    """
    if observation.object.object_id is None:
        # Identity unresolved. Promoting would attribute an event to an object
        # nobody has established exists.
        return False
    if observation.confidence.event < policy.min_event_confidence:
        return False
    if observation.confidence.identity < policy.min_identity_confidence:
        return False
    if (
        policy.require_evidence_for_placement
        and observation.event.action == "placed"
        and not observation.evidence
    ):
        return False
    return True


def _evidence_id(observation: Observation) -> str | None:
    return observation.evidence[0].evidence_id if observation.evidence else None


def _apply_observation(
    acc: _Accumulator, observation: Observation, policy: PromotionPolicy
) -> None:
    if not is_promotable(observation, policy):
        acc.rejected.append(observation.observation_id)
        return

    acc.promoted.append(observation.observation_id)
    action = observation.event.action
    at = observation.event.occurred_at

    # Every promotable observation is a sighting, whatever else it is.
    acc.last_seen = LastSeen(
        occurred_at=at,
        room=observation.location.room if observation.location else None,
        evidence_id=_evidence_id(observation),
    )
    acc.updated_at = at

    if action == "placed":
        placement = Placement(
            event_id=observation.observation_id,
            occurred_at=at,
            room=observation.location.room if observation.location else None,
            surface=observation.location.surface if observation.location else None,
            relation=observation.location.relation if observation.location else None,
            evidence_id=_evidence_id(observation),
        )
        if _contradicts_simultaneous_placement(acc, placement, observation):
            return
        acc.status = "confirmed_at_location"
        acc.location = observation.location
        acc.last_confirmed_placement = placement
        acc.current_event_id = observation.observation_id
        acc.state_reason = "placed"
        # A fresh confirmation clears any earlier invalidation: the object has
        # been seen somewhere definite since.
        acc.invalidated_at = None

    elif action == "picked_up":
        acc.status = "in_transit"
        acc.location = None
        acc.current_event_id = observation.observation_id
        acc.state_reason = "picked_up"
        # The last confirmed placement survives as history -- that is what
        # makes "I last confirmed them there, but they were picked up" possible
        # -- but it is no longer current.
        acc.invalidated_at = at

    elif action == "carried":
        acc.status = "in_transit"
        acc.location = None
        acc.current_event_id = observation.observation_id
        acc.state_reason = "carried"
        if acc.invalidated_at is None:
            acc.invalidated_at = at

    elif action == "observed":
        # Explicitly does not create a placement. Seeing something is not
        # evidence that it was put there, and treating it as such would invent
        # placements from ordinary sightings.
        acc.current_event_id = observation.observation_id

    acc.last_promoted_action = action


def _contradicts_simultaneous_placement(
    acc: _Accumulator, placement: Placement, observation: Observation
) -> bool:
    """Flag two equally-timed placements that disagree, and keep the first.

    Later normally wins, which is ordinary history. The unresolvable case is
    two high-confidence placements at the same instant in different places:
    one of them is wrong and the reducer cannot know which.
    """
    previous = acc.last_confirmed_placement
    if previous is None or previous.occurred_at != placement.occurred_at:
        return False
    if (previous.room, previous.surface, previous.relation) == (
        placement.room,
        placement.surface,
        placement.relation,
    ):
        return False
    acc.conflicts.append(
        Conflict(
            reason="two high-confidence placements at the same instant disagree",
            observation_ids=(previous.event_id, observation.observation_id),
            occurred_at=placement.occurred_at,
        )
    )
    return True


def _apply_lifecycle(acc: _Accumulator, event: LifecycleEvent) -> None:
    """A track or session went away.

    Only meaningful while in transit. An object confirmed on a table does not
    stop being there because the camera disconnected -- claiming otherwise
    would throw away a good answer for no reason.
    """
    if acc.status != "in_transit":
        return
    acc.status = "unknown"
    acc.location = None
    acc.current_event_id = event.signal_id
    acc.state_reason = (
        f"{acc.last_promoted_action}_then_{event.action}"
        if acc.last_promoted_action
        else event.action
    )
    acc.invalidated_at = acc.invalidated_at or event.occurred_at
    acc.updated_at = event.occurred_at


def reduce(
    object_id: str,
    entries: Iterable[TimelineEntry],
    *,
    policy: PromotionPolicy | None = None,
) -> ReduceResult:
    """Derive trusted state for one object from its complete timeline.

    Pass every entry every time. This is deliberately not an incremental
    update: replaying is what makes late arrivals, duplicates, and restarts
    produce identical state without a special case for each.
    """
    policy = policy or PromotionPolicy()
    ordered: Sequence[TimelineEntry] = sorted(entries, key=_entry_sort_key)
    if not ordered:
        return ReduceResult(state=None)

    acc = _Accumulator()
    seen: set[str] = set()

    for entry in ordered:
        # Duplicate delivery is idempotent. Ingestion also rejects repeats by
        # idempotency key, but the reducer must not depend on that: it is
        # replayed from stored history, where a duplicate would silently apply
        # a transition twice.
        identifier = entry.signal_id if isinstance(entry, LifecycleEvent) else entry.observation_id
        if identifier in seen:
            continue
        seen.add(identifier)

        if isinstance(entry, LifecycleEvent):
            _apply_lifecycle(acc, entry)
        else:
            _apply_observation(acc, entry, policy)

    if acc.updated_at is None:
        # Nothing cleared the bar. History exists; trusted state does not.
        return ReduceResult(
            state=None,
            rejected_ids=tuple(acc.rejected),
            conflicts=tuple(acc.conflicts),
        )

    state = ObjectState(
        object_id=object_id,
        current_status=acc.status,  # type: ignore[arg-type]
        current_location=acc.location,
        current_event_id=acc.current_event_id,
        state_reason=acc.state_reason,
        invalidated_at=acc.invalidated_at,
        last_confirmed_placement=acc.last_confirmed_placement,
        last_seen=acc.last_seen,
        updated_at=acc.updated_at,
    )
    return ReduceResult(
        state=state,
        promoted_ids=tuple(acc.promoted),
        rejected_ids=tuple(acc.rejected),
        conflicts=tuple(acc.conflicts),
    )


def with_thresholds(policy: PromotionPolicy, **changes: float | bool) -> PromotionPolicy:
    """Return a policy with individual thresholds overridden, for evaluation runs."""
    return replace(policy, **changes)  # type: ignore[arg-type]


__all__ = [
    "Conflict",
    "LifecycleEvent",
    "PromotionPolicy",
    "ReduceResult",
    "TimelineEntry",
    "is_promotable",
    "reduce",
    "with_thresholds",
]
