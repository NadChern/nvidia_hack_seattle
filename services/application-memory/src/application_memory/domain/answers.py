"""Turning trusted state into an answer a person hears.

This is the last place a wrong claim can escape, so the rules are explicit and
the wording is generated rather than modelled. `docs/01` permits the
conversational layer to shorten this text but requires it to preserve
`answer_status`, the uncertainty, and any invalidation -- dropping the second
half of "I last confirmed them there, but they were picked up afterward" turns
a truthful answer into a false one.

Pure: no database, no clock. Whether evidence is retrievable is passed in,
because checking that is I/O and this module does none.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from visual_memory_memory_contract.protocol import (
    AnsweredPlacement,
    ObjectState,
    QueryResponse,
)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A retrievable piece of evidence, resolved by the caller.

    Constructing this needs a database read and a filesystem check, both of
    which are I/O, so the caller does it and this module stays pure. Its
    existence *is* the "loadable" signal: a caller that cannot produce the
    bytes passes None rather than a reference that would 404.
    """

    evidence_id: str
    url: str
    media_type: str


def _clock(moment: dt.datetime) -> str:
    """Render a time the way a person would say it."""
    return moment.astimezone().strftime("%H:%M")


def _place(placement: AnsweredPlacement | None, *, room: str | None = None) -> str:
    """Describe a location in words, degrading gracefully as detail runs out.

    Never invents. A missing surface produces a vaguer sentence, not a guessed
    one, because docs/06 forbids filling an unknown field with a plausible
    label to satisfy the schema.
    """
    if placement is None:
        return f"in the {room.replace('_', ' ')}" if room else "somewhere I did not record"

    surface = placement.surface.replace("_", " ") if placement.surface else None
    where = placement.room.replace("_", " ") if placement.room else None
    relation = placement.relation if placement.relation not in (None, "unknown") else "on"

    if surface and where:
        return f"{relation} the {where} {surface}"
    if surface:
        return f"{relation} the {surface}"
    if where:
        return f"in the {where}"
    return "somewhere I did not record"


def _to_answered(placement: object, evidence: EvidenceRef | None) -> AnsweredPlacement | None:
    """Drop the internal event id and attach a retrievable URL.

    Trusted state carries the event id so a transition can be traced to its
    cause; an answer does not, because it means nothing to whoever is
    listening. The URL goes the other way: it is useless in storage and
    essential in an answer.
    """
    if placement is None:
        return None
    answered = AnsweredPlacement.model_validate(
        placement.model_dump(mode="json")  # type: ignore[attr-defined]
    )
    if evidence is None or evidence.evidence_id != answered.evidence_id:
        return answered
    return answered.model_copy(
        update={"evidence_url": evidence.url, "evidence_media_type": evidence.media_type}
    )


def answer_for(
    state: ObjectState | None,
    *,
    label: str,
    evidence: EvidenceRef | None = None,
    candidates: tuple[str, ...] = (),
) -> QueryResponse:
    """Decide what may be claimed about one object.

    `evidence` gates the strongest answer: docs/04 counts a confirmed answer
    whose evidence cannot be retrieved as an "unsupported confident answer", so
    None -- meaning the bytes are not there -- downgrades the claim rather than
    being ignored. Passing a reference is the caller asserting it checked.
    """
    evidence_is_loadable = evidence is not None
    if candidates:
        return QueryResponse(
            answer_status="ambiguous_object",
            spoken_answer=(
                f"I know about more than one thing called {label}, "
                "so I cannot say which one you mean."
            ),
            candidates=candidates,
        )

    if state is None:
        return QueryResponse(
            answer_status="unknown",
            spoken_answer=f"I have no record of the {label}.",
        )

    placement = _to_answered(state.last_confirmed_placement, evidence)

    if state.current_status == "confirmed_at_location" and state.current_location is not None:
        where = _place(placement, room=state.current_location.room)
        when = (
            _clock(state.last_confirmed_placement.occurred_at)
            if state.last_confirmed_placement
            else None
        )

        if not evidence_is_loadable:
            # The state is confirmed but nothing corroborates it. Reporting it
            # as confirmed is exactly the metric docs/04 tracks, so it is
            # reported as the weaker, still-true claim instead.
            return QueryResponse(
                object_id=state.object_id,
                answer_status="last_confirmed_only",
                current_status=state.current_status,
                current_location=None,
                last_confirmed_placement=placement,
                spoken_answer=(
                    f"I last recorded the {label} {where}"
                    + (f" at {when}" if when else "")
                    + ", but I no longer have the picture that showed it, "
                    "so I cannot confirm that."
                ),
            )

        return QueryResponse(
            object_id=state.object_id,
            answer_status="confirmed",
            current_status=state.current_status,
            current_location=state.current_location,
            last_confirmed_placement=placement,
            spoken_answer=(
                f"The {label} are {where}" if label.endswith("s") else f"The {label} is {where}"
            )
            + (f", confirmed at {when}." if when else "."),
        )

    if placement is not None:
        # The heart of the product: a location that was true and no longer is.
        # Both halves of this sentence must survive any rewording downstream.
        where = _place(placement)
        when = _clock(placement.occurred_at)
        moved = "picked up" if state.state_reason and "picked_up" in state.state_reason else "moved"
        return QueryResponse(
            object_id=state.object_id,
            answer_status="last_confirmed_only",
            current_status=state.current_status,
            current_location=None,
            last_confirmed_placement=placement,
            spoken_answer=(
                f"I last confirmed the {label} {where} at {when}, "
                f"but they were {moved} afterward and I have not confirmed a new location."
                if label.endswith("s")
                else f"I last confirmed the {label} {where} at {when}, "
                f"but it was {moved} afterward and I have not confirmed a new location."
            ),
        )

    if state.last_seen is not None:
        where = _place(None, room=state.last_seen.room)
        at = _clock(state.last_seen.occurred_at)
        return QueryResponse(
            object_id=state.object_id,
            answer_status="unknown",
            current_status=state.current_status,
            spoken_answer=(
                f"I last saw the {label} {where} at {at}, but I never confirmed "
                "where it was put down."
            ),
        )

    return QueryResponse(
        object_id=state.object_id,
        answer_status="unknown",
        current_status=state.current_status,
        spoken_answer=(f"I have seen the {label} but never confirmed where it was put down."),
    )


__all__ = ["EvidenceRef", "answer_for"]
