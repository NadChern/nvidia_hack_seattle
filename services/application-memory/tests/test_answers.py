"""What the assistant is allowed to say.

The last place a wrong claim can escape, so each test names the false statement
it prevents rather than the branch it covers.
"""

from __future__ import annotations

from visual_memory_memory_contract.fixtures import (
    keys_placed_and_left,
    keys_placed_then_picked_up,
)

from application_memory.domain.answers import EvidenceRef, answer_for
from application_memory.domain.reducer import reduce

OBJECT = "object-keys-01"


def a_frame_for(observations: object, media_type: str = "image/jpeg") -> EvidenceRef:
    """A reference to whatever evidence the fixture's placement actually cites.

    Derived rather than hard-coded: the reference only attaches when its id
    matches the placement, so a literal here would silently stop testing
    anything the moment the fixtures renumbered.
    """
    placed = next(o for o in observations if o.event.action == "placed")  # type: ignore[union-attr]
    evidence_id = placed.evidence[0].evidence_id
    return EvidenceRef(
        evidence_id=evidence_id,
        url=f"/v1/evidence/{evidence_id}",
        media_type=media_type,
    )


A_FRAME = a_frame_for(keys_placed_and_left())


def state_for(observations: object):  # type: ignore[no-untyped-def]
    resolved = [
        o.model_copy(update={"object": o.object.model_copy(update={"object_id": OBJECT})})
        for o in observations  # type: ignore[union-attr]
    ]
    return reduce(OBJECT, resolved).state


def test_a_picked_up_object_never_reports_a_current_location() -> None:
    """The sentence the whole product exists to produce."""
    answer = answer_for(state_for(keys_placed_then_picked_up()), label="keys", evidence=A_FRAME)

    assert answer.answer_status == "last_confirmed_only"
    assert answer.current_location is None
    assert "last confirmed" in answer.spoken_answer
    # Both halves must be present: where it was, and that it is no longer there.
    assert "coffee table" in answer.spoken_answer
    assert "picked up" in answer.spoken_answer
    assert "have not confirmed a new location" in answer.spoken_answer


def test_an_undisturbed_object_is_reported_as_confirmed() -> None:
    answer = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=A_FRAME)

    assert answer.answer_status == "confirmed"
    assert answer.current_location is not None
    assert "coffee table" in answer.spoken_answer


def test_unloadable_evidence_downgrades_a_confirmed_answer() -> None:
    """docs/04 counts a confirmed answer with no retrievable evidence as
    an unsupported confident answer. It must not be one."""
    answer = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=None)

    assert answer.answer_status == "last_confirmed_only"
    assert answer.current_location is None
    assert "cannot confirm" in answer.spoken_answer


def test_an_unknown_object_claims_nothing() -> None:
    answer = answer_for(None, label="wallet", evidence=A_FRAME)

    assert answer.answer_status == "unknown"
    assert answer.current_location is None
    assert "no record" in answer.spoken_answer


def test_an_ambiguous_label_refuses_to_choose() -> None:
    """Two objects share a name. Guessing is worse than admitting it."""
    answer = answer_for(None, label="keys", evidence=A_FRAME, candidates=("object-a", "object-b"))

    assert answer.answer_status == "ambiguous_object"
    assert answer.candidates == ("object-a", "object-b")
    assert answer.current_location is None


def test_the_answer_never_carries_an_internal_event_id() -> None:
    """An event id means something to the reducer and nothing to a listener."""
    answer = answer_for(state_for(keys_placed_then_picked_up()), label="keys", evidence=A_FRAME)

    assert answer.last_confirmed_placement is not None
    assert not hasattr(answer.last_confirmed_placement, "event_id")


def test_a_url_is_attached_only_when_the_bytes_are_retrievable() -> None:
    """A link that 404s is worse than no link.

    It looks like evidence right up until someone clicks it, which is the exact
    moment a demo falls apart.
    """
    with_frame = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=A_FRAME)
    without = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=None)

    assert with_frame.last_confirmed_placement is not None
    assert with_frame.last_confirmed_placement.evidence_url == A_FRAME.url
    assert with_frame.last_confirmed_placement.evidence_media_type == "image/jpeg"

    assert without.last_confirmed_placement is not None
    assert without.last_confirmed_placement.evidence_url is None


def test_a_clip_is_carried_the_same_way_as_a_frame() -> None:
    """Evidence is media-type agnostic: a client picks <img> or <video> from this."""
    clip = a_frame_for(keys_placed_and_left(), media_type="video/mp4")

    answer = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=clip)

    assert answer.last_confirmed_placement is not None
    assert answer.last_confirmed_placement.evidence_media_type == "video/mp4"
    assert answer.answer_status == "confirmed"


def test_a_url_for_different_evidence_is_not_attached() -> None:
    """Guards a copy-paste bug: the ref must match the placement it decorates."""
    other = EvidenceRef(
        evidence_id="ev_something_else",
        url="/v1/evidence/ev_something_else",
        media_type="image/jpeg",
    )

    answer = answer_for(state_for(keys_placed_and_left()), label="keys", evidence=other)

    assert answer.last_confirmed_placement is not None
    assert answer.last_confirmed_placement.evidence_url is None
