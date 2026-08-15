"""The VLM verifier's judgment, against a scripted model.

No model runs here. What is being tested is the part that decides what a
model's answer *means* -- and in particular the part that refuses to let a
confident-sounding reply become memory when it should not.

The scenarios are the ones measured on `media/clips`: a pickup the model
reads correctly, a control window where it correctly reports nothing, and
the several ways a reply can be useless.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence

import pytest
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    Point2D,
)

from vision_worker.verify.vlm import (
    CONFIRMED,
    CONTRADICTED,
    MALFORMED,
    NOTHING_HAPPENED,
    UNCERTAIN,
    UNREACHABLE,
    VlmVerifier,
)

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
_REF = DetectorRef(name="fixture", checkpoint="n/a", revision="v1")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def a_candidate(action: str) -> CandidateEvent:
    return CandidateEvent(
        candidate_id="cand_1",
        session_id="sess_1",
        device_id="glasses-01",
        media_epoch_id="TR_VCaaa",
        track_id="track-1",
        label="keys",
        action=action,  # type: ignore[arg-type]
        window=EvidenceWindow(
            window_started_at=T0, window_ended_at=T0 + dt.timedelta(seconds=4), frame_count=16
        ),
        object_candidate=Detection(
            label="keys",
            confidence=0.8,
            box=BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),
            centroid=Point2D(x=0.5, y=0.5),
        ),
        detector=_REF,
        tracker=_REF,
        state_machine_version="v1",
        pipeline_version="v1",
    )


def scripted(verifier: VlmVerifier, reply: str | dict[str, object]) -> VlmVerifier:
    """Replace the HTTP call with a canned reply."""
    body = reply if isinstance(reply, str) else json.dumps(reply)

    def _ask(label: str, frames: Sequence[bytes]) -> str:
        del label, frames
        return body

    verifier._ask_blocking = _ask  # type: ignore[assignment]  # noqa: SLF001
    return verifier


def failing(verifier: VlmVerifier, exc: Exception) -> VlmVerifier:
    def _ask(label: str, frames: Sequence[bytes]) -> str:
        del label, frames
        raise exc

    verifier._ask_blocking = _ask  # type: ignore[assignment]  # noqa: SLF001
    return verifier


FRAMES = (b"jpeg",) * 16


# --- Resolving a question ---------------------------------------------------


async def test_a_vanished_object_the_model_saw_taken_becomes_a_pickup() -> None:
    """Clip 2, the demo case. The pipeline knows only that the keys stopped
    being detected; the model read the frames and says a hand took them."""
    verifier = scripted(
        VlmVerifier(),
        {
            "answer": "A hand picks up the keys from the white table.",
            "action": "picked_up",
            "location_description": "a white table next to a tablet",
            "certain": True,
        },
    )

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "confirmed"
    assert result.reason_code == CONFIRMED
    assert result.resolved_action == "picked_up"
    assert result.description == "a white table next to a tablet"


async def test_a_vanished_object_still_sitting_there_is_rejected() -> None:
    """You walked away from your keys and they never moved. Nothing to
    record, and the existing placement stands."""
    verifier = scripted(
        VlmVerifier(),
        {"answer": "The keys remain on the desk.", "action": "nothing_happened", "certain": True},
    )

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "rejected"
    assert result.reason_code == NOTHING_HAPPENED
    assert result.resolved_action is None


async def test_a_question_the_model_cannot_answer_stays_unverified() -> None:
    """The important one. An unresolved `vanished` can never become an
    observation -- `emit/memory.py` refuses it -- so uncertainty here costs a
    diagnostic and nothing else."""
    verifier = scripted(
        VlmVerifier(),
        {"answer": "Too blurry to tell.", "action": "unknown", "certain": False},
    )

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "unverified"
    assert result.reason_code == UNCERTAIN
    assert result.resolved_action is None


async def test_confidence_false_is_honoured_even_with_a_definite_action() -> None:
    """A model that names an action but admits it is unsure must not be
    taken at its word -- that combination is exactly how a plausible story
    gets written to memory."""
    verifier = scripted(
        VlmVerifier(),
        {"answer": "Possibly a hand.", "action": "picked_up", "certain": False},
    )

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "unverified"


# --- Judging a claim --------------------------------------------------------


async def test_a_claim_the_model_agrees_with_is_confirmed() -> None:
    verifier = scripted(
        VlmVerifier(),
        {"answer": "The keys are set down.", "action": "placed", "certain": True},
    )

    result = await verifier.verify(a_candidate("placed"), frames=FRAMES)

    assert result.outcome == "confirmed"
    assert result.resolved_action is None, "no revision needed when the claim was right"


async def test_a_claim_the_model_contradicts_is_corrected_not_confirmed() -> None:
    """The state machine guessed from pixels; the model looked at the scene.
    When they disagree about *which* event, the one that looked wins."""
    verifier = scripted(
        VlmVerifier(),
        {"answer": "The keys are lifted, not set down.", "action": "picked_up", "certain": True},
    )

    result = await verifier.verify(a_candidate("placed"), frames=FRAMES)

    assert result.outcome == "confirmed"
    assert result.reason_code == CONTRADICTED
    assert result.resolved_action == "picked_up"


async def test_a_claim_the_model_saw_nothing_for_is_rejected() -> None:
    """Clip 1's false pickups: the camera panned and the pipeline called it
    a pickup. The model watched the same frames and saw no event."""
    verifier = scripted(
        VlmVerifier(),
        {
            "answer": "Nothing interacts with the keys.",
            "action": "nothing_happened",
            "certain": True,
        },
    )

    result = await verifier.verify(a_candidate("picked_up"), frames=FRAMES)

    assert result.outcome == "rejected"
    assert result.reason_code == NOTHING_HAPPENED


# --- When the model is not cooperating --------------------------------------


async def test_an_unreachable_model_leaves_the_candidate_unverified() -> None:
    """A model that is down must not become an event, in either direction."""
    verifier = failing(VlmVerifier(), TimeoutError("no route"))

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "unverified"
    assert result.reason_code == UNREACHABLE


async def test_a_reply_that_is_not_json_is_unverified_not_a_guess() -> None:
    verifier = scripted(VlmVerifier(), "I think somebody took them, probably.")

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "unverified"
    assert result.reason_code == MALFORMED


async def test_json_wrapped_in_reasoning_is_still_read() -> None:
    """Structured output and chain-of-thought have a history of fighting.
    A model that narrates before answering should not cost a candidate."""
    verifier = scripted(
        VlmVerifier(),
        "Let me think about this... the hand enters at frame 4.\n"
        '{"answer": "A hand takes the keys.", "action": "picked_up", "certain": true}',
    )

    result = await verifier.verify(a_candidate("vanished"), frames=FRAMES)

    assert result.outcome == "confirmed"
    assert result.resolved_action == "picked_up"


async def test_an_empty_window_is_never_sent_to_the_model() -> None:
    called = False

    def _ask(label: str, frames: Sequence[bytes]) -> str:  # pragma: no cover
        nonlocal called
        called = True
        return "{}"

    verifier = VlmVerifier()
    verifier._ask_blocking = _ask  # type: ignore[assignment]  # noqa: SLF001

    result = await verifier.verify(a_candidate("vanished"), frames=())

    assert result.outcome == "unverified"
    assert not called
