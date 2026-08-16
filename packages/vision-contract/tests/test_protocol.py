"""The models, and the invariants that make a wrong candidate impossible."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from visual_memory_vision_contract.protocol import (
    SCHEMA_VERSION,
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    IdentityMatch,
    VerifierResult,
)

NOW = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
LATER = NOW + dt.timedelta(seconds=3)

_DETECTOR = DetectorRef(name="yoloe-11s-seg", checkpoint="yoloe-11s-seg.pt", revision="rev-1")
_TRACKER = DetectorRef(name="botsort", checkpoint="botsort.yaml", revision="rev-1")


def a_detection(**overrides: object) -> Detection:
    base: dict[str, object] = {
        "label": "keys",
        "confidence": 0.91,
        "box": BoundingBox(x_min=0.41, y_min=0.52, x_max=0.49, y_max=0.58),
        "centroid": {"x": 0.45, "y": 0.55},
    }
    base.update(overrides)
    return Detection(**base)  # type: ignore[arg-type]


def a_candidate(**overrides: object) -> CandidateEvent:
    base: dict[str, object] = {
        "candidate_id": "cand_01JABC",
        "session_id": "sess_01JAB",
        "device_id": "glasses-01",
        "media_epoch_id": "TR_VCabc123",
        "track_id": "track-42",
        "label": "keys",
        "action": "placed",
        "window": EvidenceWindow(window_started_at=NOW, window_ended_at=LATER, frame_count=72),
        "object_candidate": a_detection(),
        "detector": _DETECTOR,
        "tracker": _TRACKER,
        "state_machine_version": "vision-stability-v1",
        "pipeline_version": "vision-pipeline-v1",
    }
    base.update(overrides)
    return CandidateEvent(**base)  # type: ignore[arg-type]


def test_a_well_formed_candidate_round_trips() -> None:
    original = a_candidate()

    restored = CandidateEvent.model_validate(original.model_dump(mode="json"))

    assert restored == original
    assert restored.schema_version == SCHEMA_VERSION


def test_timestamps_serialize_as_utc_with_millisecond_precision() -> None:
    payload = a_candidate().model_dump(mode="json")

    assert payload["window"]["window_started_at"] == "2026-07-29T17:42:11.240Z"


def test_hand_candidate_defaults_to_none() -> None:
    """The field docs/06 lists but this implementation never populates."""
    candidate = a_candidate()

    assert candidate.hand_candidate is None


def test_identity_none_and_identity_abstained_are_distinct() -> None:
    did_not_run = a_candidate()
    abstained = a_candidate(identity=IdentityMatch(reason_code="ambiguous"))

    assert did_not_run.identity is None
    assert abstained.identity is not None
    assert abstained.identity.object_id is None


def test_an_unknown_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        a_candidate(action="teleported")


def test_confidence_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        a_candidate(object_candidate=a_detection(confidence=1.4))


def test_a_frame_count_below_one_is_rejected() -> None:
    """A window over zero frames is not evidence of anything."""
    with pytest.raises(ValidationError):
        EvidenceWindow(window_started_at=NOW, window_ended_at=LATER, frame_count=0)


def test_verifier_outcome_must_be_one_of_exactly_three_values() -> None:
    with pytest.raises(ValidationError):
        VerifierResult(
            candidate_id="cand_01JABC",
            outcome="maybe",  # type: ignore[arg-type]
            reason_code="x",
            latency_ms=1.0,
            verifier=_DETECTOR,
            occurred_at=NOW,
        )


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValidationError):
        VerifierResult(
            candidate_id="cand_01JABC",
            outcome="confirmed",
            reason_code="x",
            latency_ms=-1.0,
            verifier=_DETECTOR,
            occurred_at=NOW,
        )


def test_unknown_fields_are_ignored_so_a_pinned_producer_keeps_working() -> None:
    payload = a_candidate().model_dump(mode="json")
    payload["a_field_from_a_later_version"] = True

    assert CandidateEvent.model_validate(payload).label == "keys"


def test_models_are_frozen() -> None:
    candidate = a_candidate()

    with pytest.raises(ValidationError):
        candidate.session_id = "sess_other"  # type: ignore[misc]
