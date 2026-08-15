"""The models, and the invariants that make a wrong answer impossible."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from visual_memory_memory_contract.protocol import (
    SCHEMA_VERSION,
    EventDetail,
    Location,
    ObjectRef,
    Observation,
    ObservationConfidence,
    Provenance,
)

NOW = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)


def an_observation(**overrides: object) -> Observation:
    base: dict[str, object] = {
        "observation_id": "obs_01JABC",
        "idempotency_key": "glasses-01/sess_01JAB/track-42/placed/x",
        "session_id": "sess_01JAB",
        "device_id": "glasses-01",
        "media_epoch_id": "TR_VCabc123",
        "object": ObjectRef(object_id="object-keys-01", label="keys", track_id="track-42"),
        "event": EventDetail(action="placed", source="vision_pipeline", occurred_at=NOW),
        "location": Location(room="living_room", surface="coffee_table", relation="on"),
        "confidence": ObservationConfidence(event=0.91, identity=0.94),
        "provenance": Provenance(pipeline_version="vision-pipeline-v1"),
    }
    base.update(overrides)
    return Observation(**base)  # type: ignore[arg-type]


def test_a_well_formed_observation_round_trips() -> None:
    original = an_observation()

    restored = Observation.model_validate(original.model_dump(mode="json"))

    assert restored == original
    assert restored.schema_version == SCHEMA_VERSION


def test_timestamps_serialize_as_utc_with_millisecond_precision() -> None:
    """Fixed precision keeps golden fixtures byte-stable."""
    payload = an_observation().model_dump(mode="json")

    assert payload["event"]["occurred_at"] == "2026-07-29T17:42:11.240Z"


def test_a_placement_without_a_location_is_refused() -> None:
    """Trusted state that answers 'where?' with null is worse than no state."""
    with pytest.raises(ValidationError, match="requires a location"):
        an_observation(location=None)


def test_a_pickup_needs_no_location() -> None:
    """Picking something up says nothing about where it went."""
    observation = an_observation(
        event=EventDetail(action="picked_up", source="vision_pipeline", occurred_at=NOW),
        location=None,
    )

    assert observation.location is None


def test_an_unknown_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        an_observation(
            event=EventDetail(
                action="teleported",  # type: ignore[arg-type]
                source="vision_pipeline",
                occurred_at=NOW,
            )
        )


def test_confidence_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        an_observation(confidence=ObservationConfidence(event=1.4, identity=0.9))


def test_identity_may_be_unresolved_but_label_and_track_are_required() -> None:
    """object_id is null until identity resolves; the rest is not optional."""
    reference = ObjectRef(label="keys", track_id="track-42")

    assert reference.object_id is None
    with pytest.raises(ValidationError):
        ObjectRef(label="keys")  # type: ignore[call-arg]


def test_unknown_fields_are_ignored_so_a_pinned_producer_keeps_working() -> None:
    payload = an_observation().model_dump(mode="json")
    payload["a_field_from_a_later_version"] = True

    assert Observation.model_validate(payload).object.label == "keys"


def test_models_are_frozen() -> None:
    observation = an_observation()

    with pytest.raises(ValidationError):
        observation.session_id = "sess_other"  # type: ignore[misc]
