"""Golden scenarios, shared by producer and consumer.

Both sides assert against these same objects: Vision checks that what it emits
matches the shape, and Memory checks that ingesting them produces the expected
answer. Per `docs/05-Team-Split.md` an interface is complete only when the
provider fixture passes inside the consumer's harness, and shared fixtures are
what makes that mechanical rather than aspirational.

Everything here is deterministic -- fixed identifiers, fixed timestamps -- so a
byte comparison is meaningful and a diff points at a real change.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from visual_memory_memory_contract.protocol import (
    DetectorRef,
    EventDetail,
    Evidence,
    Location,
    ObjectRef,
    Observation,
    ObservationConfidence,
    Provenance,
)

#: A fixed wall-clock origin. 10:42 local in the demo narrative.
T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)

SESSION = "sess_01JABDEMO0000000000000000"
DEVICE = "glasses-01"
EPOCH = "TR_VCabc123"
#: A second epoch, produced by a reconnect. Same wearer, new track SIDs.
EPOCH_AFTER_RECONNECT = "TR_VCdef456"

_PROVENANCE = Provenance(
    detector=DetectorRef(name="sam-3.1", checkpoint="exact-checkpoint-id", revision="rev-1"),
    verifier=DetectorRef(
        name="qwen3-vl-8b-instruct", checkpoint="Qwen/Qwen3-VL-8B-Instruct", revision="rev-1"
    ),
    prompt_version="placement-verifier-v1",
    pipeline_version="vision-pipeline-v1",
)


def _evidence(index: int, at: dt.datetime) -> Evidence:
    return Evidence(
        evidence_id=f"ev_01JABDEMO{index:022d}",
        captured_at=at,
        sha256=f"{index:064x}",
        frame_index=1242 + index,
    )


def _observation(
    *,
    index: int,
    action: str,
    at: dt.datetime,
    track_id: str,
    epoch: str,
    location: Location | None = None,
    event_confidence: float = 0.91,
    identity_confidence: float = 0.94,
    with_evidence: bool = True,
) -> Observation:
    return Observation(
        observation_id=f"obs_01JABDEMO{index:022d}",
        idempotency_key=f"{DEVICE}/{SESSION}/{track_id}/{action}/{index}",
        session_id=SESSION,
        device_id=DEVICE,
        media_epoch_id=epoch,
        object=ObjectRef(object_id=None, label="keys", track_id=track_id),
        event=EventDetail(
            action=action,  # type: ignore[arg-type]
            source="vision_pipeline",
            occurred_at=at,
        ),
        location=location,
        confidence=ObservationConfidence(
            event=event_confidence,
            identity=identity_confidence,
            room=0.88,
            surface=0.90,
            relation=0.82,
        ),
        evidence=(_evidence(index, at),) if with_evidence else (),
        provenance=_PROVENANCE,
    )


COFFEE_TABLE = Location(
    room="living_room", surface="coffee_table", relation="on", description="beside the laptop"
)


def keys_placed_then_picked_up() -> Sequence[Observation]:
    """The demo scenario, and the one that must never answer `confirmed`.

    Keys are placed on the coffee table, seen there, then picked up and never
    put down again. The truthful answer is that the last confirmed placement is
    the coffee table *and* that it was invalidated afterwards.

    A system that answers "on the coffee table" here is wrong in the way this
    whole project exists to avoid.
    """
    return (
        _observation(
            index=1, action="placed", at=T0, track_id="track-42", epoch=EPOCH, location=COFFEE_TABLE
        ),
        _observation(
            index=2,
            action="observed",
            at=T0 + dt.timedelta(seconds=45),
            track_id="track-42",
            epoch=EPOCH,
            location=COFFEE_TABLE,
        ),
        _observation(
            index=3,
            action="picked_up",
            at=T0 + dt.timedelta(minutes=3),
            track_id="track-42",
            epoch=EPOCH,
        ),
    )


def keys_placed_and_left() -> Sequence[Observation]:
    """Placed and never disturbed. The only scenario that may answer `confirmed`."""
    return (
        _observation(
            index=11,
            action="placed",
            at=T0,
            track_id="track-42",
            epoch=EPOCH,
            location=COFFEE_TABLE,
        ),
        _observation(
            index=12,
            action="observed",
            at=T0 + dt.timedelta(seconds=45),
            track_id="track-42",
            epoch=EPOCH,
            location=COFFEE_TABLE,
        ),
    )


def reconnect_reuses_a_track_id() -> Sequence[Observation]:
    """The trap the `media_epoch_id` field exists to prevent.

    Both epochs contain `track-1`, because a tracker restarts its numbering
    after a dropout. They are different physical objects -- keys before the
    reconnect, a phone after it. Joining them on `track_id` alone merges two
    objects into one and produces a memory that is confidently wrong.
    """
    return (
        _observation(
            index=21, action="placed", at=T0, track_id="track-1", epoch=EPOCH, location=COFFEE_TABLE
        ),
        Observation(
            observation_id="obs_01JABDEMO0000000000000000022",
            idempotency_key=f"{DEVICE}/{SESSION}/track-1/placed/22",
            session_id=SESSION,
            device_id=DEVICE,
            media_epoch_id=EPOCH_AFTER_RECONNECT,
            object=ObjectRef(object_id=None, label="phone", track_id="track-1"),
            event=EventDetail(
                action="placed",
                source="vision_pipeline",
                occurred_at=T0 + dt.timedelta(minutes=5),
            ),
            location=Location(room="kitchen", surface="counter", relation="on"),
            confidence=ObservationConfidence(event=0.93, identity=0.95),
            evidence=(_evidence(22, T0 + dt.timedelta(minutes=5)),),
            provenance=_PROVENANCE,
        ),
    )


def weak_placement_after_a_strong_one() -> Sequence[Observation]:
    """A low-confidence sighting must not overwrite a confirmed placement."""
    return (
        _observation(
            index=31,
            action="placed",
            at=T0,
            track_id="track-42",
            epoch=EPOCH,
            location=COFFEE_TABLE,
        ),
        _observation(
            index=32,
            action="placed",
            at=T0 + dt.timedelta(minutes=1),
            track_id="track-42",
            epoch=EPOCH,
            location=Location(room="kitchen", surface="counter", relation="on"),
            event_confidence=0.21,
            identity_confidence=0.30,
        ),
    )


def placement_without_evidence() -> Sequence[Observation]:
    """A placement nothing can corroborate. It must not answer `confirmed`."""
    return (
        _observation(
            index=41,
            action="placed",
            at=T0,
            track_id="track-42",
            epoch=EPOCH,
            location=COFFEE_TABLE,
            with_evidence=False,
        ),
    )


SCENARIOS = {
    "keys_placed_then_picked_up": keys_placed_then_picked_up,
    "keys_placed_and_left": keys_placed_and_left,
    "reconnect_reuses_a_track_id": reconnect_reuses_a_track_id,
    "weak_placement_after_a_strong_one": weak_placement_after_a_strong_one,
    "placement_without_evidence": placement_without_evidence,
}


def scenario(name: str) -> Sequence[Observation]:
    """Look up a scenario by name, failing loudly on a typo."""
    try:
        return SCENARIOS[name]()
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {name!r}; known scenarios: {known}") from None


__all__ = [
    "COFFEE_TABLE",
    "DEVICE",
    "EPOCH",
    "EPOCH_AFTER_RECONNECT",
    "SCENARIOS",
    "SESSION",
    "T0",
    "keys_placed_and_left",
    "keys_placed_then_picked_up",
    "placement_without_evidence",
    "reconnect_reuses_a_track_id",
    "scenario",
    "weak_placement_after_a_strong_one",
]
