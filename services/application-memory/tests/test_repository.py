"""The store, against a real database.

The reducer suite proves the rules. This proves they survive a round trip
through SQLite -- identity resolution, idempotency, recompute-on-write, and
epoch-scoped fan-out, none of which the pure reducer can check.
"""

from __future__ import annotations

import datetime as dt
import threading

import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker
from visual_memory_memory_contract import protocol
from visual_memory_memory_contract.fixtures import (
    DEVICE,
    EPOCH,
    SESSION,
    T0,
    keys_placed_and_left,
    keys_placed_then_picked_up,
    reconnect_reuses_a_track_id,
)

from application_memory.domain.reducer import PromotionPolicy
from application_memory.store import models, repository
from application_memory.store.models import ObjectIdentity, Observation

POLICY = PromotionPolicy()


def ingest(db: DbSession, observations: object) -> list[repository.IngestResult]:
    results = [
        repository.record_observation(db, observation, policy=POLICY)
        for observation in observations  # type: ignore[union-attr]
    ]
    db.commit()
    return results


def test_the_demo_scenario_survives_a_round_trip(db: DbSession) -> None:
    """The reducer's most important guarantee, but through the database.

    Timestamps come back from SQLite naive, so this is where a naive-versus-
    aware comparison would blow up -- a failure the pure reducer suite can
    never see.
    """
    results = ingest(db, keys_placed_then_picked_up())

    final = results[-1].state
    assert final is not None
    assert final.current_status == "in_transit"
    assert final.current_location is None
    assert final.last_confirmed_placement is not None
    assert final.last_confirmed_placement.surface == "coffee_table"


def test_identity_is_resolved_and_reused_within_an_epoch(db: DbSession) -> None:
    results = ingest(db, keys_placed_then_picked_up())

    object_ids = {result.object_id for result in results}
    assert len(object_ids) == 1
    assert next(iter(object_ids)) is not None


def test_a_reconnect_produces_two_objects_not_one(db: DbSession) -> None:
    """The same track_id in two epochs must not merge.

    This is the bug `media_epoch_id` exists to prevent: without epoch in the
    identity key, the keys would appear to move to the kitchen counter.
    """
    results = ingest(db, reconnect_reuses_a_track_id())

    assert results[0].object_id != results[1].object_id
    assert db.query(ObjectIdentity).count() == 2


def test_a_repeated_observation_is_not_applied_twice(db: DbSession) -> None:
    observations = list(keys_placed_and_left())
    ingest(db, observations)

    again = repository.record_observation(db, observations[0], policy=POLICY)
    db.commit()

    assert again.duplicate is True
    assert db.query(Observation).count() == len(observations)


def test_a_retry_with_a_fresh_id_is_still_deduplicated(db: DbSession) -> None:
    """A producer that retries and mints a new observation_id.

    The idempotency key is what identifies the logical event. Honouring only
    the id would let a timeout add an event that never happened.
    """
    first = keys_placed_and_left()[0]
    retry = first.model_copy(update={"observation_id": "obs_01JABDEMO_retry"})

    ingest(db, [first])
    result = repository.record_observation(db, retry, policy=POLICY)
    db.commit()

    assert result.duplicate is True
    assert result.observation_id == first.observation_id
    assert db.query(Observation).count() == 1


def test_the_same_id_with_different_content_is_refused(db: DbSession) -> None:
    """Two different events cannot share an identifier."""
    first = keys_placed_and_left()[0]
    impostor = first.model_copy(update={"idempotency_key": "a-different-key"})

    ingest(db, [first])

    with pytest.raises(repository.ConflictingObservation):
        repository.record_observation(db, impostor, policy=POLICY)


def test_an_unpromotable_observation_is_kept_as_history(db: DbSession) -> None:
    """Low confidence stores the record and leaves trusted state alone."""
    weak = keys_placed_and_left()[0]
    weak = weak.model_copy(
        update={"confidence": weak.confidence.model_copy(update={"event": 0.1, "identity": 0.1})}
    )

    result = ingest(db, [weak])[0]

    assert result.promoted is False
    assert result.state is None
    assert db.query(Observation).count() == 1


def test_a_late_observation_recomputes_rather_than_appends(db: DbSession) -> None:
    """The pickup arrives first, the placement second.

    Final state must be identical to in-order delivery: in transit, with the
    coffee table preserved as the last confirmed placement.
    """
    placed, _seen, picked_up = keys_placed_then_picked_up()

    ingest(db, [picked_up, placed])

    state = repository.state_of(db, repository.find_objects_by_label(db, "keys")[0])
    assert state is not None
    assert state.current_status == "in_transit"
    assert state.last_confirmed_placement is not None
    assert state.last_confirmed_placement.surface == "coffee_table"


def test_preresolved_object_id_still_receives_lifecycle_signals(db: DbSession) -> None:
    """A registered id still gets a tracker-scope mapping for lifecycle fan-out."""
    object_id = "object_registered_keys"
    observations = [
        item.model_copy(
            update={
                "object": item.object.model_copy(update={"object_id": object_id}),
            }
        )
        for item in keys_placed_then_picked_up()
    ]
    ingest(db, observations)

    states = repository.record_lifecycle(db, envelope_for(), policy=POLICY)
    db.commit()

    assert db.query(ObjectIdentity).filter(ObjectIdentity.object_id == object_id).count() == 1
    assert len(states) == 1
    assert states[0].current_status == "unknown"


def test_lifecycle_fans_out_to_every_object_in_the_epoch(db: DbSession) -> None:
    """The docs/06 sign-off, exercised.

    The gateway names an epoch, not an object. Memory turns that into a
    transition for each object whose identity began in that epoch.
    """
    ingest(db, keys_placed_then_picked_up())

    envelope = protocol.LifecycleEnvelope(
        signal_id="lc_01JABC",
        idempotency_key=f"{DEVICE}/{SESSION}/{EPOCH}/track_lost",
        session_id=SESSION,
        device_id=DEVICE,
        signal=protocol.LifecycleDetail(
            action="track_lost",
            occurred_at=T0 + dt.timedelta(minutes=4),
            reason="track_unsubscribed",
        ),
        scope=protocol.LifecycleScope(media_epoch_id=EPOCH),
        provenance=protocol.LifecycleProvenance(component="media-gateway", version="0.1.0"),
    )

    states = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert states
    assert all(state.current_status == "unknown" for state in states)
    assert all(state.last_confirmed_placement is not None for state in states)


def test_a_repeated_lifecycle_signal_is_ignored(db: DbSession) -> None:
    ingest(db, keys_placed_then_picked_up())
    envelope = protocol.LifecycleEnvelope(
        signal_id="lc_01JABC",
        idempotency_key=f"{DEVICE}/{SESSION}/{EPOCH}/track_lost",
        session_id=SESSION,
        device_id=DEVICE,
        signal=protocol.LifecycleDetail(
            action="track_lost",
            occurred_at=T0 + dt.timedelta(minutes=4),
            reason="track_unsubscribed",
        ),
        scope=protocol.LifecycleScope(media_epoch_id=EPOCH),
        provenance=protocol.LifecycleProvenance(component="media-gateway", version="0.1.0"),
    )

    repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()
    second = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert second == []


def test_registered_object_survives_session_delete_and_answers_in_a_new_session(
    db: DbSession,
) -> None:
    enrolled, _ = repository.create_enrolled_object(
        db, label="keys", idempotency_key="register/keys"
    )
    object_id = enrolled.object_id
    session_b = "sess_second"
    placed_a = keys_placed_and_left()[0]
    placed_b = keys_placed_and_left()[0].model_copy(
        update={
            "observation_id": "obs_session_b",
            "idempotency_key": "glasses-01/sess_second/track-2/placed/1",
            "session_id": session_b,
            "media_epoch_id": "TR_second",
            "object": placed_a.object.model_copy(
                update={"object_id": object_id, "track_id": "track-2"}
            ),
            "event": placed_a.event.model_copy(
                update={"occurred_at": T0 + dt.timedelta(minutes=10)}
            ),
        }
    )
    placed_a = placed_a.model_copy(
        update={"object": placed_a.object.model_copy(update={"object_id": object_id})}
    )
    ingest(db, [placed_a, placed_b])

    repository.delete_session(db, SESSION, policy=POLICY)
    db.commit()

    assert repository.find_objects_by_label(db, "keys", session_id=session_b) == [object_id]
    state = repository.state_of(db, object_id)
    assert state is not None
    assert state.current_status == "confirmed_at_location"
    assert db.get(models.EnrolledObjectRow, object_id) is not None


def test_deleting_a_session_removes_the_claim(db: DbSession) -> None:
    """State is derived, so there is no second place a memory survives."""
    ingest(db, keys_placed_and_left())
    assert repository.find_objects_by_label(db, "keys")

    counts = repository.delete_session(db, SESSION)
    db.commit()

    assert counts["observations"] == 2
    assert repository.find_objects_by_label(db, "keys") == []
    assert db.query(Observation).count() == 0


def test_retention_selects_only_stale_sessions(db: DbSession) -> None:
    ingest(db, keys_placed_and_left())

    future = repository.utcnow() + dt.timedelta(hours=25)
    past = repository.utcnow() - dt.timedelta(hours=25)

    assert repository.sessions_older_than(db, future) == [SESSION]
    assert repository.sessions_older_than(db, past) == []


def test_a_signal_that_applies_to_nothing_is_still_recorded(db: DbSession) -> None:
    """A receipt, so "never arrived" and "nothing to apply" stay distinguishable.

    Those need very different fixes, and on an integration day that distinction
    is most of the debugging.
    """
    envelope = protocol.LifecycleEnvelope(
        signal_id="lc_01JORPHAN",
        idempotency_key=f"{DEVICE}/{SESSION}/nothing/session_ended",
        session_id=SESSION,
        device_id=DEVICE,
        signal=protocol.LifecycleDetail(
            action="session_ended",
            occurred_at=T0 + dt.timedelta(minutes=1),
            reason="session_deleted",
        ),
        scope=protocol.LifecycleScope(),
        provenance=protocol.LifecycleProvenance(component="media-gateway", version="0.1.0"),
    )

    states = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert states == []
    stored = db.query(models.LifecycleSignal).all()
    assert len(stored) == 1
    # No object, so the reducer never sees it.
    assert stored[0].object_id is None


def envelope_for(
    action: str = "track_lost", epoch: str | None = EPOCH
) -> protocol.LifecycleEnvelope:
    return protocol.LifecycleEnvelope(
        signal_id=f"lc_{action}",
        idempotency_key=f"{DEVICE}/{SESSION}/{epoch or SESSION}/{action}",
        session_id=SESSION,
        device_id=DEVICE,
        signal=protocol.LifecycleDetail(
            action=action,  # type: ignore[arg-type]
            occurred_at=T0 + dt.timedelta(minutes=4),
            reason="track_unsubscribed" if action == "track_lost" else "session_deleted",
        ),
        scope=protocol.LifecycleScope(media_epoch_id=epoch),
        provenance=protocol.LifecycleProvenance(component="media-gateway", version="0.1.0"),
    )


def test_a_receipt_does_not_swallow_the_signal_when_objects_appear(db: DbSession) -> None:
    """A signal that lands before Vision's observations must still apply later.

    The gateway posts on disconnect; Vision may be behind. If the receipt
    written for "nothing to apply" also counted as "already applied", the
    resend would be ignored and the object would stay in_transit forever --
    a stale claim, which is the one thing this system must never produce.
    """
    envelope = envelope_for()

    # Arrives first, reaching nothing.
    assert repository.record_lifecycle(db, envelope, policy=POLICY) == []
    ingest(db, keys_placed_then_picked_up())

    # The gateway re-sends; now there is something to apply.
    applied = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert len(applied) == 1
    assert applied[0].current_status == "unknown"
    # The receipt was replaced, not left to collide with the applied row.
    receipts = (
        db.query(models.LifecycleSignal).filter(models.LifecycleSignal.object_id.is_(None)).count()
    )
    assert receipts == 0


def test_an_applied_signal_still_blocks_a_repeat(db: DbSession) -> None:
    """Loosening the dedup must not reintroduce double application."""
    envelope = envelope_for()
    ingest(db, keys_placed_then_picked_up())

    first = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()
    second = repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert len(first) == 1
    assert second == []


def test_a_repeated_receipt_is_not_stored_twice(db: DbSession) -> None:
    """Two deliveries that both reach nothing would collide on the key."""
    envelope = envelope_for()

    repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()
    repository.record_lifecycle(db, envelope, policy=POLICY)
    db.commit()

    assert db.query(models.LifecycleSignal).count() == 1


def test_concurrent_writers_do_not_collide_creating_the_session(
    sessions: sessionmaker[DbSession],
) -> None:
    """FastAPI runs sync endpoints in a threadpool, so this is the real shape.

    Get-then-insert made 11 of 12 concurrent writers fail on the session
    primary key before the savepoint was added.
    """
    base = keys_placed_and_left()[0]
    failures: list[str] = []

    def write(n: int) -> None:
        try:
            with sessions() as db:
                repository.record_observation(
                    db,
                    base.model_copy(
                        update={"observation_id": f"obs_{n}", "idempotency_key": f"key_{n}"}
                    ),
                    policy=POLICY,
                )
                db.commit()
        except Exception as exc:  # pragma: no cover - the assertion reports it
            failures.append(type(exc).__name__)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    with sessions() as db:
        assert db.query(Observation).count() == 16
