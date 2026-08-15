"""Reads and writes, and the recompute that keeps derived state honest.

The only place that knows both the reducer and the database. Endpoints call
this; the reducer never does.
"""

from __future__ import annotations

import datetime as dt
import sys
from array import array
from dataclasses import dataclass

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession
from visual_memory_memory_contract import protocol
from visual_memory_memory_contract.ids import new_object_id

from application_memory.domain.reducer import (
    LifecycleEvent,
    PromotionPolicy,
    ReduceResult,
    TimelineEntry,
    reduce,
)
from application_memory.store import models


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _aware(value: dt.datetime) -> dt.datetime:
    """Re-attach UTC to a datetime SQLite handed back naive.

    Without this a stored timestamp and a fresh one cannot be compared, and the
    reducer raises `can't compare offset-naive and offset-aware datetimes` --
    at write time, on a path that worked in unit tests because those never
    round-trip through the database.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What happened to one submitted observation."""

    observation_id: str
    object_id: str | None
    state: protocol.ObjectState | None
    promoted: bool
    #: True when this exact request had already been applied. The caller
    #: returns the original outcome rather than reporting a second write.
    duplicate: bool = False


class ConflictingObservation(Exception):
    """Same observation_id, different content."""


@dataclass(frozen=True, slots=True)
class DeletedObject:
    object_id: str
    crop_relative_paths: tuple[str, ...]
    registry_version: int


def _pack_float32(values: tuple[float, ...]) -> bytes:
    packed = array("f", values)
    if packed.itemsize != 4:  # pragma: no cover - CPython's supported platforms
        raise RuntimeError("this platform does not provide 32-bit C floats")
    if sys.byteorder != "little":  # pragma: no cover - supported deployment is little-endian
        packed.byteswap()
    return packed.tobytes()


def _unpack_float32(value: bytes, dim: int) -> tuple[float, ...]:
    unpacked = array("f")
    unpacked.frombytes(value)
    if sys.byteorder != "little":  # pragma: no cover - supported deployment is little-endian
        unpacked.byteswap()
    if len(unpacked) != dim:
        raise ValueError(f"stored vector has {len(unpacked)} values, expected {dim}")
    return tuple(float(item) for item in unpacked)


def _next_registry_version(db: DbSession) -> int:
    state = db.get(models.RegistryStateRow, 1)
    if state is None:
        try:
            with db.begin_nested():
                db.add(models.RegistryStateRow(id=1, version=0))
        except IntegrityError:
            # Another request initialized the singleton first.
            pass
    version = db.scalar(
        update(models.RegistryStateRow)
        .where(models.RegistryStateRow.id == 1)
        .values(version=models.RegistryStateRow.version + 1)
        .returning(models.RegistryStateRow.version)
    )
    if version is None:  # pragma: no cover - only if the singleton vanished mid-write
        raise RuntimeError("registry version row disappeared")
    return version


def registry_version(db: DbSession) -> int:
    state = db.get(models.RegistryStateRow, 1)
    return state.version if state is not None else 0


def create_enrolled_object(
    db: DbSession, *, label: str, idempotency_key: str
) -> tuple[protocol.EnrolledObject, bool]:
    existing = db.scalars(
        select(models.EnrolledObjectRow).where(
            models.EnrolledObjectRow.idempotency_key == idempotency_key
        )
    ).first()
    if existing is not None:
        if existing.label != label:
            raise ConflictingObservation(
                "object idempotency key already belongs to a different label"
            )
        return _enrolled_object(existing), False

    now = utcnow()
    version = _next_registry_version(db)
    row = models.EnrolledObjectRow(
        object_id=new_object_id(),
        label=label,
        idempotency_key=idempotency_key,
        created_at=now,
        updated_at=now,
        registry_version=version,
    )
    try:
        with db.begin_nested():
            db.add(row)
        return _enrolled_object(row), True
    except IntegrityError:
        winner = db.scalars(
            select(models.EnrolledObjectRow).where(
                models.EnrolledObjectRow.idempotency_key == idempotency_key
            )
        ).first()
        if winner is None:  # pragma: no cover - only if the row vanished again
            raise
        if winner.label != label:
            raise ConflictingObservation(
                "object idempotency key already belongs to a different label"
            ) from None
        return _enrolled_object(winner), False


def enrolled_object(db: DbSession, object_id: str) -> protocol.EnrolledObject | None:
    row = db.get(models.EnrolledObjectRow, object_id)
    return _enrolled_object(row) if row is not None else None


def find_object_view_by_content(
    db: DbSession, *, object_id: str, view_index: int, crop_sha256: str
) -> protocol.ObjectView | None:
    row = db.scalars(
        select(models.ObjectViewRow).where(
            models.ObjectViewRow.object_id == object_id,
            models.ObjectViewRow.view_index == view_index,
            models.ObjectViewRow.crop_sha256 == crop_sha256,
        )
    ).first()
    return _object_view(row) if row is not None else None


def put_object_view(
    db: DbSession,
    *,
    object_id: str,
    view_id: str,
    upload: protocol.ObjectViewUpload,
    crop_relative_path: str,
    max_views: int,
    max_dim: int,
) -> protocol.ObjectView:
    owner = db.get(models.EnrolledObjectRow, object_id)
    if owner is None:
        raise LookupError(object_id)
    duplicate = find_object_view_by_content(
        db,
        object_id=object_id,
        view_index=upload.view_index,
        crop_sha256=upload.crop_sha256,
    )
    if duplicate is not None:
        return duplicate
    if upload.dim > max_dim:
        raise ValueError(f"embedding dim {upload.dim} exceeds configured maximum {max_dim}")
    count = int(
        db.scalar(
            select(func.count())
            .select_from(models.ObjectViewRow)
            .where(models.ObjectViewRow.object_id == object_id)
        )
        or 0
    )
    if count >= max_views:
        raise OverflowError(f"object already has the maximum {max_views} reference views")

    now = utcnow()
    version = _next_registry_version(db)
    row = models.ObjectViewRow(
        view_id=view_id,
        object_id=object_id,
        view_index=upload.view_index,
        quality=upload.quality.model_dump(mode="json"),
        embedder_id=upload.embedder_id,
        pooling=upload.pooling,
        dim=upload.dim,
        summary=_pack_float32(upload.summary),
        pooled_spatial=_pack_float32(upload.pooled_spatial),
        crop_sha256=upload.crop_sha256.lower(),
        crop_media_type=upload.crop_media_type,
        crop_relative_path=crop_relative_path,
        created_at=now,
        registry_version=version,
    )
    db.add(row)
    owner.updated_at = now
    owner.registry_version = version
    db.flush()
    return _object_view(row)


def object_view_row(db: DbSession, object_id: str, view_id: str) -> models.ObjectViewRow | None:
    row = db.get(models.ObjectViewRow, view_id)
    return row if row is not None and row.object_id == object_id else None


def list_gallery(db: DbSession, *, since_version: int | None = None) -> protocol.ObjectGallery:
    version = registry_version(db)
    if since_version is not None and since_version >= version:
        return protocol.ObjectGallery(registry_version=version, unchanged=True)
    object_rows = db.scalars(
        select(models.EnrolledObjectRow).order_by(models.EnrolledObjectRow.object_id)
    ).all()
    view_rows = db.scalars(
        select(models.ObjectViewRow).order_by(
            models.ObjectViewRow.object_id, models.ObjectViewRow.view_index
        )
    ).all()
    return protocol.ObjectGallery(
        registry_version=version,
        objects=tuple(_enrolled_object(row) for row in object_rows),
        views=tuple(_object_view(row) for row in view_rows),
    )


def delete_enrolled_object(db: DbSession, object_id: str) -> DeletedObject | None:
    row = db.get(models.EnrolledObjectRow, object_id)
    if row is None:
        return None
    views = db.scalars(
        select(models.ObjectViewRow).where(models.ObjectViewRow.object_id == object_id)
    ).all()
    paths = tuple(view.crop_relative_path for view in views)
    # Explicit deletion keeps behavior identical on databases where FK cascade
    # configuration differs; no session row is involved.
    db.execute(delete(models.ObjectViewRow).where(models.ObjectViewRow.object_id == object_id))
    db.delete(row)
    version = _next_registry_version(db)
    db.add(
        models.AuditRow(
            occurred_at=utcnow(),
            action="registered_object_deleted",
            session_id=None,
            detail=f"object_id={object_id}, views={len(paths)}",
        )
    )
    return DeletedObject(object_id, paths, version)


def _enrolled_object(row: models.EnrolledObjectRow) -> protocol.EnrolledObject:
    return protocol.EnrolledObject(
        object_id=row.object_id,
        label=row.label,
        idempotency_key=row.idempotency_key,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        registry_version=row.registry_version,
    )


def _object_view(row: models.ObjectViewRow) -> protocol.ObjectView:
    quality = protocol.ObjectViewQuality.model_validate(row.quality)
    return protocol.ObjectView(
        view_id=row.view_id,
        object_id=row.object_id,
        view_index=row.view_index,
        quality=quality,
        embedder_id=row.embedder_id,
        pooling=row.pooling,
        dim=row.dim,
        summary=_unpack_float32(row.summary, row.dim),
        pooled_spatial=_unpack_float32(row.pooled_spatial, row.dim),
        crop_sha256=row.crop_sha256,
        crop_media_type=row.crop_media_type,
        crop_reference=f"/v1/objects/{row.object_id}/views/{row.view_id}/crop",
        created_at=_aware(row.created_at),
        registry_version=row.registry_version,
    )


def ensure_session(db: DbSession, *, session_id: str, device_id: str) -> models.Session:
    """Create the session row if it is missing, tolerating a concurrent creator.

    Get-then-insert is a race: FastAPI runs sync endpoints in a threadpool, so
    several observations for one session arrive at once, all miss, and all
    insert. Measured with 12 concurrent writers, 11 failed on the primary key.

    The insert therefore happens inside a savepoint. Losing the race rolls back
    only that savepoint -- the caller's transaction survives -- and the winner's
    row is read instead.
    """
    now = utcnow()
    row = db.get(models.Session, session_id)
    if row is not None:
        row.last_seen_at = now
        return row

    try:
        with db.begin_nested():
            row = models.Session(
                session_id=session_id, device_id=device_id, created_at=now, last_seen_at=now
            )
            db.add(row)
        return row
    except IntegrityError:
        existing = db.get(models.Session, session_id)
        if existing is None:  # pragma: no cover - only if the row vanished again
            raise
        existing.last_seen_at = now
        return existing


def resolve_identity(db: DbSession, observation: protocol.Observation) -> str:
    """Map a tracker identity to a stable object id and always retain its scope.

    A producer-resolved personal id wins, but it still needs an
    `ObjectIdentity` row: lifecycle fan-out reads that table, not observations.
    Returning early used to make registered in-transit objects immune to
    `track_lost` and `session_ended`.
    """
    scope = select(models.ObjectIdentity).where(
        models.ObjectIdentity.session_id == observation.session_id,
        models.ObjectIdentity.media_epoch_id == observation.media_epoch_id,
        models.ObjectIdentity.track_id == observation.object.track_id,
        models.ObjectIdentity.label == observation.object.label,
    )
    existing = db.scalars(scope).first()
    resolved_id = observation.object.object_id
    if existing is not None:
        if resolved_id is not None and existing.object_id != resolved_id:
            existing.object_id = resolved_id
        return resolved_id or existing.object_id

    # Same get-then-insert race as `ensure_session`, and worse if it slipped
    # through: two concurrent observations for one tracker identity would mint
    # two object ids, splitting one object's timeline in half so neither half
    # holds enough history to answer with. The unique constraint prevents that;
    # the savepoint turns losing the race into reading the winner's row rather
    # than a 500.
    object_id = resolved_id or new_object_id()
    try:
        with db.begin_nested():
            db.add(
                models.ObjectIdentity(
                    session_id=observation.session_id,
                    media_epoch_id=observation.media_epoch_id,
                    track_id=observation.object.track_id,
                    label=observation.object.label,
                    object_id=object_id,
                )
            )
        return object_id
    except IntegrityError:
        winner = db.scalars(scope).first()
        if winner is None:  # pragma: no cover - only if the row vanished again
            raise
        if resolved_id is not None and winner.object_id != resolved_id:
            winner.object_id = resolved_id
            return resolved_id
        return winner.object_id


def find_objects_by_label(db: DbSession, label: str, *, session_id: str | None = None) -> list[str]:
    """Every object matching a label, so the caller can detect ambiguity."""
    query = select(models.ObjectStateRow.object_id).where(models.ObjectStateRow.label == label)
    if session_id is not None:
        query = query.where(models.ObjectStateRow.session_id == session_id)
    return list(db.scalars(query))


def timeline_for(db: DbSession, object_id: str) -> list[TimelineEntry]:
    """Every stored entry for one object, ready to replay.

    Returns observations and lifecycle signals together, because the reducer
    orders them against each other -- a track_lost between two sightings means
    something different from one after both.
    """
    entries: list[TimelineEntry] = []

    rows = db.scalars(
        select(models.Observation).where(models.Observation.object_id == object_id)
    ).all()
    for row in rows:
        entries.append(protocol.Observation.model_validate(row.payload))

    signals = db.scalars(
        select(models.LifecycleSignal).where(models.LifecycleSignal.object_id == object_id)
    ).all()
    for signal in signals:
        entries.append(
            LifecycleEvent(
                signal_id=signal.signal_id,
                action=signal.action,  # type: ignore[arg-type]
                reason=signal.reason,  # type: ignore[arg-type]
                occurred_at=_aware(signal.occurred_at),
                media_epoch_id=signal.media_epoch_id,
            )
        )
    return entries


def recompute(
    db: DbSession, object_id: str, *, policy: PromotionPolicy, label: str, session_id: str
) -> ReduceResult:
    """Replay an object's whole timeline and store the derived state.

    Called after every write. Replaying rather than patching is what makes a
    late observation, a duplicate, and a restart produce the same answer.
    """
    result = reduce(object_id, timeline_for(db, object_id), policy=policy)

    row = db.get(models.ObjectStateRow, object_id)
    if result.state is None:
        # Nothing clears the bar any more. Remove the cached claim rather than
        # leaving a stale one that no observation supports.
        if row is not None:
            db.delete(row)
        return result

    serialized = result.state.model_dump(mode="json")
    if row is None:
        db.add(
            models.ObjectStateRow(
                object_id=object_id,
                session_id=session_id,
                label=label,
                current_status=result.state.current_status,
                updated_at=result.state.updated_at,
                state=serialized,
            )
        )
    else:
        row.current_status = result.state.current_status
        row.updated_at = result.state.updated_at
        row.state = serialized
        row.label = label
        # Label-scoped queries may be session-filtered. A stable personal
        # object observed in a new session must move with its latest state.
        row.session_id = session_id
    return result


def record_observation(
    db: DbSession, observation: protocol.Observation, *, policy: PromotionPolicy
) -> IngestResult:
    """Store an observation and recompute the object it concerns."""
    existing = db.get(models.Observation, observation.observation_id)
    if existing is not None:
        if existing.idempotency_key != observation.idempotency_key:
            raise ConflictingObservation(
                f"observation {observation.observation_id} already exists with different content"
            )
        return IngestResult(
            observation_id=existing.observation_id,
            object_id=existing.object_id,
            state=state_of(db, existing.object_id),
            promoted=existing.promoted,
            duplicate=True,
        )

    by_key = db.scalars(
        select(models.Observation).where(
            models.Observation.idempotency_key == observation.idempotency_key
        )
    ).first()
    if by_key is not None:
        # Same logical event, different observation id -- a retry that minted a
        # fresh id. Honour the key: applying it again would add an event that
        # never happened.
        return IngestResult(
            observation_id=by_key.observation_id,
            object_id=by_key.object_id,
            state=state_of(db, by_key.object_id),
            promoted=by_key.promoted,
            duplicate=True,
        )

    ensure_session(db, session_id=observation.session_id, device_id=observation.device_id)
    object_id = resolve_identity(db, observation)
    resolved = observation.model_copy(
        update={"object": observation.object.model_copy(update={"object_id": object_id})}
    )

    db.add(
        models.Observation(
            observation_id=resolved.observation_id,
            idempotency_key=resolved.idempotency_key,
            session_id=resolved.session_id,
            device_id=resolved.device_id,
            media_epoch_id=resolved.media_epoch_id,
            object_id=object_id,
            label=resolved.object.label,
            track_id=resolved.object.track_id,
            action=resolved.event.action,
            occurred_at=resolved.event.occurred_at,
            ingested_at=utcnow(),
            event_confidence=resolved.confidence.event,
            identity_confidence=resolved.confidence.identity,
            promoted=False,
            payload=resolved.model_dump(mode="json"),
        )
    )
    db.flush()

    result = recompute(
        db,
        object_id,
        policy=policy,
        label=resolved.object.label,
        session_id=resolved.session_id,
    )
    promoted = resolved.observation_id in result.promoted_ids
    stored = db.get(models.Observation, resolved.observation_id)
    if stored is not None:
        stored.promoted = promoted

    return IngestResult(
        observation_id=resolved.observation_id,
        object_id=object_id,
        state=result.state,
        promoted=promoted,
    )


def state_of(db: DbSession, object_id: str | None) -> protocol.ObjectState | None:
    if object_id is None:
        return None
    row = db.get(models.ObjectStateRow, object_id)
    return protocol.ObjectState.model_validate(row.state) if row is not None else None


def objects_in_scope(
    db: DbSession, *, session_id: str, media_epoch_id: str | None
) -> list[tuple[str, str]]:
    """Objects a lifecycle signal applies to, as `(object_id, label)`.

    An epoch-scoped signal reaches every object whose identity was established
    in that epoch; a session-scoped one reaches the whole session. This is the
    fan-out the gateway cannot do, because it has never seen an object.
    """
    query = select(models.ObjectIdentity.object_id, models.ObjectIdentity.label).where(
        models.ObjectIdentity.session_id == session_id
    )
    if media_epoch_id is not None:
        query = query.where(models.ObjectIdentity.media_epoch_id == media_epoch_id)
    return [(object_id, label) for object_id, label in db.execute(query).all()]


def _receipt_exists(db: DbSession, idempotency_key: str) -> bool:
    """Whether a signal with this key was already recorded as applying to nothing."""
    return (
        db.scalars(
            select(models.LifecycleSignal).where(
                models.LifecycleSignal.idempotency_key == idempotency_key,
                models.LifecycleSignal.object_id.is_(None),
            )
        ).first()
        is not None
    )


def record_lifecycle(
    db: DbSession, envelope: protocol.LifecycleEnvelope, *, policy: PromotionPolicy
) -> list[protocol.ObjectState]:
    """Apply a lifecycle signal to every object it scopes, returning new states."""
    # Only an *applied* signal blocks a repeat. A receipt -- stored when the
    # signal reached no objects -- must not, or a signal that arrives before
    # Vision's observations would be permanently swallowed: the resend that
    # should apply it finds the receipt and returns early, and the object stays
    # `in_transit` forever instead of becoming `unknown`. That is exactly the
    # stale claim this system exists to prevent.
    applied = db.scalars(
        select(models.LifecycleSignal).where(
            models.LifecycleSignal.idempotency_key == envelope.idempotency_key,
            models.LifecycleSignal.object_id.is_not(None),
        )
    ).first()
    if applied is not None:
        return []

    ensure_session(db, session_id=envelope.session_id, device_id=envelope.device_id)

    if envelope.scope.object_id is not None:
        row = db.get(models.ObjectStateRow, envelope.scope.object_id)
        targets = [(envelope.scope.object_id, row.label if row else "")]
    else:
        targets = objects_in_scope(
            db, session_id=envelope.session_id, media_epoch_id=envelope.scope.media_epoch_id
        )

    payload = envelope.model_dump(mode="json")
    now = utcnow()
    states: list[protocol.ObjectState] = []

    if not targets:
        # The signal applies to nothing -- no objects were tracked in that
        # epoch, or the session never produced any. Record it anyway, with no
        # object, purely as a receipt.
        #
        # Storing nothing would leave an operator unable to tell "the gateway
        # never reached us" from "it did and there was nothing to apply", and
        # those need very different fixes. `timeline_for` filters by object_id,
        # so this row never reaches the reducer.
        if _receipt_exists(db, envelope.idempotency_key):
            # Already noted. A second receipt would collide on the key.
            return []
        db.add(
            models.LifecycleSignal(
                signal_id=envelope.signal_id,
                idempotency_key=envelope.idempotency_key,
                session_id=envelope.session_id,
                device_id=envelope.device_id,
                media_epoch_id=envelope.scope.media_epoch_id,
                object_id=None,
                action=envelope.signal.action,
                reason=envelope.signal.reason,
                occurred_at=envelope.signal.occurred_at,
                received_at=now,
                payload=payload,
            )
        )
        db.flush()
        return []

    # A receipt may already hold this key from an earlier delivery that found
    # no objects. Clear it: the signal is being applied for real now, and the
    # per-target rows reuse the same key.
    db.execute(
        delete(models.LifecycleSignal).where(
            models.LifecycleSignal.idempotency_key == envelope.idempotency_key,
            models.LifecycleSignal.object_id.is_(None),
        )
    )

    for index, (object_id, _label) in enumerate(targets):
        db.add(
            models.LifecycleSignal(
                # One stored row per affected object, so replay reaches each
                # object's timeline. The envelope's own key stays unique.
                signal_id=f"{envelope.signal_id}#{index}" if index else envelope.signal_id,
                idempotency_key=f"{envelope.idempotency_key}#{index}"
                if index
                else envelope.idempotency_key,
                session_id=envelope.session_id,
                device_id=envelope.device_id,
                media_epoch_id=envelope.scope.media_epoch_id,
                object_id=object_id,
                action=envelope.signal.action,
                reason=envelope.signal.reason,
                occurred_at=envelope.signal.occurred_at,
                received_at=now,
                payload=payload,
            )
        )
    db.flush()

    for object_id, label in targets:
        result = recompute(
            db, object_id, policy=policy, label=label, session_id=envelope.session_id
        )
        if result.state is not None:
            states.append(result.state)
    return states


def delete_session(
    db: DbSession,
    session_id: str,
    *,
    policy: PromotionPolicy | None = None,
) -> dict[str, int]:
    """Remove one session without destroying a stable object's other sessions.

    `ObjectStateRow.session_id` identifies the latest contributing session; it
    is not ownership. Deleting rows by that cache field used to erase a
    registered object's state even when another session still had live
    observations. Delete source events first, then recompute every affected
    stable object from what remains.
    """
    resolved_policy = policy or PromotionPolicy()
    affected_ids = {
        object_id
        for object_id in db.scalars(
            select(models.Observation.object_id).where(
                models.Observation.session_id == session_id,
                models.Observation.object_id.is_not(None),
            )
        ).all()
        if object_id is not None
    }
    affected_ids.update(
        db.scalars(
            select(models.ObjectIdentity.object_id).where(
                models.ObjectIdentity.session_id == session_id
            )
        ).all()
    )

    counts = {
        "observations": db.query(models.Observation)
        .filter(models.Observation.session_id == session_id)
        .count(),
        "lifecycle_signals": db.query(models.LifecycleSignal)
        .filter(models.LifecycleSignal.session_id == session_id)
        .count(),
        "object_states": db.query(models.ObjectStateRow)
        .filter(models.ObjectStateRow.session_id == session_id)
        .count(),
        "evidence": db.query(models.EvidenceRow)
        .filter(models.EvidenceRow.session_id == session_id)
        .count(),
    }

    for table in (
        models.Observation,
        models.LifecycleSignal,
        models.EvidenceRow,
        models.ObjectIdentity,
    ):
        db.execute(delete(table).where(table.session_id == session_id))  # type: ignore[arg-type]
    db.execute(delete(models.Session).where(models.Session.session_id == session_id))
    db.flush()

    for object_id in affected_ids:
        latest = db.scalars(
            select(models.Observation)
            .where(models.Observation.object_id == object_id)
            .order_by(models.Observation.occurred_at.desc())
        ).first()
        if latest is None:
            row = db.get(models.ObjectStateRow, object_id)
            if row is not None:
                db.delete(row)
            continue
        recompute(
            db,
            object_id,
            policy=resolved_policy,
            label=latest.label,
            session_id=latest.session_id,
        )

    db.add(
        models.AuditRow(
            occurred_at=utcnow(),
            action="session_deleted",
            session_id=session_id,
            detail=", ".join(f"{name}={count}" for name, count in sorted(counts.items())),
        )
    )
    return counts


def sessions_older_than(db: DbSession, cutoff: dt.datetime) -> list[str]:
    """Sessions whose retention window has passed."""
    rows = db.scalars(
        select(models.Session.session_id).where(models.Session.last_seen_at < cutoff)
    ).all()
    return list(rows)


__all__ = [
    "ConflictingObservation",
    "DeletedObject",
    "IngestResult",
    "create_enrolled_object",
    "delete_enrolled_object",
    "delete_session",
    "enrolled_object",
    "ensure_session",
    "find_object_view_by_content",
    "find_objects_by_label",
    "list_gallery",
    "object_view_row",
    "objects_in_scope",
    "put_object_view",
    "recompute",
    "record_lifecycle",
    "record_observation",
    "registry_version",
    "resolve_identity",
    "sessions_older_than",
    "state_of",
    "timeline_for",
    "utcnow",
]
