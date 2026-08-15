"""Relational schema.

**Observations are immutable; state is derived.** That is the central choice
here and it comes straight from docs/06's requirement that late events
recompute a timeline without reordering history. `object_states` is a cache
recomputed on every write, never a source of truth -- which is also why
deleting a session's observations genuinely deletes the memory, with no second
place for a claim to survive.

Timestamps are stored as timezone-aware UTC. SQLite has no native datetime, so
`DateTime(timezone=True)` round-trips through a string and is read back naive;
`repository.py` re-attaches UTC on the way out rather than letting a naive
datetime reach the reducer, where comparing it to an aware one would raise.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class Session(Base):
    """One continuous period of a device being connected.

    Minted by the Media Gateway, which is the only component present when a
    session starts. Memory owns persistence and deletion.
    """

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))


class ObjectIdentity(Base):
    """Maps a tracker identity to a stable object.

    The key includes `media_epoch_id` on purpose. A tracker restarts its
    numbering after a reconnect, so `track-42` before a dropout and `track-42`
    after it are different physical objects. Making the epoch part of the
    primary key means the database cannot express the merge, rather than
    relying on every caller to remember not to.
    """

    __tablename__ = "object_identities"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "media_epoch_id", "track_id", "label", name="uq_identity_scope"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.session_id"))
    #: Null for observations with no media provenance -- a manual correction,
    #: a backfill, or a fixture.
    media_epoch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    track_id: Mapped[str] = mapped_column(String(128))
    label: Mapped[str] = mapped_column(String(128), index=True)
    object_id: Mapped[str] = mapped_column(String(64), index=True)


class Observation(Base):
    """An immutable record of something a perception service reported.

    Stored whether or not it promotes. docs/06 requires low-confidence
    observations to be retained for history without touching trusted state, so
    `promoted` records the reducer's decision rather than filtering the row out.
    """

    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_object_time", "object_id", "occurred_at"),
        Index("ix_observations_session", "session_id"),
    )

    observation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: A repeat returns the original result instead of re-applying. Unique so
    #: the database refuses the second write rather than trusting the caller.
    idempotency_key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    device_id: Mapped[str] = mapped_column(String(128))
    media_epoch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Null while identity is unresolved. Such an observation is history only.
    object_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(128))
    track_id: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    event_confidence: Mapped[float] = mapped_column(Float)
    identity_confidence: Mapped[float] = mapped_column(Float)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    #: The full validated envelope, so the reducer replays exactly what was
    #: accepted rather than a lossy projection of it.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class LifecycleSignal(Base):
    """A track or session ending, as reported by the Media Gateway."""

    __tablename__ = "lifecycle_signals"

    signal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    device_id: Mapped[str] = mapped_column(String(128))
    media_epoch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    object_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(64))
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ObjectStateRow(Base):
    """Derived trusted state.

    A cache of what the reducer computed, kept so a query is one read rather
    than a replay. It is never authoritative: delete the observations and this
    row is recomputed to nothing.
    """

    __tablename__ = "object_states"

    object_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(128), index=True)
    current_status: Mapped[str] = mapped_column(String(32), index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    #: The serialized ObjectState, so the query layer returns exactly the
    #: documented shape without rebuilding it field by field.
    state: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvidenceRow(Base):
    """A stored frame, addressed by id and verified by digest."""

    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    media_type: Mapped[str] = mapped_column(String(64), default="image/jpeg")
    #: Relative to the configured evidence directory, so the store can move
    #: without rewriting rows.
    relative_path: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))


class AuditRow(Base):
    """Access, export, and deletion events.

    docs/07 requires these to be recorded *without* logging sensitive media, so
    this table holds counts and identifiers and never bytes, transcripts, or
    file contents.
    """

    __tablename__ = "audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")


__all__ = [
    "AuditRow",
    "Base",
    "EvidenceRow",
    "LifecycleSignal",
    "ObjectIdentity",
    "ObjectStateRow",
    "Observation",
    "Session",
]
