"""Building lifecycle envelopes.

Pure: no HTTP, no relay, no clock beyond what the caller supplies. The gateway
reports that a *track* or a *session* went away and nothing more, because it
has never run a detector and holds no object identity. Memory turns that into a
per-object transition -- see `docs/06-Data-Contract.md` § Lifecycle signals.

This is deliberately not an observation. An observation with a null object
would fail the promotion rules, and widening those rules to admit one would
weaken them for every real observation.
"""

from __future__ import annotations

import datetime as dt

from visual_memory_media_contract.protocol import (
    EpochEndReason,
    LifecycleDetail,
    LifecycleEnvelope,
    LifecycleProvenance,
    LifecycleScope,
    SessionEndReason,
)

from media_gateway import __version__
from media_gateway.domain.ids import lifecycle_idempotency_key, new_signal_id


def track_lost(
    *,
    session_id: str,
    device_id: str,
    media_epoch_id: str,
    reason: EpochEndReason,
    occurred_at: dt.datetime,
) -> LifecycleEnvelope:
    """One media epoch ended.

    Scoped by epoch rather than by object: the transition applies to every
    object whose in-transit state originated in that epoch, and only Memory
    knows which those are.
    """
    return LifecycleEnvelope(
        signal_id=new_signal_id(),
        idempotency_key=lifecycle_idempotency_key(
            device_id=device_id,
            session_id=session_id,
            scope_id=media_epoch_id,
            action="track_lost",
        ),
        session_id=session_id,
        device_id=device_id,
        signal=LifecycleDetail(action="track_lost", occurred_at=occurred_at, reason=reason),
        scope=LifecycleScope(media_epoch_id=media_epoch_id),
        provenance=LifecycleProvenance(component="media-gateway", version=__version__),
    )


def session_ended(
    *,
    session_id: str,
    device_id: str,
    reason: SessionEndReason,
    occurred_at: dt.datetime,
) -> LifecycleEnvelope:
    """The whole session ended.

    No epoch in the scope, so the signal reaches every in-transit object in the
    session rather than one track's worth.
    """
    return LifecycleEnvelope(
        signal_id=new_signal_id(),
        idempotency_key=lifecycle_idempotency_key(
            device_id=device_id,
            session_id=session_id,
            scope_id=session_id,
            action="session_ended",
        ),
        session_id=session_id,
        device_id=device_id,
        signal=LifecycleDetail(action="session_ended", occurred_at=occurred_at, reason=reason),
        scope=LifecycleScope(),
        provenance=LifecycleProvenance(component="media-gateway", version=__version__),
    )


__all__ = ["session_ended", "track_lost"]
