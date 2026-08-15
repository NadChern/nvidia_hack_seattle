"""Observable state for dashboards and release reports.

docs/03-Hackathon-Stack.md makes queue depth, dropped frames, verifier
outcomes, and latency release metrics rather than debugging aids, so this is a
first-class surface rather than a debug page.

The dimension histogram is the most useful field here in practice: it is what
tells you a camera is delivering a size the guard rejects, which otherwise
looks like a pipeline that is simply receiving nothing.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Request

from media_gateway import __version__
from media_gateway.deps import authorize_request
from media_gateway.domain.epoch import EpochRegistry, MediaEpoch
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.domain.session import Session, SessionRegistry
from media_gateway.relay.hub import RelayHub
from media_gateway.transport.memory_sink import MemorySink

router = APIRouter(tags=["status"])


def _session_view(session: Session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "device_id": session.device_id,
        "room": session.room,
        "created_at": session.created_at,
        "last_seen_at": session.last_seen_at,
        # Reported, never part of readiness: an idle gateway is healthy.
        "publisher_present": session.publisher_present,
    }


def _epoch_view(epoch: MediaEpoch) -> dict[str, Any]:
    view: dict[str, Any] = {
        "epoch_id": epoch.epoch_id,
        "session_id": epoch.session_id,
        "stream_kind": epoch.stream_kind,
        "track_sid": epoch.track_sid,
        "participant_identity": epoch.participant_identity,
        "started_at": epoch.started_at,
        "active": epoch.active,
        "received": epoch.received,
        "relayed": epoch.relayed,
        "next_sequence": epoch.next_sequence,
    }
    if epoch.guard is not None:
        view["guard"] = {
            "mode": epoch.guard.mode,
            "expected": (
                f"{epoch.guard.latched[0]}x{epoch.guard.latched[1]}"
                if epoch.guard.latched
                else None
            ),
            "admitted": epoch.guard.admitted,
            "rejected": epoch.guard.rejected,
            # How often `sustained` changed its mind. A handful at the start of
            # a stream is an encoder ramping to its negotiated resolution and
            # is expected; a number that keeps climbing means the publisher is
            # still changing size, and every change costs a motion sample.
            "relatched": epoch.guard.relatched,
            # Every size the track actually produced, whether admitted or not.
            "dimensions": dict(epoch.guard.histogram),
        }
    return view


def _relay_view(hub: RelayHub) -> dict[str, Any]:
    def describe(kind: str) -> list[dict[str, Any]]:
        return [
            {
                "encoding": subscriber.encoding,
                "queued": subscriber.depth,
                "sent": subscriber.sent,
                "dropped": subscriber.dropped,
                "closed": subscriber.close_reason,
            }
            for subscriber in hub.subscribers(kind)  # type: ignore[arg-type]
        ]

    return {
        "subscribers": len(hub),
        "video": describe("video"),
        "audio": describe("audio"),
    }


@router.get("/v1/status")
def status(request: Request) -> dict[str, Any]:
    """Report configuration, live sessions, epochs, relay state, and counters."""
    authorize_request(request)

    state = request.app.state
    settings = state.settings
    sessions: SessionRegistry = state.sessions
    epochs: EpochRegistry = state.epochs
    metrics: MetricsRegistry = state.metrics
    hub: RelayHub = state.hub

    sink: MemorySink | None = getattr(state, "memory_sink", None)

    reason = state.readiness.evaluate()
    started_at: dt.datetime = state.started_at

    return {
        "service": settings.service_name,
        "version": __version__,
        "environment": settings.environment,
        "media_source": settings.media_source,
        "ready": reason is None,
        "not_ready_reason": reason,
        "started_at": started_at,
        "uptime_s": round((dt.datetime.now(dt.UTC) - started_at).total_seconds(), 1),
        "config": {
            "livekit_url": settings.livekit_url,
            "sample_fps": settings.sample_fps,
            "video_encoding": settings.video_encoding,
            "dimension_guard_mode": settings.dimension_guard_mode,
            "expected_video_size": (
                f"{settings.expected_video_width}x{settings.expected_video_height}"
            ),
            "subscribe_video_quality": settings.subscribe_video_quality,
            "audio_chunk_ms": settings.audio_chunk_ms,
            "raw_buffer_seconds": settings.raw_buffer_seconds,
        },
        "sessions": [_session_view(session) for session in sessions.active()],
        "epochs": [_epoch_view(epoch) for epoch in epochs.active()],
        "relay": _relay_view(hub),
        # Whether lifecycle signals are reaching Memory. Silence here with a
        # configured sink is the tell that trusted state is going stale.
        "memory_sink": sink.snapshot() if sink is not None else {"enabled": False},
        "metrics": metrics.snapshot(),
    }
