"""Session lifecycle and token issuance.

The gateway mints `session_id` because it is the only component that observes a
session start: the device asks it for a token. The Memory Service remains the
authority for session persistence and deletion, and adopting its id later is a
configuration change rather than a code change.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

from media_gateway.deps import authorize_device_request, authorize_request
from media_gateway.domain.ids import new_session_id
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.domain.ratelimit import FixedWindowLimiter
from media_gateway.domain.session import Session, SessionRegistry
from media_gateway.errors import CapacityError, UnavailableError
from media_gateway.transport.memory_sink import register_session
from media_gateway.transport.tokens import (
    MintedToken,
    mint_access_token,
    publisher_identity,
    viewer_identity,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"], prefix="/v1/sessions")

UNKNOWN_CLIENT = "unknown"


class CreateSessionRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)


class SessionTokenResponse(BaseModel):
    """Everything one room participant needs to join, and nothing more."""

    session_id: str
    device_id: str
    room: str
    livekit_url: str
    identity: str
    token: str
    expires_at: dt.datetime


class SessionSummary(BaseModel):
    session_id: str
    device_id: str
    room: str
    created_at: dt.datetime
    last_seen_at: dt.datetime
    publisher_present: bool


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


def _summarize(session: Session) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        device_id=session.device_id,
        room=session.room,
        created_at=session.created_at,
        last_seen_at=session.last_seen_at,
        publisher_present=session.publisher_present,
    )


def _client_of(request: Request) -> str:
    return request.client.host if request.client else UNKNOWN_CLIENT


def _token_response(
    *, session: Session, minted: MintedToken, livekit_url: str
) -> SessionTokenResponse:
    return SessionTokenResponse(
        session_id=session.session_id,
        device_id=session.device_id,
        room=session.room,
        livekit_url=livekit_url,
        identity=minted.identity,
        token=minted.token,
        expires_at=minted.expires_at,
    )


@router.post("", response_model=SessionTokenResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: Request, body: CreateSessionRequest) -> SessionTokenResponse:
    """Start a session and issue a short-lived, room-scoped publisher token."""
    authorize_device_request(request, device_id=body.device_id)

    limiter: FixedWindowLimiter = request.app.state.session_limiter
    client = _client_of(request)
    if not limiter.allow(client):
        raise CapacityError(
            "too many session requests",
            retry_after_s=round(limiter.retry_after_s(client), 1),
        )

    settings = request.app.state.settings
    sessions: SessionRegistry = request.app.state.sessions
    metrics: MetricsRegistry = request.app.state.metrics

    # docs/06 splits minting from owning: the gateway names the session because
    # it is the only component present when one starts, and Memory owns what
    # the session *was*. Registering before creating means the adopted id is
    # the one everything downstream uses, so Memory can start minting its own
    # later with no code change here.
    #
    # A Memory that is unreachable is not fatal -- `register_session` falls
    # back to the local id. Refusing the session would stop the glasses
    # connecting over a problem that recovers on its own.
    adopted = await register_session(
        settings, session_id=new_session_id(), device_id=body.device_id
    )
    # create() still enforces the allowlist and the concurrency budget, so an
    # unauthorized device is refused here even though Memory saw the id.
    session = sessions.create(device_id=body.device_id, session_id=adopted)

    try:
        minted = mint_access_token(
            settings,
            identity=publisher_identity(session.device_id),
            room=session.room,
            role="publisher",
        )
    except Exception:
        # Do not leave a session holding a slot when no one can ever join it.
        sessions.end(session.session_id)
        raise

    # Join the room now so the gateway is already subscribed when the
    # publisher arrives, rather than racing its first track.
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is not None and settings.media_source == "livekit":
        try:
            await supervisor.join(session)
        except Exception:
            sessions.end(session.session_id)
            logger.exception(
                "could not join the livekit room",
                extra={"session_id": session.session_id, "room": session.room},
            )
            raise UnavailableError("could not join the livekit room") from None

    # Session counting lives in the pipeline, which observes sessions from
    # every source. Counting here too would double count the LiveKit path,
    # where a session is created here and announced again when media arrives.
    metrics.tokens_issued += 1
    logger.info(
        "issued a publisher token",
        extra={
            "session_id": session.session_id,
            "device_id": session.device_id,
            "room": session.room,
            "identity": minted.identity,
            "ttl_s": settings.token_ttl_s,
        },
    )

    return _token_response(
        session=session,
        minted=minted,
        livekit_url=settings.client_livekit_url,
    )


@router.post("/{session_id}/token", response_model=SessionTokenResponse)
def refresh_session_token(request: Request, session_id: str) -> SessionTokenResponse:
    """Re-mint a publisher token without splitting the logical session."""
    settings = request.app.state.settings
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)
    # A client refreshing on schedule is active even if its media connection is
    # between epochs. Keep the logical session from expiring underneath it.
    authorize_device_request(request, device_id=session.device_id)
    session.touch()
    minted = mint_access_token(
        settings,
        identity=publisher_identity(session.device_id),
        room=session.room,
        role="publisher",
    )
    request.app.state.metrics.tokens_issued += 1
    logger.info(
        "refreshed a publisher token",
        extra={
            "session_id": session.session_id,
            "device_id": session.device_id,
            "identity": minted.identity,
        },
    )
    return _token_response(
        session=session,
        minted=minted,
        livekit_url=settings.client_livekit_url,
    )


@router.post("/{session_id}/viewer", response_model=SessionTokenResponse)
def create_viewer_token(request: Request, session_id: str) -> SessionTokenResponse:
    """Issue a read-only room grant for the operator console."""
    authorize_request(request)
    settings = request.app.state.settings
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)
    minted = mint_access_token(
        settings,
        identity=viewer_identity(session.session_id),
        room=session.room,
        role="viewer",
    )
    request.app.state.metrics.tokens_issued += 1
    logger.info(
        "issued a viewer token",
        extra={
            "session_id": session.session_id,
            "device_id": session.device_id,
            "identity": minted.identity,
        },
    )
    return _token_response(
        session=session,
        minted=minted,
        livekit_url=settings.client_livekit_url,
    )


@router.get("", response_model=SessionListResponse)
def list_sessions(request: Request) -> SessionListResponse:
    authorize_request(request)
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    return SessionListResponse(sessions=[_summarize(s) for s in sessions.active()])


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(request: Request, session_id: str) -> Response:
    """End a session and tell subscribers why it stopped."""
    sessions: SessionRegistry = request.app.state.sessions
    session = sessions.get(session_id)
    authorize_device_request(request, device_id=session.device_id)

    session = sessions.end(session_id)
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is not None:
        await supervisor.leave(session_id)
    # The pipeline emits session_ended to subscribers and owns the counter.
    request.app.state.pipeline.session_ended(
        session_id=session.session_id, reason="session_deleted"
    )
    request.app.state.device_events.close_session(session.session_id, "session_ended")
    request.app.state.manual_triggers.clear(session.session_id)
    logger.info("session deleted", extra={"session_id": session.session_id})

    return Response(status_code=status.HTTP_204_NO_CONTENT)
