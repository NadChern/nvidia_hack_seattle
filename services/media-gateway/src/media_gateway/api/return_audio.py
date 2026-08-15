"""Feed synthesized speech back to a device.

The Speech Service streams PCM over the WebSocket. The tone endpoint is a
stand-in so the return path can be exercised end to end before that service
exists.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from media_gateway.deps import authorize_request, authorize_websocket
from media_gateway.errors import NotFoundError, UnauthorizedError
from media_gateway.transport.return_audio import ReturnAudio
from media_gateway.transport.supervisor import SessionSupervisor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["return-audio"], prefix="/v1/return-audio")

CLOSE_UNAUTHORIZED = 1008
CLOSE_NOT_FOUND = 1011
CLOSE_GOING_AWAY = 1001


class TonePlayed(BaseModel):
    session_id: str
    frames: int
    hz: float
    seconds: float


def _return_audio_for(app_state: object, session_id: str) -> ReturnAudio:
    """Find the outbound audio track for a session, or explain why not."""
    supervisor: SessionSupervisor | None = getattr(app_state, "supervisor", None)
    worker = supervisor.worker_for(session_id) if supervisor else None
    if worker is None:
        raise NotFoundError(
            "no live room for this session",
            session_id=session_id,
        )
    if not worker.return_audio.published:  # pragma: no cover - published on join
        raise NotFoundError("return audio is not published", session_id=session_id)
    return worker.return_audio


@router.post("/{session_id}/tone", response_model=TonePlayed)
async def play_tone(
    request: Request,
    session_id: str,
    hz: Annotated[float, Query(gt=20, le=20_000)] = 660.0,
    seconds: Annotated[float, Query(gt=0, le=30)] = 2.0,
) -> TonePlayed:
    """Play a tone on the assistant's track.

    Bounded deliberately: this is reachable by anything that can authenticate,
    and an unbounded tone would occupy the device's speaker indefinitely.
    """
    authorize_request(request)
    audio = _return_audio_for(request.app.state, session_id)

    frames = await audio.play_tone(hz=hz, seconds=seconds)
    logger.info(
        "played a return-audio tone",
        extra={"session_id": session_id, "hz": hz, "seconds": seconds, "frames": frames},
    )
    return TonePlayed(session_id=session_id, frames=frames, hz=hz, seconds=seconds)


@router.websocket("/{session_id}")
async def stream_return_audio(websocket: WebSocket, session_id: str) -> None:
    """Accept PCM for a session's assistant track.

    Binary frames of interleaved little-endian int16 at the configured sample
    rate. Unlike the relay's outbound streams this needs no framing: it is a
    single-purpose one-way feed with no interleaved control messages.
    """
    try:
        authorize_websocket(websocket)
    except UnauthorizedError as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason=exc.code)
        return

    try:
        audio = _return_audio_for(websocket.app.state, session_id)
    except NotFoundError as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_NOT_FOUND, reason=exc.code)
        return

    await websocket.accept()
    logger.info("return audio producer connected", extra={"session_id": session_id})
    frames = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload is None:
                # Text would mean the producer is speaking a different
                # protocol; failing loudly beats silently playing nothing.
                await websocket.close(code=CLOSE_NOT_FOUND, reason="binary_pcm_only")
                return
            # capture_frame awaits until the source has room, so a producer
            # that outruns real time is slowed rather than buffering forever.
            frames += await audio.feed(payload)
    except WebSocketDisconnect:
        pass
    finally:
        logger.info(
            "return audio producer disconnected",
            extra={"session_id": session_id, "frames": frames},
        )
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close(code=CLOSE_GOING_AWAY)
