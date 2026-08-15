"""Relay WebSocket endpoints.

A subscriber receives, in order: `stream_hello`, a synthetic `epoch_started`
for every epoch already running, then live media. Re-announcing active epochs
is what lets a consumer that connected mid-epoch, or reconnected after a drop,
reset tracker state rather than resume against stale identities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from visual_memory_media_contract.protocol import StreamKind, VideoEncoding

from media_gateway.deps import authorize_websocket
from media_gateway.errors import CapacityError, UnauthorizedError
from media_gateway.pipeline import MediaPipeline
from media_gateway.relay.hub import RelayHub, Subscriber

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])

#: Close codes. 1008 is a policy violation, 1011 an internal condition the
#: client cannot fix; both are what `docs/12` specifies for these cases.
CLOSE_UNAUTHORIZED = 1008
CLOSE_CAPACITY = 1011
CLOSE_BACKPRESSURE = 1011
CLOSE_GOING_AWAY = 1001


@router.websocket("/v1/stream/video")
async def stream_video(
    websocket: WebSocket,
    encoding: Annotated[VideoEncoding, Query()] = "jpeg",
) -> None:
    await _serve(websocket, stream_kind="video", encoding=encoding)


@router.websocket("/v1/stream/audio")
async def stream_audio(websocket: WebSocket) -> None:
    await _serve(websocket, stream_kind="audio", encoding=None)


async def _serve(
    websocket: WebSocket,
    *,
    stream_kind: StreamKind,
    encoding: VideoEncoding | None,
) -> None:
    try:
        authorize_websocket(websocket)
    except UnauthorizedError as exc:
        # Accept first: a rejected handshake gives the client no reason at all.
        await websocket.accept()
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason=exc.code)
        return

    hub: RelayHub = websocket.app.state.hub
    pipeline: MediaPipeline = websocket.app.state.pipeline
    keepalive_s: float = websocket.app.state.settings.ws_keepalive_s

    try:
        subscriber = hub.subscribe(stream_kind=stream_kind, encoding=encoding)
    except CapacityError as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_CAPACITY, reason=exc.code)
        return

    await websocket.accept()
    logger.info(
        "relay subscriber connected",
        extra={"stream_kind": stream_kind, "encoding": encoding, "subscribers": len(hub)},
    )

    try:
        await websocket.send_bytes(pipeline.build_hello(stream_kind=stream_kind, encoding=encoding))
        for frame in pipeline.replay_epochs_for(stream_kind):
            await websocket.send_bytes(frame)

        # Pump and watch race: whichever finishes first ends the connection.
        pump = asyncio.create_task(_pump(websocket, subscriber, pipeline, stream_kind, keepalive_s))
        watch = asyncio.create_task(_watch_for_close(websocket))
        done, pending = await asyncio.wait({pump, watch}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        # A disconnect can race with cancellation while the pump is inside
        # send_bytes(). Always reap the losing task so its disconnect does not
        # escape later through AnyIO's unhandled-task reporting in tests (or
        # through the event loop's exception handler in production).
        await asyncio.gather(*pending, return_exceptions=True)
        if pump in done and not pump.cancelled():
            # Surface a genuine pump failure rather than losing it.
            pump.result()
    except WebSocketDisconnect:
        logger.info("relay subscriber disconnected", extra={"stream_kind": stream_kind})
    finally:
        hub.unsubscribe(subscriber)


async def _pump(
    websocket: WebSocket,
    subscriber: Subscriber,
    pipeline: MediaPipeline,
    stream_kind: StreamKind,
    keepalive_s: float,
) -> None:
    """Forward frames until the subscriber closes or the socket goes away."""
    while True:
        try:
            payload = await asyncio.wait_for(subscriber.next(), timeout=keepalive_s)
        except TimeoutError:
            # Idle. A keepalive lets a consumer tell "no publisher yet" from
            # "this socket is dead".
            await websocket.send_bytes(pipeline.keepalive(stream_kind))
            continue

        if payload is None:
            await _close_for(websocket, subscriber)
            return

        await websocket.send_bytes(payload)


async def _watch_for_close(websocket: WebSocket) -> None:
    """Return as soon as the client goes away.

    The relay is one-way, so without this nothing ever reads the socket and a
    departed consumer is discovered only when the next send fails -- up to
    `ws_keepalive_s` later on an idle stream, which is exactly the state a
    gateway sits in with no glasses connected. Until then the subscriber holds
    one of the hub's slots, and an audio subscriber keeps accumulating into a
    bounded queue that would eventually be reported as backpressure rather
    than as the disconnect it actually is.

    Inbound frames are ignored rather than refused: a consumer that speaks out
    of turn is harmless here, and the relay has nothing to say back.
    """
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return


async def _close_for(websocket: WebSocket, subscriber: Subscriber) -> None:
    reason = subscriber.close_reason or "closed"
    code = CLOSE_BACKPRESSURE if reason == "audio_backpressure" else CLOSE_GOING_AWAY
    if websocket.client_state is WebSocketState.CONNECTED:
        await websocket.close(code=code, reason=reason)
