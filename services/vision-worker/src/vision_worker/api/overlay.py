"""Live detections, for a viewer drawing them over its own video.

A viewer connects, receives one `overlay_hello` describing what it is watching,
then one `OverlayFrame` per processed frame for as long as it keeps up.

No pixels cross this socket. The console publishes the camera itself, so it
already has the frames and only needs coordinates -- see `overlay/hub.py` for
why that is the whole design rather than an optimisation.

Close codes match `media_gateway.api.stream` so the two relay-shaped sockets in
this system behave the same way: 1008 for a policy violation the client caused,
1011 for a condition it cannot fix, 1001 on shutdown.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from visual_memory_vision_contract.protocol import SCHEMA_VERSION

from vision_worker.deps import authorize_websocket
from vision_worker.errors import UnauthorizedError
from vision_worker.overlay.hub import OverlayHub

logger = logging.getLogger(__name__)

router = APIRouter(tags=["overlay"])

CLOSE_UNAUTHORIZED = 1008
CLOSE_CAPACITY = 1011
CLOSE_GOING_AWAY = 1001


@router.websocket("/v1/overlay")
async def overlay(websocket: WebSocket) -> None:
    try:
        authorize_websocket(websocket)
    except UnauthorizedError as exc:
        # Accept first, then close with a reason. A handshake rejected outright
        # gives the client a bare failure with nothing to act on -- the same
        # reasoning as the gateway's relay sockets.
        await websocket.accept()
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason=exc.code)
        return

    hub: OverlayHub | None = websocket.app.state.overlay_hub
    if hub is None:
        await websocket.accept()
        await websocket.close(code=CLOSE_CAPACITY, reason="overlay_disabled")
        return

    session_id = websocket.query_params.get("session_id")
    subscriber = hub.subscribe(session_id=session_id)
    if subscriber is None:
        await websocket.accept()
        await websocket.close(code=CLOSE_CAPACITY, reason="too_many_overlay_viewers")
        return

    await websocket.accept()
    try:
        # What the viewer is looking at, before any frames arrive. A console
        # connected to a pipeline that is not receiving video would otherwise
        # sit silent and indistinguishable from a broken socket.
        await websocket.send_json(
            {
                "type": "overlay_hello",
                "schema_version": SCHEMA_VERSION,
                "source_fps": websocket.app.state.settings.source_fps,
                "detector_kind": websocket.app.state.settings.detector_kind,
                "depth_kind": websocket.app.state.settings.depth_kind,
                "session_id": session_id,
            }
        )
        while True:
            frame = await subscriber.next()
            if frame is None:
                break
            await websocket.send_text(frame.model_dump_json())
    except WebSocketDisconnect:
        # The ordinary way a viewer leaves: a closed tab. Not worth a log line
        # above debug, and never worth an error.
        logger.debug("overlay viewer disconnected")
    finally:
        hub.unsubscribe(subscriber)
        # `application_state`, not `client_state`. Starlette gates `send` on
        # what *this* side has already sent, and `close` is a send: on shutdown
        # the close frame has usually gone out before this runs, and checking
        # the client's state instead raises `Cannot call "send" once a close
        # message has been sent` out of the ASGI app. Harmless to the stream,
        # noisy in the logs, and it only appears when the server stops -- which
        # is exactly when nobody is reading them.
        if websocket.application_state is WebSocketState.CONNECTED:
            await websocket.close(code=CLOSE_GOING_AWAY)


__all__ = ["router"]
