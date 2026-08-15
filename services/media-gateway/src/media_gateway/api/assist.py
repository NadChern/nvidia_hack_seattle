"""Remote-assist request, listing, and accept -- the server side of "she
pressed the button".

No LiveKit and no tokens here. `POST /v1/sessions/{session_id}/helper` (the
grant that actually lets a helper into the room) lives in `api/sessions.py`
beside `create_viewer_token`, which it mirrors.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from media_gateway.deps import (
    authorize_device_request,
    authorize_helper_request,
    authorize_helper_websocket,
)
from media_gateway.domain.assist import (
    AssistAcceptedEvent,
    AssistEventHub,
    AssistEventSubscriber,
    AssistRequest,
    AssistRequestedEvent,
    AssistRequestRegistry,
    AssistState,
)
from media_gateway.domain.device_events import AssistDeviceEvent, DeviceEventHub
from media_gateway.domain.session import SessionRegistry
from media_gateway.errors import CapacityError, ConflictError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assist"], prefix="/v1/assist")

CLOSE_CAPACITY = 1011
CLOSE_BACKPRESSURE = 1011
CLOSE_GOING_AWAY = 1001


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _request_payload(request: AssistRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "session_id": request.session_id,
        "device_id": request.device_id,
        "state": request.state,
        "requested_at": request.requested_at.isoformat(),
        "expires_at": request.expires_at.isoformat(),
    }


def _notify_hud(hub: DeviceEventHub, *, session_id: str, state: AssistState) -> None:
    hub.publish(session_id, AssistDeviceEvent(state=state, occurred_at=_utcnow()))


@router.post("/{session_id}/request", status_code=201)
async def request_assist(request: Request, session_id: str) -> dict[str, Any]:
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)
    # Same shape as the manual trigger: the wearer's own device credential
    # works, and so does the operator token -- the console stands in for a
    # wearer with no glasses by using the latter.
    authorize_device_request(request, device_id=session.device_id)

    registry: AssistRequestRegistry = request.app.state.assist_requests
    assist_request = registry.request(session_id=session_id, device_id=session.device_id)

    events: AssistEventHub = request.app.state.assist_events
    events.publish(
        AssistRequestedEvent(
            request_id=assist_request.request_id,
            session_id=session_id,
            occurred_at=assist_request.requested_at,
        )
    )
    _notify_hud(request.app.state.device_events, session_id=session_id, state="requested")

    logger.info(
        "assist requested",
        extra={"session_id": session_id, "request_id": assist_request.request_id},
    )
    return _request_payload(assist_request)


@router.get("/requests")
async def list_assist_requests(request: Request) -> dict[str, list[dict[str, Any]]]:
    authorize_helper_request(request)
    registry: AssistRequestRegistry = request.app.state.assist_requests
    return {"requests": [_request_payload(r) for r in registry.pending()]}


@router.post("/{session_id}/accept")
async def accept_assist(request: Request, session_id: str) -> dict[str, Any]:
    authorize_helper_request(request)
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)

    registry: AssistRequestRegistry = request.app.state.assist_requests
    accepted = registry.accept(session_id)
    if accepted is None:
        raise ConflictError("no pending assist request for this session", session_id=session_id)

    events: AssistEventHub = request.app.state.assist_events
    events.publish(
        AssistAcceptedEvent(
            request_id=accepted.request_id,
            session_id=session_id,
            occurred_at=_utcnow(),
        )
    )
    _notify_hud(request.app.state.device_events, session_id=session_id, state="accepted")

    logger.info(
        "assist accepted",
        extra={"session_id": session_id, "request_id": accepted.request_id},
    )
    return {"state": "accepted", "helper_identity": f"helper-{session.session_id}"}


@router.websocket("/events")
async def stream_assist_events(websocket: WebSocket) -> None:
    try:
        authorize_helper_websocket(websocket)
    except Exception:
        await websocket.accept()
        await websocket.close(code=1008, reason="unauthorized")
        return

    hub: AssistEventHub = websocket.app.state.assist_events
    try:
        subscriber = hub.subscribe()
    except CapacityError as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_CAPACITY, reason=exc.code)
        return

    await websocket.accept()
    await websocket.send_json(
        {
            "schema_version": "1.0",
            "type": "hello",
            "occurred_at": _utcnow().isoformat(),
        }
    )
    logger.info("assist event subscriber connected", extra={"subscribers": len(hub)})

    pump = asyncio.create_task(_pump(websocket, subscriber))
    watch = asyncio.create_task(_watch(websocket))
    try:
        _, pending = await asyncio.wait({pump, watch}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pump.done() and not pump.cancelled():
            pump.result()
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()
        watch.cancel()
        hub.unsubscribe(subscriber)
        logger.info("assist event subscriber disconnected")


async def _pump(websocket: WebSocket, subscriber: AssistEventSubscriber) -> None:
    while True:
        next_event = asyncio.create_task(subscriber.queue.get())
        closed = asyncio.create_task(subscriber.closed.wait())
        done, pending = await asyncio.wait(
            {next_event, closed},
            timeout=websocket.app.state.settings.ws_keepalive_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if not done:
            await websocket.send_json(
                {
                    "schema_version": "1.0",
                    "type": "keepalive",
                    "occurred_at": _utcnow().isoformat(),
                }
            )
            continue
        if closed in done and subscriber.closed.is_set():
            reason = subscriber.close_reason or "closed"
            code = CLOSE_BACKPRESSURE if reason == "event_backpressure" else CLOSE_GOING_AWAY
            await websocket.close(code=code, reason=reason)
            return
        if next_event in done:
            event = next_event.result()
            await websocket.send_json(event.model_dump(mode="json"))


async def _watch(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return


__all__ = [
    "accept_assist",
    "list_assist_requests",
    "request_assist",
    "router",
    "stream_assist_events",
]
