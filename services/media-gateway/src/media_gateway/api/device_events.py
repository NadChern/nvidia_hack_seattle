"""Authenticated transcript/reply events for the glasses HUD and console."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from media_gateway.deps import (
    authorize_device_request,
    authorize_device_websocket,
    authorize_request,
)
from media_gateway.domain.device_events import (
    DeviceEvent,
    DeviceEventHub,
    DeviceEventSubscriber,
)
from media_gateway.domain.manual_trigger import ManualTriggerRegistry
from media_gateway.domain.register_trigger import RegisterTriggerRegistry
from media_gateway.domain.session import SessionRegistry
from media_gateway.errors import CapacityError, NotFoundError, UnauthorizedError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["device-events"])

CLOSE_POLICY = 1008
CLOSE_CAPACITY = 1011
CLOSE_BACKPRESSURE = 1011
CLOSE_GOING_AWAY = 1001


@router.post("/v1/device/{session_id}/events", status_code=202)
async def publish_device_event(
    request: Request,
    session_id: str,
    event: DeviceEvent,
) -> dict[str, int]:
    """Accept one Agent-owned event without waiting for any HUD client.

    `async def` is load-bearing. A sync endpoint runs in Starlette's threadpool,
    and `hub.publish` reaches `asyncio.Queue.put_nowait` and `asyncio.Event.set`
    on the subscriber, both of which resolve loop-owned futures. Calling them
    from off-loop is undefined: a HUD can miss the wakeup for an event that is
    already sitting in its queue. Every transcript and every reply takes this
    path, so it must stay on the loop.
    """
    authorize_request(request)
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    sessions.get(session_id)
    hub: DeviceEventHub = request.app.state.device_events
    return {"subscribers": hub.publish(session_id, event)}


@router.post("/v1/device/{session_id}/manual-trigger")
async def arm_manual_trigger(request: Request, session_id: str) -> dict[str, Any]:
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)
    authorize_device_request(request, device_id=session.device_id)
    triggers: ManualTriggerRegistry = request.app.state.manual_triggers
    return {"expires_at": triggers.arm(session_id)}


@router.post("/v1/device/{session_id}/manual-trigger/consume")
async def consume_manual_trigger(request: Request, session_id: str) -> dict[str, bool]:
    authorize_request(request)
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    sessions.get(session_id)
    triggers: ManualTriggerRegistry = request.app.state.manual_triggers
    return {"armed": triggers.consume(session_id)}


class RegisterTriggerRequest(BaseModel):
    #: Optional item name. Omitted for the placeholder path, where the agent
    #: allocates a name and the wearer renames it in the console later.
    label: str | None = Field(default=None, max_length=128)


@router.post("/v1/device/{session_id}/register")
async def arm_register_trigger(
    request: Request, session_id: str, body: RegisterTriggerRequest | None = None
) -> dict[str, Any]:
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    session = sessions.get(session_id)
    authorize_device_request(request, device_id=session.device_id)
    triggers: RegisterTriggerRegistry = request.app.state.register_triggers
    label = body.label if body is not None else None
    return {"expires_at": triggers.arm(session_id, label)}


@router.post("/v1/device/{session_id}/register/consume")
async def consume_register_trigger(request: Request, session_id: str) -> dict[str, Any]:
    authorize_request(request)
    sessions: SessionRegistry = request.app.state.sessions
    sessions.sweep()
    sessions.get(session_id)
    triggers: RegisterTriggerRegistry = request.app.state.register_triggers
    arm = triggers.consume(session_id)
    if arm is None:
        return {"armed": False, "label": None}
    return {"armed": True, "label": arm.label}


@router.websocket("/v1/device/{session_id}/events")
async def stream_device_events(websocket: WebSocket, session_id: str) -> None:
    sessions: SessionRegistry = websocket.app.state.sessions
    sessions.sweep()
    try:
        session = sessions.get(session_id)
        authorize_device_websocket(websocket, device_id=session.device_id)
    except (NotFoundError, UnauthorizedError) as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_POLICY, reason=exc.code)
        return

    hub: DeviceEventHub = websocket.app.state.device_events
    try:
        subscriber = hub.subscribe(session_id)
    except CapacityError as exc:
        await websocket.accept()
        await websocket.close(code=CLOSE_CAPACITY, reason=exc.code)
        return

    await websocket.accept()
    await websocket.send_json(
        {
            "schema_version": "1.0",
            "type": "hello",
            "session_id": session_id,
            "device_id": session.device_id,
            "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
        }
    )
    logger.info(
        "device event subscriber connected",
        extra={"session_id": session_id, "subscribers": len(hub)},
    )

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
        logger.info("device event subscriber disconnected", extra={"session_id": session_id})


async def _pump(websocket: WebSocket, subscriber: DeviceEventSubscriber) -> None:
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
                    "session_id": subscriber.session_id,
                    "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
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
            payload: dict[str, Any] = event.model_dump(mode="json")
            payload["session_id"] = subscriber.session_id
            await websocket.send_json(payload)


async def _watch(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return


__all__ = ["publish_device_event", "router", "stream_device_events"]
