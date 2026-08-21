"""Speech-free register-button poller.

The register button decouples enrollment from STT: the wearer focuses "Register"
on the HUD and taps, the glasses arm a gateway trigger, and this listener -- run
independently of hands-free speech -- consumes it and drives a center-anchor
registration. It never touches the STT socket, so a headset with no speech stack
still registers objects.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from agent.config import Settings
from agent.events import ConsumedRegister, GatewayEventTransport

logger = logging.getLogger(__name__)


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    session_id: str
    publisher_present: bool


class SessionList(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    sessions: tuple[SessionSummary, ...] = ()


class RegisterConsumer(Protocol):
    async def consume_register_trigger(self, session_id: str) -> ConsumedRegister: ...


class RegistrationStarter(Protocol):
    def start(self, *, label: str, session_id: str, mode: str = ...) -> bool: ...


def _placeholder_label() -> str:
    """A unique name for a button press that carried none.

    Uniqueness matters: RegisterTool keys idempotency on the label, so two
    presses must not collide. The wearer renames it from the thumbnail in the
    console later.
    """
    return f"item {uuid.uuid4().hex[:6]}"


class RegisterTriggerListener:
    """Polls live sessions for register presses and starts registration."""

    def __init__(
        self,
        settings: Settings,
        workflow: RegistrationStarter,
        *,
        events: RegisterConsumer | None = None,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._events = events or GatewayEventTransport(settings)

    def _headers(self) -> dict[str, str]:
        token = self._settings.internal_api_token
        return {"authorization": f"Bearer {token.get_secret_value()}"} if token is not None else {}

    async def _active_sessions(self, client: httpx.AsyncClient) -> list[str]:
        response = await client.get("/v1/sessions")
        response.raise_for_status()
        listing = SessionList.model_validate(response.json())
        return [item.session_id for item in listing.sessions if item.publisher_present]

    async def _poll_once(self, client: httpx.AsyncClient) -> None:
        for session_id in await self._active_sessions(client):
            armed = await self._events.consume_register_trigger(session_id)
            if not armed.armed:
                continue
            label = armed.label or _placeholder_label()
            started = self._workflow.start(
                label=label, session_id=session_id, mode="center-anchor"
            )
            logger.info(
                "register button consumed",
                extra={
                    "session_id": session_id,
                    "labelled": armed.label is not None,
                    "started": started,
                },
            )

    async def run(self) -> None:
        """Run until cancelled, polling every session-poll interval."""
        async with httpx.AsyncClient(
            base_url=self._settings.gateway_base_url,
            headers=self._headers(),
            timeout=self._settings.request_timeout_s,
        ) as client:
            while True:
                try:
                    await self._poll_once(client)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "register-trigger poll failed",
                        extra={"error_type": type(exc).__name__},
                    )
                await asyncio.sleep(self._settings.session_poll_interval_s)


__all__ = ["RegisterTriggerListener", "SessionList", "SessionSummary"]
