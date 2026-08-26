"""Bounded adapter that creates, captures, and polls one registration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from visual_memory_memory_contract.client import MemoryClient, MemoryError_

from agent.config import Settings
from agent.errors import DependencyUnavailableError


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    object_id: str | None
    label: str
    succeeded: bool
    reason_code: str
    selected_views: int = 0


RegisterOperation = Callable[[str, str, str], RegistrationOutcome]


class RegisterTool:
    """Creates in Memory, arms Vision, and waits for a terminal status."""

    def __init__(
        self,
        settings: Settings,
        *,
        register_operation: RegisterOperation | None = None,
    ) -> None:
        self._settings = settings
        self._register_operation = register_operation or self._register_over_http

    async def register(
        self, label: str, session_id: str, mode: str = "grounded"
    ) -> RegistrationOutcome:
        normalized = " ".join(label.strip().casefold().split())
        if not normalized:
            return RegistrationOutcome(None, "object", False, "invalid_label")
        if not session_id:
            return RegistrationOutcome(None, normalized, False, "session_required")
        try:
            return await asyncio.to_thread(self._register_operation, normalized, session_id, mode)
        except (MemoryError_, httpx.HTTPError, TimeoutError, OSError) as exc:
            raise DependencyUnavailableError("registration services are unavailable") from exc

    def _register_over_http(
        self, label: str, session_id: str, mode: str = "grounded"
    ) -> RegistrationOutcome:
        token = self._settings.resolved_memory_api_token
        token_value = token.get_secret_value() if token is not None else None
        idempotency_key = f"registration/{session_id}/{label}"
        with MemoryClient(
            self._settings.memory_base_url,
            token=token_value,
            timeout=self._settings.request_timeout_s,
        ) as memory:
            enrolled = memory.create_object(label=label, idempotency_key=idempotency_key)

        headers = {"authorization": f"Bearer {token_value}"} if token_value else {}
        with httpx.Client(
            base_url=self._settings.vision_base_url,
            headers=headers,
            timeout=self._settings.request_timeout_s,
        ) as vision:
            response = vision.post(
                f"/v1/objects/{enrolled.object_id}/capture",
                json={
                    "capture_seconds": self._settings.registration_capture_seconds,
                    "mode": mode,
                },
            )
            response.raise_for_status()
            deadline = time.monotonic() + self._settings.registration_timeout_s
            while time.monotonic() < deadline:
                status_response = vision.get(f"/v1/objects/{enrolled.object_id}/status")
                status_response.raise_for_status()
                body = status_response.json()
                state = str(body.get("state", ""))
                if state == "succeeded":
                    return RegistrationOutcome(
                        enrolled.object_id,
                        label,
                        True,
                        str(body.get("reason_code") or "enrollment_complete"),
                        int(body.get("selected_views") or 0),
                    )
                if state == "failed":
                    return RegistrationOutcome(
                        enrolled.object_id,
                        label,
                        False,
                        str(body.get("reason_code") or "registration_failed"),
                        int(body.get("selected_views") or 0),
                    )
                time.sleep(self._settings.registration_poll_interval_s)
        raise TimeoutError("registration did not reach a terminal state")


__all__ = ["RegisterOperation", "RegisterTool", "RegistrationOutcome"]
