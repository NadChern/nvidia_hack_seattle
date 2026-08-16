"""Bounded remote-assist request adapter with trusted session scope."""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.config import Settings
from agent.errors import DependencyUnavailableError

ASSIST_REQUESTED_REPLY = "I've sent a request to your remote assistant."


@dataclass(frozen=True, slots=True)
class AssistRequestOutcome:
    requested: bool
    session_id: str | None
    request_id: str | None
    state: Literal["requested"] | None
    reason_code: str

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


class _GatewayAssistRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    request_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    state: Literal["requested"]
    requested_at: dt.datetime
    expires_at: dt.datetime

    @model_validator(mode="after")
    def _valid_window(self) -> Self:
        if self.requested_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("remote-assist timestamps must include a timezone")
        if self.expires_at <= self.requested_at:
            raise ValueError("remote-assist request must expire after it was created")
        return self


RequestAssistOperation = Callable[[str], Awaitable[AssistRequestOutcome]]


class AssistTool:
    """Create one idempotent remote-human request through the Media Gateway."""

    def __init__(
        self,
        settings: Settings,
        *,
        request_operation: RequestAssistOperation | None = None,
    ) -> None:
        self._settings = settings
        self._request_operation = request_operation or self._request_over_http

    async def request(self, session_id: str) -> AssistRequestOutcome:
        if not session_id:
            return AssistRequestOutcome(False, None, None, None, "session_required")
        try:
            return await self._request_operation(session_id)
        except (httpx.HTTPError, ValidationError, TimeoutError, OSError, ValueError) as exc:
            raise DependencyUnavailableError("remote-assist service is unavailable") from exc

    async def _request_over_http(self, session_id: str) -> AssistRequestOutcome:
        token = self._settings.internal_api_token
        headers = (
            {"authorization": f"Bearer {token.get_secret_value()}"} if token is not None else {}
        )
        async with httpx.AsyncClient(
            base_url=self._settings.gateway_base_url,
            headers=headers,
            timeout=self._settings.request_timeout_s,
        ) as client:
            response = await client.post(f"/v1/assist/{session_id}/request")
            response.raise_for_status()

        payload = _GatewayAssistRequest.model_validate(response.json())
        if payload.session_id != session_id:
            raise ValueError("remote-assist response session does not match the request")
        return AssistRequestOutcome(
            requested=True,
            session_id=payload.session_id,
            request_id=payload.request_id,
            state=payload.state,
            reason_code="requested",
        )


__all__ = [
    "ASSIST_REQUESTED_REPLY",
    "AssistRequestOutcome",
    "AssistTool",
    "RequestAssistOperation",
]
