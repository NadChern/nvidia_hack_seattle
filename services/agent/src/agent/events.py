"""Push transcripts and guarded replies to the Gateway HUD event channel."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import httpx
from visual_memory_memory_contract import AnswerStatus

from agent.config import Settings
from agent.models import GuardVerdict


@dataclass(frozen=True, slots=True)
class ConsumedRegister:
    """The result of consuming a register-button press."""

    armed: bool
    label: str | None = None


class GatewayEventTransport:
    """Small bounded HTTP producer; event delivery never owns turn success."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.gateway_base_url
        self._timeout_s = settings.gateway_event_timeout_s
        self._transport = transport
        token = settings.internal_api_token
        self._headers = (
            {"authorization": f"Bearer {token.get_secret_value()}"} if token is not None else {}
        )

    async def _post(self, session_id: str, payload: dict[str, object]) -> None:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.post(f"/v1/device/{session_id}/events", json=payload)
            response.raise_for_status()

    async def consume_manual_trigger(self, session_id: str) -> bool:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.post(f"/v1/device/{session_id}/manual-trigger/consume")
            response.raise_for_status()
            return bool(response.json().get("armed", False))

    async def consume_register_trigger(self, session_id: str) -> ConsumedRegister:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.post(f"/v1/device/{session_id}/register/consume")
            response.raise_for_status()
            body = response.json()
            return ConsumedRegister(
                armed=bool(body.get("armed", False)),
                label=body.get("label"),
            )

    async def send_transcript(
        self,
        *,
        session_id: str,
        text: str,
        epoch_id: str,
        pts_samples_start: int,
        samples: int,
        sample_rate: int,
    ) -> None:
        await self._post(
            session_id,
            {
                "schema_version": "1.0",
                "type": "transcript",
                "text": text,
                "epoch_id": epoch_id,
                "pts_samples_start": pts_samples_start,
                "samples": samples,
                "sample_rate": sample_rate,
                "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )

    async def send_reply(
        self,
        *,
        session_id: str,
        question: str,
        reply: str,
        answer_status: AnswerStatus | None,
        object_id: str | None,
        guard: GuardVerdict,
        latency_ms: int,
    ) -> None:
        await self._post(
            session_id,
            {
                "schema_version": "1.0",
                "type": "reply",
                "question": question,
                "reply": reply,
                "answer_status": answer_status,
                "object_id": object_id,
                "guard": guard,
                "latency_ms": latency_ms,
                "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )


__all__ = ["ConsumedRegister", "GatewayEventTransport"]
