"""Hands-free transcript listener with return-audio and assist-call gates."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterable
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field
from visual_memory_memory_contract import AnswerStatus
from websockets.asyncio.client import connect

from agent.config import Settings
from agent.events import GatewayEventTransport
from agent.guard import guard_reply
from agent.metrics import AgentMetrics
from agent.models import GuardVerdict
from agent.reply import ReplyTransport
from agent.stub import QueryBackend

logger = logging.getLogger(__name__)


class Transcript(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str = Field(min_length=1, max_length=2_000)
    session_id: str
    epoch_id: str
    pts_samples_start: int
    samples: int
    sample_rate: int


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    session_id: str
    publisher_present: bool
    #: Backward-compatible accepted-call signal from an older Gateway.
    assist_active: bool = False
    #: Pending and accepted requests both suppress model-bound audio. The
    #: Gateway reports null after an ended or expired request.
    assist_state: Literal["requested", "accepted", "ended"] | None = None


class SessionList(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    sessions: tuple[SessionSummary, ...] = ()


class ReplySender(Protocol):
    async def send(self, session_id: str, text: str) -> None: ...


class EventSender(Protocol):
    async def consume_manual_trigger(self, session_id: str) -> bool: ...

    async def send_transcript(
        self,
        *,
        session_id: str,
        text: str,
        epoch_id: str,
        pts_samples_start: int,
        samples: int,
        sample_rate: int,
    ) -> None: ...

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
    ) -> None: ...


class HandsFreeListener:
    """Discovers live gateway sessions and owns one STT socket per session."""

    def __init__(
        self,
        settings: Settings,
        backend: QueryBackend,
        reply: ReplySender | None = None,
        events: EventSender | None = None,
        *,
        metrics: AgentMetrics | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._reply = reply or ReplyTransport(settings)
        self._events = events or GatewayEventTransport(settings)
        self._metrics = metrics or AgentMetrics()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._suppress_until: dict[str, float] = {}
        self._assist_suppressed: set[str] = set()

    def _set_assist_suppressed(self, session_id: str, suppressed: bool) -> None:
        if suppressed and session_id not in self._assist_suppressed:
            self._assist_suppressed.add(session_id)
            self._metrics.record_assist_gate_closed()
            logger.info("remote-assist audio gate closed", extra={"session_id": session_id})
        elif not suppressed and session_id in self._assist_suppressed:
            self._assist_suppressed.remove(session_id)
            self._metrics.record_assist_gate_opened()
            logger.info("remote-assist audio gate opened", extra={"session_id": session_id})

    def _headers(self) -> dict[str, str]:
        token = self._settings.internal_api_token
        return {"authorization": f"Bearer {token.get_secret_value()}"} if token is not None else {}

    async def _send_transcript_event(self, transcript: Transcript) -> None:
        try:
            await self._events.send_transcript(
                session_id=transcript.session_id,
                text=transcript.text,
                epoch_id=transcript.epoch_id,
                pts_samples_start=transcript.pts_samples_start,
                samples=transcript.samples,
                sample_rate=transcript.sample_rate,
            )
        except Exception as exc:
            logger.warning(
                "could not push transcript to device events",
                extra={
                    "session_id": transcript.session_id,
                    "error_type": type(exc).__name__,
                },
            )

    async def _send_reply_event(
        self,
        *,
        transcript: Transcript,
        question: str,
        reply: str,
        answer_status: AnswerStatus | None,
        object_id: str | None,
        guard: GuardVerdict,
        latency_ms: int,
    ) -> None:
        try:
            await self._events.send_reply(
                session_id=transcript.session_id,
                question=question,
                reply=reply,
                answer_status=answer_status,
                object_id=object_id,
                guard=guard,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning(
                "could not push reply to device events",
                extra={
                    "session_id": transcript.session_id,
                    "error_type": type(exc).__name__,
                },
            )

    async def process(self, transcript: Transcript) -> bool:
        """Forward a transcript only while its remote-assist gate is open."""
        if transcript.session_id in self._assist_suppressed:
            self._metrics.record_assist_transcript_suppressed()
            logger.info(
                "transcript ignored during remote-assist audio suppression",
                extra={"session_id": transcript.session_id},
            )
            return False

        await self._send_transcript_event(transcript)
        if time.monotonic() < self._suppress_until.get(transcript.session_id, 0.0):
            self._metrics.record_hands_free_ignored()
            logger.info(
                "transcript ignored during return-audio echo cooldown",
                extra={"session_id": transcript.session_id},
            )
            return False

        question = " ".join(transcript.text.casefold().split())
        self._metrics.record_hands_free_triggered()
        logger.info(
            "hands-free query triggered",
            extra={"session_id": transcript.session_id, "epoch_id": transcript.epoch_id},
        )
        try:
            started = time.perf_counter()
            draft = await self._backend.query(question, transcript.session_id)
            if draft.assist_requested:
                # Close locally before acknowledgement audio or another queued
                # transcript can race the Gateway's next session-state poll.
                self._set_assist_suppressed(transcript.session_id, True)
                self._metrics.record_assist_request_started()
            if draft.registration_started:
                # The tool-owned background workflow speaks its own fixed
                # prompt and terminal line. Sending the model draft here would
                # duplicate audio and incorrectly guard it as a where-answer.
                return True
            if transcript.session_id in self._assist_suppressed and not draft.assist_requested:
                # A Console-created request may become visible while this model
                # turn is in flight. Never speak its stale reply over a human call.
                self._metrics.record_assist_transcript_suppressed()
                return False
            guarded = guard_reply(draft.text, draft.tool_result)
            latency_ms = max(0, round((time.perf_counter() - started) * 1000))
            self._metrics.record_guard(guarded.verdict)
            await self._send_reply_event(
                transcript=transcript,
                question=question,
                reply=guarded.reply,
                answer_status=guarded.answer_status,
                object_id=guarded.object_id,
                guard=guarded.verdict,
                latency_ms=latency_ms,
            )
            await self._reply.send(transcript.session_id, guarded.reply)
            if not draft.assist_requested:
                self._suppress_until[transcript.session_id] = (
                    time.monotonic() + self._settings.reply_echo_suppression_s
                )
            self._metrics.record_hands_free_reply()
            logger.info(
                "hands-free reply sent",
                extra={
                    "session_id": transcript.session_id,
                    "answer_status": guarded.answer_status,
                    "guard": guarded.verdict,
                },
            )
            return True
        except Exception:
            self._metrics.record_hands_free_error()
            raise

    async def _active_sessions(self, client: httpx.AsyncClient) -> set[str]:
        response = await client.get("/v1/sessions")
        response.raise_for_status()
        listing = SessionList.model_validate(response.json())
        active: set[str] = set()
        reported: set[str] = set()
        for item in listing.sessions:
            reported.add(item.session_id)
            assist_suppresses = item.assist_state in {"requested", "accepted"}
            # ``assist_active`` preserves safety with a Gateway that predates
            # the additive state field and only knows accepted calls.
            assist_suppresses = assist_suppresses or item.assist_active
            self._set_assist_suppressed(item.session_id, assist_suppresses)
            if item.publisher_present and not assist_suppresses:
                active.add(item.session_id)

        # Absence from the authoritative active-session listing means the
        # session ended; do not retain one suppression entry per old session.
        for session_id in self._assist_suppressed - reported:
            self._set_assist_suppressed(session_id, False)
        return active

    def _stt_url(self, session_id: str) -> str:
        base = self._settings.speech_base_url
        scheme = "wss" if base.startswith("https://") else "ws"
        host_and_path = base.split("://", maxsplit=1)[1]
        return f"{scheme}://{host_and_path}/v1/stt/{session_id}"

    async def _consume_messages(
        self,
        session_id: str,
        messages: AsyncIterable[str | bytes],
    ) -> None:
        """Process each transcript without reconnecting for downstream errors."""

        async for message in messages:
            if not isinstance(message, str):
                continue
            try:
                transcript = Transcript.model_validate(json.loads(message))
                # Session scope comes from the socket selected by this service,
                # not from untrusted JSON on that socket.
                if transcript.session_id != session_id:
                    continue
                await self.process(transcript)
                if session_id in self._assist_suppressed:
                    # Close the STT socket immediately instead of waiting for
                    # the next Gateway polling interval.
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Memory, model, guard, and reply failures are per-turn. Closing
                # the healthy STT socket here drops audio during the reconnect
                # delay and cannot repair the downstream dependency.
                logger.warning(
                    "hands-free transcript processing failed",
                    extra={
                        "session_id": session_id,
                        "error_type": type(exc).__name__,
                    },
                )

    async def _listen(self, session_id: str) -> None:
        while session_id not in self._assist_suppressed:
            try:
                async with connect(
                    self._stt_url(session_id),
                    additional_headers=self._headers(),
                    max_size=64_000,
                    open_timeout=self._settings.request_timeout_s,
                ) as websocket:
                    await self._consume_messages(session_id, websocket)
                    if session_id in self._assist_suppressed:
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "hands-free STT listener reconnecting",
                    extra={"session_id": session_id, "error_type": type(exc).__name__},
                )
                await asyncio.sleep(self._settings.listener_reconnect_s)

    async def run(self) -> None:
        """Run until cancelled, reconciling listeners with gateway sessions."""
        try:
            async with httpx.AsyncClient(
                base_url=self._settings.gateway_base_url,
                headers=self._headers(),
                timeout=self._settings.request_timeout_s,
            ) as client:
                while True:
                    try:
                        active = await self._active_sessions(client)
                        for session_id in active - self._tasks.keys():
                            self._tasks[session_id] = asyncio.create_task(
                                self._listen(session_id),
                                name=f"hands-free-{session_id}",
                            )
                        for session_id in self._tasks.keys() - active:
                            task = self._tasks.pop(session_id)
                            task.cancel()
                            self._suppress_until.pop(session_id, None)
                        await asyncio.sleep(self._settings.session_poll_interval_s)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "hands-free session discovery failed",
                            extra={"error_type": type(exc).__name__},
                        )
                        await asyncio.sleep(self._settings.session_poll_interval_s)
        finally:
            tasks = list(self._tasks.values())
            self._tasks.clear()
            self._suppress_until.clear()
            self._assist_suppressed.clear()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "EventSender",
    "HandsFreeListener",
    "ReplySender",
    "SessionList",
    "SessionSummary",
    "Transcript",
]
