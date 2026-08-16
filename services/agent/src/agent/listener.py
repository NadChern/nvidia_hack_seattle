"""Wake-prefix-gated hands-free transcript listener."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterable
from typing import Protocol

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

#: The only question shape this assistant answers. Kept in one place so the
#: anchored and scanning forms below cannot drift apart.
_QUESTION_ALTERNATION = (
    r"(?:where\b|do\s+you\s+know\s+where\b|can\s+you\s+tell\s+me\s+where\b|"
    r"could\s+you\s+tell\s+me\s+where\b)"
)

_QUESTION_SHAPE = re.compile(rf"^{_QUESTION_ALTERNATION}")
_REGISTRATION_ALTERNATION = (
    r"(?:remember|register|scan|learn|save|memorize)\s+(?:my|our|the|this|these|those)\b"
)
_SUPPORTED_INTENT_SHAPE = re.compile(rf"^(?:{_QUESTION_ALTERNATION}|{_REGISTRATION_ALTERNATION})")

#: The same supported shapes starting at any word boundary. Used only after a deliberate
#: press, never for the wake word: the button already establishes intent, so
#: scanning costs nothing there, whereas after a wake prefix the question must
#: follow the prefix or "hey memory" stops meaning anything.
_QUESTION_SHAPE_ANYWHERE = re.compile(
    rf"(?:^|(?<=\W))(?:{_QUESTION_ALTERNATION}|{_REGISTRATION_ALTERNATION})"
)


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


def triggered_question(text: str, wake_prefixes: str | tuple[str, ...]) -> str | None:
    """Find a wake prefix followed by a bounded where-question.

    Prefixes may appear after a disfluency or leaked reply audio. The supported
    question shape remains mandatory; scanning is not permission to trigger on
    an ordinary mention of the product name.
    """
    normalized = " ".join(text.casefold().split())
    configured = (wake_prefixes,) if isinstance(wake_prefixes, str) else wake_prefixes
    matches: list[tuple[int, str]] = []

    for configured_prefix in configured:
        prefix = " ".join(configured_prefix.casefold().split())
        if not prefix:
            continue
        start = 0
        while (hit := normalized.find(prefix, start)) != -1:
            end = hit + len(prefix)
            before_ok = hit == 0 or not normalized[hit - 1].isalnum()
            after_ok = end == len(normalized) or not normalized[end].isalnum()
            if before_ok and after_ok:
                question = normalized[end:].lstrip(" ,:;.!?-")
                if question and _SUPPORTED_INTENT_SHAPE.match(question) is not None:
                    matches.append((hit, question))
            start = hit + 1

    if not matches:
        return None
    return min(matches, key=lambda match: match[0])[1]


def contains_wake_prefix(text: str, wake_prefixes: str | tuple[str, ...]) -> bool:
    """Whether a wake prefix was spoken, regardless of what followed it.

    `triggered_question` deliberately answers a narrower question: prefix *and*
    a supported question. This one exists to notice "Hey memory." on its own,
    which is what a wearer says before pausing to think.
    """
    normalized = " ".join(text.casefold().split())
    configured = (wake_prefixes,) if isinstance(wake_prefixes, str) else wake_prefixes

    for configured_prefix in configured:
        prefix = " ".join(configured_prefix.casefold().split())
        if not prefix:
            continue
        start = 0
        while (hit := normalized.find(prefix, start)) != -1:
            end = hit + len(prefix)
            before_ok = hit == 0 or not normalized[hit - 1].isalnum()
            after_ok = end == len(normalized) or not normalized[end].isalnum()
            if before_ok and after_ok:
                return True
            start = hit + 1
    return False


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
        #: session_id -> monotonic deadline for a wake prefix whose utterance
        #: carried no question. Single-use; see `wake_carry_over_s`.
        self._pending_wake: dict[str, float] = {}

    def _arm_wake_carry_over(self, session_id: str) -> None:
        if self._settings.wake_carry_over_s > 0:
            self._pending_wake[session_id] = time.monotonic() + self._settings.wake_carry_over_s

    def _consume_wake_carry_over(self, session_id: str) -> bool:
        """True once, if a wake prefix arrived in a recent earlier utterance."""
        deadline = self._pending_wake.pop(session_id, None)
        return deadline is not None and deadline > time.monotonic()

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

    async def _manual_question(self, transcript: Transcript) -> str | None:
        """Forward a whole utterance that followed a deliberate press.

        The press *is* the intent signal, so the press path does not pre-filter
        by shape -- the agent's LLM router decides find vs. register vs. no
        supported intent (a clean no-tool reply). This is what lets natural
        phrasings like "remember these keys" or "save this mug" register by
        voice. The wake-word path still gates by shape (`_SUPPORTED_INTENT_SHAPE`
        / `_QUESTION_SHAPE_ANYWHERE`), where always-on ambient audio makes a
        cheap pre-filter load-bearing.

        The arm is consumed first now: with no shape filter there is nothing to
        discard a consumed press against, so a press always results in a forward
        and is never silently spent on a shape miss. Safe here because the UI
        emits one transcript per hold ("a transcript appears only after you stop
        speaking"), so the arm maps to exactly one complete utterance. The cost
        is one gateway round-trip per non-wake transcript to check the arm --
        acceptable, since only a real press returns armed.
        """
        try:
            armed = await self._events.consume_manual_trigger(transcript.session_id)
        except Exception as exc:
            logger.warning(
                "could not consume manual trigger",
                extra={
                    "session_id": transcript.session_id,
                    "error_type": type(exc).__name__,
                },
            )
            return None
        if not armed:
            return None
        return " ".join(transcript.text.casefold().split()) or None

    def _carried_over_question(self, transcript: Transcript) -> str | None:
        """Accept a bare question when the wake prefix was the previous utterance.

        The VAD ends an utterance at any pause longer than its silence window,
        and a wake phrase invites exactly such a pause -- "Hey memory." then a
        beat, then the question. Without this, the prefix and the question
        arrive as two transcripts and neither can fire alone, which reads to
        the wearer as being cut off mid-sentence.

        Both gates still hold, only split across two utterances: a prefix was
        spoken recently, and this utterance is a supported where-question.
        """
        normalized = " ".join(transcript.text.casefold().split())
        found = _QUESTION_SHAPE_ANYWHERE.search(normalized)
        if found is None:
            return None
        if not self._consume_wake_carry_over(transcript.session_id):
            return None
        logger.info(
            "question answered against a wake prefix from the previous utterance",
            extra={"session_id": transcript.session_id},
        )
        return normalized[found.start() :]

    async def process(self, transcript: Transcript) -> bool:
        """Handle one transcript. False means it stopped before any model call."""
        await self._send_transcript_event(transcript)
        prefixes = self._settings.accepted_wake_prefixes
        question = triggered_question(transcript.text, prefixes)

        if question is None:
            question = self._carried_over_question(transcript)
        if question is None:
            question = await self._manual_question(transcript)

        if question is None:
            # "Hey memory." on its own is not a question, but it is a wearer
            # who is about to ask one. Hold the wake open rather than making
            # them start over.
            if contains_wake_prefix(transcript.text, prefixes):
                self._arm_wake_carry_over(transcript.session_id)
                logger.info(
                    "wake prefix held open for the next utterance",
                    extra={
                        "session_id": transcript.session_id,
                        "carry_over_s": self._settings.wake_carry_over_s,
                    },
                )
            self._metrics.record_hands_free_ignored()
            return False

        self._metrics.record_hands_free_triggered()
        logger.info(
            "hands-free query triggered",
            extra={"session_id": transcript.session_id, "epoch_id": transcript.epoch_id},
        )
        try:
            started = time.perf_counter()
            draft = await self._backend.query(question, transcript.session_id)
            if draft.registration_started:
                # The tool-owned background workflow speaks its own fixed
                # prompt and terminal line. Sending the model draft here would
                # duplicate audio and incorrectly guard it as a where-answer.
                return True
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
        return {item.session_id for item in listing.sessions if item.publisher_present}

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
        while True:
            try:
                async with connect(
                    self._stt_url(session_id),
                    additional_headers=self._headers(),
                    max_size=64_000,
                    open_timeout=self._settings.request_timeout_s,
                ) as websocket:
                    await self._consume_messages(session_id, websocket)
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
                            # A wake held open for a session that has gone away
                            # must not outlive it, and the map must not grow
                            # one entry per session for the life of the process.
                            self._pending_wake.pop(session_id, None)
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
    "triggered_question",
]
