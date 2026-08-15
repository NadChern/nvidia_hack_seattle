"""Background registration narration and orchestration."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from agent.guard import guard_registration_reply, registration_message
from agent.metrics import AgentMetrics
from agent.tools.register import RegisterTool

logger = logging.getLogger(__name__)


class WorkflowReply(Protocol):
    async def send(self, session_id: str, text: str) -> None: ...


class RegistrationWorkflow:
    """One background task per session; prompt and terminal TTS are scripted."""

    def __init__(
        self,
        tool: RegisterTool,
        reply: WorkflowReply,
        *,
        metrics: AgentMetrics,
    ) -> None:
        self._tool = tool
        self._reply = reply
        self._metrics = metrics
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, *, label: str, session_id: str) -> bool:
        normalized = " ".join(label.strip().casefold().split())
        if not normalized or len(normalized) > 128 or not session_id:
            return False
        existing = self._tasks.get(session_id)
        if existing is not None and not existing.done():
            return False
        self._metrics.record_registration_started()
        task = asyncio.create_task(
            self._run(label=normalized, session_id=session_id),
            name=f"registration-{session_id}",
        )
        self._tasks[session_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(session_id, None))
        return True

    async def drain(self) -> None:
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, *, label: str, session_id: str) -> None:
        prompt = registration_message("prompt", label)
        guarded_prompt = guard_registration_reply(prompt, step="prompt", label=label)
        try:
            await self._reply.send(session_id, guarded_prompt.reply)
            outcome = await self._tool.register(label, session_id)
            step = "succeeded" if outcome.succeeded else "failed"
            terminal = registration_message(step, label)
            guarded_terminal = guard_registration_reply(terminal, step=step, label=label)
            await self._reply.send(session_id, guarded_terminal.reply)
            if outcome.succeeded:
                self._metrics.record_registration_succeeded()
            else:
                self._metrics.record_registration_failed()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._metrics.record_registration_failed()
            logger.warning(
                "registration workflow failed",
                extra={"session_id": session_id, "error_type": type(exc).__name__},
            )
            failure = registration_message("failed", label)
            try:
                guarded_failure = guard_registration_reply(failure, step="failed", label=label)
                await self._reply.send(session_id, guarded_failure.reply)
            except Exception:
                logger.warning(
                    "registration terminal reply could not be sent",
                    extra={"session_id": session_id},
                )


__all__ = ["RegistrationWorkflow", "WorkflowReply"]
