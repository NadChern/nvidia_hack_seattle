"""Bounded Google ADK runner and in-memory conversational sessions."""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from visual_memory_memory_contract import QueryResponse

from agent.agent import REQUEST_SESSION_STATE
from agent.config import Settings
from agent.errors import AgentServiceError, DependencyUnavailableError
from agent.stub import DraftAnswer

APP_NAME = "visual-memory-agent"
USER_ID = "wearer"


class BoundedSessionService(InMemorySessionService):
    """ADK's in-memory service with persisted events trimmed by invocation."""

    def trim_to_turns(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        max_turns: int,
    ) -> None:
        session = self.sessions.get(app_name, {}).get(user_id, {}).get(session_id)
        if session is None:
            return

        invocation_ids: list[str] = []
        for event in session.events:
            invocation_id = event.invocation_id
            if invocation_id and invocation_id not in invocation_ids:
                invocation_ids.append(invocation_id)
        keep = frozenset(invocation_ids[-max_turns:])
        session.events = [event for event in session.events if event.invocation_id in keep]


class AdkRunnerBackend:
    """One shared Runner, serialized per conversation session."""

    def __init__(
        self,
        settings: Settings,
        agent: LlmAgent,
        *,
        sessions: BoundedSessionService | None = None,
    ) -> None:
        self._settings = settings
        self._sessions = sessions or BoundedSessionService()
        self._runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=self._sessions,
        )
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        # The service stores few demo sessions, but this still prevents random
        # request IDs from growing the lock table forever.
        self._max_locks = 256

    @property
    def sessions(self) -> BoundedSessionService:
        return self._sessions

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        else:
            self._locks.move_to_end(session_id)
        while len(self._locks) > self._max_locks:
            oldest, old_lock = next(iter(self._locks.items()))
            if old_lock.locked():
                self._locks.move_to_end(oldest)
                break
            self._locks.popitem(last=False)
        return lock

    async def _ensure_session(self, session_id: str) -> None:
        existing = await self._sessions.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
        )
        if existing is None:
            await self._sessions.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
            )

    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        # Anonymous API calls still need an ADK conversation key, but the
        # generated value is internal and is never sent to Memory.
        adk_session_id = session_id or f"anonymous-{uuid.uuid4().hex}"
        lock = self._lock_for(adk_session_id)

        async with lock:
            await self._ensure_session(adk_session_id)
            reply = ""
            tool_result: QueryResponse | None = None
            try:
                events = self._runner.run_async(
                    user_id=USER_ID,
                    session_id=adk_session_id,
                    new_message=types.Content(role="user", parts=[types.Part(text=text)]),
                    state_delta={REQUEST_SESSION_STATE: session_id or ""},
                    # One tool call plus one final model response. The small
                    # extra allowance lets ADK surface a malformed attempt
                    # without creating an unbounded agent loop.
                    run_config=RunConfig(max_llm_calls=3),
                )
                async for event in events:
                    for response in event.get_function_responses():
                        if response.name != "where_is" or response.response is None:
                            continue
                        payload: Any = response.response
                        tool_result = QueryResponse.model_validate(payload)
                    if event.is_final_response() and event.content and event.content.parts:
                        text_parts = [part.text for part in event.content.parts if part.text]
                        if text_parts:
                            reply = "".join(text_parts).strip()
            except AgentServiceError:
                raise
            except Exception as exc:
                raise DependencyUnavailableError("language model is unavailable") from exc
            finally:
                self._sessions.trim_to_turns(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=adk_session_id,
                    max_turns=self._settings.max_turns_kept,
                )

            return DraftAnswer(text=reply, tool_result=tool_result)


__all__ = ["APP_NAME", "USER_ID", "AdkRunnerBackend", "BoundedSessionService"]
