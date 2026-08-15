"""The Agent's sole trusted tool: ask Memory where one object is."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from visual_memory_memory_contract import QueryResponse
from visual_memory_memory_contract.client import MemoryClient, MemoryError_

from agent.config import Settings
from agent.errors import DependencyUnavailableError

AskMemory = Callable[[str, str | None], QueryResponse]


class MemoryTool:
    """Bounded adapter around :class:`MemoryClient`.

    The caller supplies ``session_id`` from the authenticated request context;
    this class never creates one and the language model never gets to choose
    one. A fresh client per call keeps ownership and shutdown deterministic.
    """

    def __init__(self, settings: Settings, *, ask_memory: AskMemory | None = None) -> None:
        self._settings = settings
        self._ask_memory = ask_memory or self._ask_over_http

    def _ask_over_http(self, label: str, session_id: str | None) -> QueryResponse:
        token = self._settings.resolved_memory_api_token
        with MemoryClient(
            self._settings.memory_base_url,
            token=token.get_secret_value() if token else None,
            timeout=self._settings.request_timeout_s,
        ) as client:
            return client.ask(label=label, session_id=session_id)

    async def where_is(self, label: str, session_id: str | None) -> QueryResponse:
        """Return Memory's complete query response without flattening it."""
        try:
            return await asyncio.to_thread(self._ask_memory, label, session_id)
        except MemoryError_ as exc:
            raise DependencyUnavailableError("memory service is unavailable") from exc


__all__ = ["AskMemory", "MemoryTool"]
