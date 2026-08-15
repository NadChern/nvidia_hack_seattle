from __future__ import annotations

import pytest
from conftest import confirmed_answer

from agent.config import Settings
from agent.tools.memory import MemoryTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_where_is_returns_the_whole_query_response() -> None:
    expected = confirmed_answer()
    captured: list[tuple[str, str | None]] = []

    def ask(label: str, session_id: str | None):  # type: ignore[no-untyped-def]
        captured.append((label, session_id))
        return expected

    tool = MemoryTool(Settings(), ask_memory=ask)

    result = await tool.where_is("keys", "sess_01")

    assert result is expected
    assert result.current_location is not None
    assert result.last_confirmed_placement is not None
    assert captured == [("keys", "sess_01")]


async def test_where_is_never_invents_a_session_id() -> None:
    captured: list[str | None] = []

    def ask(label: str, session_id: str | None):  # type: ignore[no-untyped-def]
        del label
        captured.append(session_id)
        return confirmed_answer()

    tool = MemoryTool(Settings(), ask_memory=ask)

    await tool.where_is("keys", None)

    assert captured == [None]
