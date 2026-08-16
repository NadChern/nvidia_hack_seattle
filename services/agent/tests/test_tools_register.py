"""Registration tool preserves labels/session and translates dependency failures."""

from __future__ import annotations

import pytest
from visual_memory_memory_contract.client import MemoryError_

from agent.config import Settings
from agent.errors import DependencyUnavailableError
from agent.tools.register import RegisterTool, RegistrationOutcome

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_register_wraps_blocking_operation_and_returns_terminal_outcome() -> None:
    calls: list[tuple[str, str]] = []

    def register(label: str, session_id: str) -> RegistrationOutcome:
        calls.append((label, session_id))
        return RegistrationOutcome("object_keys", label, True, "enrollment_complete", 3)

    tool = RegisterTool(Settings(environment="ci"), register_operation=register)

    outcome = await tool.register("  Keys  ", "sess_1")

    assert outcome.succeeded is True
    assert outcome.selected_views == 3
    assert calls == [("keys", "sess_1")]


async def test_register_requires_a_real_session_before_any_side_effect() -> None:
    calls: list[tuple[str, str]] = []

    def register(label: str, session_id: str) -> RegistrationOutcome:
        calls.append((label, session_id))
        return RegistrationOutcome(None, label, False, "should_not_run")

    tool = RegisterTool(Settings(environment="ci"), register_operation=register)

    outcome = await tool.register("keys", "")

    assert outcome.reason_code == "session_required"
    assert calls == []


async def test_dependency_failure_is_explicit() -> None:
    def register(label: str, session_id: str) -> RegistrationOutcome:
        del label, session_id
        raise MemoryError_("offline")

    tool = RegisterTool(Settings(environment="ci"), register_operation=register)

    with pytest.raises(DependencyUnavailableError):
        await tool.register("keys", "sess_1")
