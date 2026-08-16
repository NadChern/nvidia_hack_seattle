"""Remote-assist tool keeps session scope trusted and failures explicit."""

from __future__ import annotations

import httpx
import pytest

from agent.config import Settings
from agent.errors import DependencyUnavailableError
from agent.tools.assist import AssistRequestOutcome, AssistTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_request_preserves_the_trusted_session() -> None:
    calls: list[str] = []

    async def request(session_id: str) -> AssistRequestOutcome:
        calls.append(session_id)
        return AssistRequestOutcome(True, session_id, "assist_01", "requested", "requested")

    tool = AssistTool(Settings(environment="ci"), request_operation=request)

    outcome = await tool.request("sess_01")

    assert outcome.requested is True
    assert outcome.state == "requested"
    assert calls == ["sess_01"]


async def test_request_requires_a_real_session_before_any_side_effect() -> None:
    calls: list[str] = []

    async def request(session_id: str) -> AssistRequestOutcome:
        calls.append(session_id)
        return AssistRequestOutcome(True, session_id, "assist_01", "requested", "requested")

    tool = AssistTool(Settings(environment="ci"), request_operation=request)

    outcome = await tool.request("")

    assert outcome.requested is False
    assert outcome.reason_code == "session_required"
    assert calls == []


async def test_dependency_failure_is_explicit() -> None:
    async def request(session_id: str) -> AssistRequestOutcome:
        del session_id
        raise httpx.ConnectError("offline")

    tool = AssistTool(Settings(environment="ci"), request_operation=request)

    with pytest.raises(DependencyUnavailableError):
        await tool.request("sess_01")
