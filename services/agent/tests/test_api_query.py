from __future__ import annotations

import pytest
from conftest import confirmed_answer
from httpx import ASGITransport, AsyncClient

from agent.config import Settings
from agent.guard import NO_TOOL_REPLY, registration_message
from agent.main import create_app
from agent.stub import StubLlm
from agent.tools.memory import MemoryTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def app_with_stub():  # type: ignore[no-untyped-def]
    def ask(label: str, session_id: str | None):  # type: ignore[no-untyped-def]
        assert label == "keys"
        assert session_id == "sess_01"
        return confirmed_answer()

    settings = Settings(environment="ci", agent_backend="stub")
    return create_app(settings, backend=StubLlm(MemoryTool(settings, ask_memory=ask)))


async def test_query_with_stub_backend() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app_with_stub()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/agent/query",
            json={"text": "Where did I leave my keys?", "session_id": "sess_01"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "The keys are on the living room coffee table.",
        "answer_status": "confirmed",
        "object_id": "object-keys-01",
        "guard": "passed",
        "latency_ms": response.json()["latency_ms"],
    }


async def test_registration_query_returns_only_the_fixed_prompt() -> None:
    class Starter:
        def start(self, *, label: str, session_id: str) -> bool:
            assert (label, session_id) == ("keys", "sess_01")
            return True

    settings = Settings(environment="ci", agent_backend="stub")
    memory = MemoryTool(settings, ask_memory=lambda _label, _session: confirmed_answer())
    app = create_app(settings, backend=StubLlm(memory, Starter()))  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/agent/query",
            json={"text": "Remember my keys", "session_id": "sess_01"},
        )

    assert response.status_code == 200
    assert response.json()["reply"] == registration_message("prompt", "keys")
    assert response.json()["guard"] == "registration:prompt"
    assert response.json()["answer_status"] is None


async def test_no_tool_call_yields_the_fixed_line() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app_with_stub()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/agent/query",
            json={"text": "Tell me a joke.", "session_id": "sess_01"},
        )

    assert response.status_code == 200
    assert response.json()["reply"] == NO_TOOL_REPLY
    assert response.json()["answer_status"] is None
    assert response.json()["guard"] == "vetoed:1"
