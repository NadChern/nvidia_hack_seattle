"""Pending and accepted remote assistance must close model-bound audio."""

from __future__ import annotations

import httpx
import pytest

from agent.config import Settings
from agent.listener import HandsFreeListener
from agent.stub import DraftAnswer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeBackend:
    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        raise AssertionError("not exercised by this test")


def _sessions_response(sessions: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={"sessions": sessions})


async def test_an_accepted_assist_call_excludes_its_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sessions_response(
            [
                {
                    "session_id": "sess_quiet",
                    "publisher_present": True,
                    "assist_active": False,
                    "assist_state": None,
                },
                {
                    "session_id": "sess_ringing",
                    "publisher_present": True,
                    "assist_active": False,
                    "assist_state": "requested",
                },
                {
                    "session_id": "sess_on_a_call",
                    "publisher_present": True,
                    "assist_active": True,
                    "assist_state": "accepted",
                },
                {
                    "session_id": "sess_no_publisher",
                    "publisher_present": False,
                    "assist_active": False,
                },
            ]
        )

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        active = await listener._active_sessions(client)  # noqa: SLF001

    assert active == {"sess_quiet"}


async def test_an_older_gateway_accepted_call_is_still_suppressed() -> None:
    """The old boolean remains a fail-safe when `assist_state` is absent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _sessions_response(
            [{"session_id": "sess_01", "publisher_present": True, "assist_active": True}]
        )

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        active = await listener._active_sessions(client)  # noqa: SLF001

    assert active == set()


async def test_session_poll_failure_does_not_reopen_a_closed_gate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    listener._set_assist_suppressed("sess_01", True)  # noqa: SLF001
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        with pytest.raises(httpx.HTTPStatusError):
            await listener._active_sessions(client)  # noqa: SLF001

    assert listener._assist_suppressed == {"sess_01"}  # noqa: SLF001


async def test_a_session_reappears_once_its_assist_call_ends() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        on_a_call = calls["n"] == 1
        return _sessions_response(
            [
                {
                    "session_id": "sess_01",
                    "publisher_present": True,
                    "assist_active": on_a_call,
                    "assist_state": "accepted" if on_a_call else None,
                }
            ]
        )

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        during_call = await listener._active_sessions(client)  # noqa: SLF001
        after_call = await listener._active_sessions(client)  # noqa: SLF001

    assert during_call == set()
    assert after_call == {"sess_01"}
