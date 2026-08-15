"""The hands-free listener must stop listening to a session mid assist call.

`_active_sessions` is what `run()`'s reconciliation loop diffs against its
live listener tasks: a session dropping out of this set gets its STT task
cancelled on the next poll, and reappearing gets a fresh one started with no
special-casing. Excluding `assist_active` sessions here is therefore the
whole fix -- see trap 3 in role-prompts/Jacky-Remote-Assist.md, and the
gateway side of this in test_room_worker_ingest.py.
"""

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
                {"session_id": "sess_quiet", "publisher_present": True, "assist_active": False},
                {"session_id": "sess_on_a_call", "publisher_present": True, "assist_active": True},
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


async def test_a_session_the_gateway_has_not_upgraded_yet_still_counts_as_quiet() -> None:
    """An older gateway response with no `assist_active` field defaults to False."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _sessions_response([{"session_id": "sess_01", "publisher_present": True}])

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        active = await listener._active_sessions(client)  # noqa: SLF001

    assert active == {"sess_01"}


async def test_a_session_reappears_once_its_assist_call_ends() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        on_a_call = calls["n"] == 1
        return _sessions_response(
            [{"session_id": "sess_01", "publisher_present": True, "assist_active": on_a_call}]
        )

    listener = HandsFreeListener(Settings(environment="ci"), FakeBackend())
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    async with client:
        during_call = await listener._active_sessions(client)  # noqa: SLF001
        after_call = await listener._active_sessions(client)  # noqa: SLF001

    assert during_call == set()
    assert after_call == {"sess_01"}
