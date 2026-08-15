from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from agent.config import Settings
from agent.events import GatewayEventTransport

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_gateway_event_transport_posts_guard_and_internal_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"subscribers": 1})

    transport = GatewayEventTransport(
        Settings(
            environment="ci",
            internal_api_token=SecretStr("an-internal-token-of-at-least-32-chars"),
        ),
        transport=httpx.MockTransport(handler),
    )

    await transport.send_reply(
        session_id="sess_01",
        question="where are my keys",
        reply="I cannot safely confirm a location.",
        answer_status="unknown",
        object_id=None,
        guard="vetoed:3",
        latency_ms=42,
    )

    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/device/sess_01/events"
    assert request.headers["authorization"] == ("Bearer an-internal-token-of-at-least-32-chars")
    assert body["type"] == "reply"
    assert body["guard"] == "vetoed:3"
    assert body["answer_status"] == "unknown"
