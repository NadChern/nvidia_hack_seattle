from __future__ import annotations

import socket

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from agent.config import Settings
from agent.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_status_reports_boundary_and_never_the_key() -> None:
    secret = "sk-this-must-never-appear"
    app = create_app(Settings(environment="ci", llm_api_key=SecretStr(secret)))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/status")

    assert response.status_code == 200
    assert response.json() == {
        "backend": "local",
        "model": "openai/qwen3:4b",
        "endpoint_host": "127.0.0.1",
        "vision_endpoint_host": "127.0.0.1",
        "registration_capture_seconds": 6.0,
        "registration_timeout_s": 20.0,
        "metrics": {
            "queries": 0,
            "guard_passed": 0,
            "guard_vetoed": {},
            "hands_free_ignored": 0,
            "hands_free_triggered": 0,
            "hands_free_replies": 0,
            "hands_free_errors": 0,
            "assist_requests_started": 0,
            "assist_transcripts_suppressed": 0,
            "assist_gate_closed": 0,
            "assist_gate_opened": 0,
            "registrations_started": 0,
            "registrations_succeeded": 0,
            "registrations_failed": 0,
        },
    }
    assert secret not in response.text


async def test_status_uses_the_endpoint_scope_resolved_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def local_address(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", local_address)
    settings = Settings(environment="ci", llm_base_url="http://model.internal/v1")
    app = create_app(settings)

    def resolver_unavailable(*_args: object, **_kwargs: object) -> None:
        raise socket.gaierror("temporary resolver outage")

    monkeypatch.setattr(socket, "getaddrinfo", resolver_unavailable)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/status")

    assert response.status_code == 200
    assert response.json()["backend"] == "local"
    assert response.json()["endpoint_host"] == "model.internal"


async def test_status_requires_the_configured_bearer_token() -> None:
    token = "a" * 32
    app = create_app(
        Settings(
            environment="ci",
            agent_backend="stub",
            internal_api_token=SecretStr(token),
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        refused = await client.get("/v1/status")
        accepted = await client.get("/v1/status", headers={"authorization": f"Bearer {token}"})

    assert refused.status_code == 401
    assert accepted.status_code == 200
