import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from media_gateway.errors import UnavailableError
from media_gateway.readiness import Readiness


@pytest.mark.anyio
async def test_liveness(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_readiness(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_reports_a_failing_check(app: FastAPI) -> None:
    with TestClient(app) as client:
        app.state.readiness.register("livekit", lambda: "unreachable")
        response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["reason"] == "livekit: unreachable"


def test_liveness_ignores_dependency_failures(app: FastAPI) -> None:
    """A LiveKit outage must not get an otherwise healthy process restarted."""
    with TestClient(app) as client:
        app.state.readiness.register("livekit", lambda: "unreachable")

        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_readiness_fails_immediately_on_shutdown(app: FastAPI) -> None:
    with TestClient(app) as client:
        app.state.readiness.begin_shutdown()
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["reason"] == "shutting_down"


def test_gateway_errors_map_to_typed_responses(app: FastAPI) -> None:
    @app.get("/boom")
    def boom() -> None:
        raise UnavailableError("livekit is unreachable", livekit_url="ws://127.0.0.1:7880")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 503
    assert response.json() == {
        "code": "unavailable",
        "message": "livekit is unreachable",
        "context": {"livekit_url": "ws://127.0.0.1:7880"},
    }


def test_readiness_registry_reports_the_first_failure() -> None:
    readiness = Readiness()
    readiness.register("ok", lambda: None)
    readiness.register("bad", lambda: "down")

    assert readiness.evaluate() == "bad: down"

    readiness.unregister("bad")
    assert readiness.evaluate() is None
