"""Liveness and readiness.

The generated version of this file imported the module-level `app` and skipped
the lifespan, which worked when readiness was a constant. It is not: readiness
reports whether the database can actually be reached, so it needs the lifespan
that opens one. Asserting the real behaviour is the point -- a readiness probe
that answers "ready" without checking anything is worse than none.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from application_memory.config import Settings
from application_memory.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


async def get(app: FastAPI, path: str) -> tuple[int, dict[str, str]]:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(path)
    return response.status_code, response.json()


async def test_liveness_does_not_depend_on_the_database(app: FastAPI) -> None:
    """Liveness answers unless the loop is wedged.

    Tying it to the database would make a transient storage problem look like a
    dead process and get the container killed instead of retried.
    """
    status_code, body = await get(app, "/health/live")

    assert status_code == 200
    assert body["status"] == "ok"


async def test_readiness_checks_the_database(app: FastAPI) -> None:
    status_code, body = await get(app, "/health/ready")

    assert status_code == 200
    assert body["status"] == "ready"


async def test_readiness_reports_not_ready_when_the_database_stops_answering(
    app: FastAPI,
) -> None:
    """The database going away *after* startup must be reported, not crash.

    A database that cannot be opened at startup is a different case: the
    service exits, which is correct -- a process that cannot ever work should
    not linger reporting unready. This covers the other half, where storage
    fails while the service is already running.
    """
    async with app.router.lifespan_context(app):
        healthy = await AsyncClient(transport=ASGITransport(app=app), base_url="http://test").get(
            "/health/ready"
        )
        assert healthy.status_code == 200

        app.state.engine.dispose()
        app.state.sessions = _factory_that_raises()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def _factory_that_raises() -> object:
    class Broken:
        def __call__(self) -> Broken:
            return self

        def __enter__(self) -> Broken:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, *_: object) -> None:
            raise OSError("database file is gone")

    return Broken()
