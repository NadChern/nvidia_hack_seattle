from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from agent.config import Settings
from agent.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize("path,status", [("/health/live", "ok"), ("/health/ready", "ready")])
async def test_health(path: str, status: str) -> None:
    app = create_app(Settings(environment="ci", agent_backend="stub"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert response.json() == {"status": status, "service": "agent"}
