"""What the published port must not hand out.

These lived in `test_dev_publisher.py` because the schema gate and the
publisher page shared one condition -- `environment != "deploy"`. The page is
gone; the gate is not, and it protects something more valuable than a
development aid, so it keeps its own tests rather than leaving with the file
it happened to be filed under.
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def dev_app() -> FastAPI:
    return create_app(
        Settings(environment="dev", media_source="scripted", scripted_frame_interval_s=30.0)
    )


def deploy_app() -> FastAPI:
    return create_app(
        Settings(
            environment="deploy",
            media_source="scripted",
            scripted_frame_interval_s=30.0,
            internal_api_token="an-internal-token-of-at-least-32-chars",
            device_id_allowlist=("glasses-01",),
        )
    )


async def get(app: FastAPI, path: str):  # type: ignore[no-untyped-def]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        return await http.get(path)


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
async def test_the_api_schema_is_not_served_in_deploy(path: str) -> None:
    """The published port must not hand out the API surface.

    Every other route in deploy refuses an unauthenticated caller. Serving the
    schema -- session minting, the relay, return audio, and their field shapes
    -- to anyone who can reach the port undoes that for no benefit.
    """
    response = await get(deploy_app(), path)

    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/openapi.json", "/docs"])
async def test_the_api_schema_is_still_served_in_dev(path: str) -> None:
    response = await get(dev_app(), path)

    assert response.status_code == 200
