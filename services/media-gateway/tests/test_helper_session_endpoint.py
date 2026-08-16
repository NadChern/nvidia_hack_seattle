"""`POST /v1/sessions/{id}/helper` -- the grant only an accepted assist call earns."""

from __future__ import annotations

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.main import create_app

pytestmark = pytest.mark.anyio

SECRET = "a-livekit-secret-of-at-least-32-chars"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def livekit_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "ci",
        "media_source": "scripted",
        "scripted_frame_interval_s": 30.0,
        "livekit_api_key": "test-key",
        "livekit_api_secret": SECRET,
        "livekit_url": "ws://127.0.0.1:7880",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def client_for(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_helper_token_is_refused_with_no_accepted_assist_call() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            response = await http.post(f"/v1/sessions/{created['session_id']}/helper")

    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


async def test_the_two_tab_flow_request_accept_then_helper_token() -> None:
    """Tab A raises a request; tab B accepts it and is minted a helper grant.

    This is the server side of the two-tab milestone: no phone, no glasses,
    just the request/accept/helper sequence any client follows.
    """
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            session_id = created["session_id"]

            await http.post(f"/v1/assist/{session_id}/request")
            accepted = await http.post(f"/v1/assist/{session_id}/accept")
            helper = await http.post(f"/v1/sessions/{session_id}/helper")

            # The session is still an ordinary viewer target too -- accepting
            # an assist call must not disturb the operator console's own path.
            viewer = await http.post(f"/v1/sessions/{session_id}/viewer")

    assert accepted.status_code == 200
    assert helper.status_code == 200
    body = helper.json()
    assert body["session_id"] == session_id
    assert body["identity"] == f"helper-{session_id}"
    grants = jwt.decode(body["token"], SECRET, algorithms=["HS256"])["video"]
    # Not scoped to microphone-only at the grant level right now -- see
    # HELPER_PUBLISH_SOURCES in tokens.py for why, and test_helper_token.py
    # for the xfail tracking a proper fix.
    assert "canPublishSources" not in grants
    assert viewer.status_code == 200


async def test_accepting_twice_leaves_only_one_helper_token_mintable() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            session_id = created["session_id"]
            await http.post(f"/v1/assist/{session_id}/request")

            first_accept = await http.post(f"/v1/assist/{session_id}/accept")
            second_accept = await http.post(f"/v1/assist/{session_id}/accept")
            # The first accept's grant remains mintable even after the second
            # is refused -- refusing a duplicate accept must not revoke it.
            helper = await http.post(f"/v1/sessions/{session_id}/helper")

    assert first_accept.status_code == 200
    assert second_accept.status_code == 409
    assert helper.status_code == 200
