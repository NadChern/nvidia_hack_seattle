"""Session creation and token issuance."""

import datetime as dt

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.domain.ratelimit import FixedWindowLimiter
from media_gateway.main import create_app

pytestmark = pytest.mark.anyio

SECRET = "a-livekit-secret-of-at-least-32-chars"
TOKEN = "an-internal-token-of-at-least-32-chars"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def livekit_settings(**overrides: object) -> Settings:
    """A gateway with credentials, so tokens can actually be signed."""
    base: dict[str, object] = {
        "environment": "ci",
        # Scripted keeps the lifespan from requiring a reachable LiveKit while
        # still exercising the real token path.
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


async def create(app: FastAPI, device_id: str = "glasses-01", **kwargs: object):  # type: ignore[no-untyped-def]
    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            return await http.post("/v1/sessions", json={"device_id": device_id}, **kwargs)  # type: ignore[arg-type]


async def test_creating_a_session_returns_everything_needed_to_join() -> None:
    app = create_app(livekit_settings())

    response = await create(app)

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"].startswith("sess_")
    assert body["device_id"] == "glasses-01"
    assert body["room"] == f"vma-{body['session_id']}"
    assert body["livekit_url"] == "ws://127.0.0.1:7880"
    assert body["identity"] == "glasses-01"
    assert body["token"]


async def test_the_token_is_scoped_to_one_room_with_least_privilege() -> None:
    app = create_app(livekit_settings())

    body = (await create(app)).json()
    claims = jwt.decode(body["token"], SECRET, algorithms=["HS256"])

    grants = claims["video"]
    assert grants["roomJoin"] is True
    assert grants["room"] == body["room"]
    assert grants["canPublish"] is True
    assert grants["canSubscribe"] is True
    # A data channel would be an unaudited side path between participants.
    assert grants.get("canPublishData", False) is False
    assert grants.get("roomAdmin", False) is False
    assert grants.get("roomCreate", False) is False
    assert grants.get("roomRecord", False) is False


async def test_the_token_expires_within_the_configured_ttl() -> None:
    app = create_app(livekit_settings(token_ttl_s=120))

    body = (await create(app)).json()
    claims = jwt.decode(body["token"], SECRET, algorithms=["HS256"])

    # LiveKit stamps `nbf` rather than `iat`.
    assert claims["exp"] - claims["nbf"] == 120
    expires_at = dt.datetime.fromisoformat(body["expires_at"])
    assert expires_at > dt.datetime.now(dt.UTC)


async def test_a_caller_cannot_request_a_longer_lifetime_than_configured() -> None:
    from media_gateway.transport.tokens import mint_access_token

    settings = livekit_settings(token_ttl_s=60)

    minted = mint_access_token(settings, identity="i", room="r", role="publisher", ttl_s=86_400)
    claims = jwt.decode(minted.token, SECRET, algorithms=["HS256"])

    assert claims["exp"] - claims["nbf"] == 60


async def test_viewer_token_is_read_only_and_scoped_to_the_existing_room() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            response = await http.post(f"/v1/sessions/{created['session_id']}/viewer")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == created["session_id"]
    assert body["device_id"] == created["device_id"]
    assert body["room"] == created["room"]
    assert body["identity"] == f"viewer-{created['session_id']}"
    claims = jwt.decode(body["token"], SECRET, algorithms=["HS256"])
    grants = claims["video"]
    assert grants["roomJoin"] is True
    assert grants["room"] == created["room"]
    assert grants.get("canPublish", False) is False
    assert grants.get("canPublishData", False) is False
    assert grants["canSubscribe"] is True


async def test_refresh_reuses_the_session_room_and_publisher_identity() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            response = await http.post(f"/v1/sessions/{created['session_id']}/token")

    assert response.status_code == 200
    refreshed = response.json()
    assert refreshed["session_id"] == created["session_id"]
    assert refreshed["device_id"] == created["device_id"]
    assert refreshed["room"] == created["room"]
    assert refreshed["identity"] == created["identity"]
    assert refreshed["token"]


@pytest.mark.parametrize("suffix", ["token", "viewer"])
async def test_tokens_for_an_unknown_session_are_explicit(suffix: str) -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            response = await http.post(f"/v1/sessions/sess_missing/{suffix}")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize("suffix", ["token", "viewer"])
async def test_an_expired_session_cannot_be_resurrected_by_minting(suffix: str) -> None:
    app = create_app(livekit_settings(session_ttl_s=1))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()
            session = app.state.sessions.get(created["session_id"])
            session.last_seen_at -= dt.timedelta(seconds=2)
            response = await http.post(f"/v1/sessions/{created['session_id']}/{suffix}")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_the_identity_is_the_device_so_it_survives_a_rejoin() -> None:
    """Identity must be stable; the track SID is what marks a new epoch."""
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            first = await http.post("/v1/sessions", json={"device_id": "glasses-01"})
            await http.delete(f"/v1/sessions/{first.json()['session_id']}")
            second = await http.post("/v1/sessions", json={"device_id": "glasses-01"})

    assert first.json()["identity"] == second.json()["identity"] == "glasses-01"
    assert first.json()["session_id"] != second.json()["session_id"]


async def test_sessions_without_livekit_credentials_are_refused_clearly() -> None:
    """Scripted mode has no credentials; the failure must say so."""
    app = create_app(
        Settings(environment="ci", media_source="scripted", scripted_frame_interval_s=30.0)
    )

    response = await create(app)

    assert response.status_code == 503
    assert response.json()["code"] == "unavailable"


async def test_a_refused_token_does_not_leak_a_session_slot() -> None:
    """A session nobody can join must not hold the concurrency budget."""
    app = create_app(
        Settings(
            environment="ci",
            media_source="scripted",
            scripted_frame_interval_s=30.0,
            max_concurrent_sessions=1,
        )
    )

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            await http.post("/v1/sessions", json={"device_id": "glasses-01"})
            second = await http.post("/v1/sessions", json={"device_id": "glasses-01"})

    # Still 503 for the missing credentials, not 429 from an orphaned slot.
    assert second.status_code == 503


async def test_the_concurrency_limit_is_enforced() -> None:
    app = create_app(livekit_settings(max_concurrent_sessions=1))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            first = await http.post("/v1/sessions", json={"device_id": "glasses-01"})
            second = await http.post("/v1/sessions", json={"device_id": "glasses-02"})

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json()["code"] == "capacity_exhausted"


async def test_an_unlisted_device_is_refused() -> None:
    app = create_app(livekit_settings(device_id_allowlist=("glasses-01",)))

    response = await create(app, device_id="glasses-99")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def test_repeated_requests_are_rate_limited() -> None:
    app = create_app(livekit_settings(sessions_rate_limit_per_minute=2))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            codes = [
                (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).status_code
                for _ in range(4)
            ]

    assert codes[:2] == [201, 201]
    assert codes[2] == 429


async def test_an_unauthenticated_request_is_refused() -> None:
    app = create_app(livekit_settings(internal_api_token=TOKEN))

    response = await create(app)

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


async def test_the_configured_token_is_accepted() -> None:
    app = create_app(livekit_settings(internal_api_token=TOKEN))

    response = await create(app, headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 201


@pytest.mark.parametrize("suffix", ["token", "viewer"])
async def test_session_token_endpoints_require_internal_authentication(suffix: str) -> None:
    app = create_app(livekit_settings(internal_api_token=TOKEN))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-01"},
                    headers={"Authorization": f"Bearer {TOKEN}"},
                )
            ).json()
            response = await http.post(f"/v1/sessions/{created['session_id']}/{suffix}")

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


async def test_sessions_can_be_listed_and_deleted() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            created = (await http.post("/v1/sessions", json={"device_id": "glasses-01"})).json()

            listed = (await http.get("/v1/sessions")).json()
            assert [s["session_id"] for s in listed["sessions"]] == [created["session_id"]]

            deleted = await http.delete(f"/v1/sessions/{created['session_id']}")
            assert deleted.status_code == 204

            after = (await http.get("/v1/sessions")).json()
            assert after["sessions"] == []


async def test_deleting_an_unknown_session_is_explicit() -> None:
    app = create_app(livekit_settings())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as http:
            response = await http.delete("/v1/sessions/sess_missing")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_the_token_never_appears_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(livekit_settings())

    with caplog.at_level("INFO"):
        body = (await create(app)).json()

    assert body["token"] not in caplog.text
    assert "eyJ" not in caplog.text


def test_the_limiter_resets_after_its_window() -> None:
    clock = {"now": 0.0}
    limiter = FixedWindowLimiter(limit=1, window_s=60.0, now=lambda: clock["now"])

    assert limiter.allow("a") is True
    assert limiter.allow("a") is False

    clock["now"] = 61.0
    assert limiter.allow("a") is True


def test_the_limiter_tracks_clients_independently() -> None:
    limiter = FixedWindowLimiter(limit=1, window_s=60.0)

    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False
