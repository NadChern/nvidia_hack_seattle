"""Pairing codes and least-privilege device credentials."""

import datetime as dt

import pytest
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.domain.pairing import DeviceCredentialSigner, PairingRegistry
from media_gateway.errors import UnauthorizedError
from media_gateway.main import create_app

pytestmark = pytest.mark.anyio

INTERNAL = "an-internal-token-of-at-least-32-chars"
SECRET = "a-livekit-secret-of-at-least-32-chars"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "ci",
        "media_source": "scripted",
        "scripted_frame_interval_s": 30.0,
        "livekit_api_key": "test-key",
        "livekit_api_secret": SECRET,
        "livekit_url": "ws://127.0.0.1:7880",
        "internal_api_token": INTERNAL,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def auth(token: str = INTERNAL) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_pairing_code_is_single_use_and_device_credential_creates_a_session() -> None:
    app = create_app(settings())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            issued = (await http.post("/v1/pairing", headers=auth())).json()
            claim = await http.post(
                "/v1/pairing/claim",
                json={"pairing_code": issued["pairing_code"], "device_id": "glasses-01"},
            )
            repeated = await http.post(
                "/v1/pairing/claim",
                json={"pairing_code": issued["pairing_code"], "device_id": "glasses-01"},
            )
            created = await http.post(
                "/v1/sessions",
                json={"device_id": "glasses-01"},
                headers=auth(claim.json()["credential"]),
            )

    assert claim.status_code == 200
    assert claim.json()["device_id"] == "glasses-01"
    assert repeated.status_code == 401
    assert created.status_code == 201


async def test_device_credential_is_scoped_to_its_device_and_minimal_surfaces() -> None:
    app = create_app(settings())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
            credential = (
                await http.post(
                    "/v1/pairing/claim",
                    json={"pairing_code": code, "device_id": "glasses-01"},
                )
            ).json()["credential"]
            wrong_device = await http.post(
                "/v1/sessions",
                json={"device_id": "glasses-02"},
                headers=auth(credential),
            )
            listing = await http.get("/v1/sessions", headers=auth(credential))
            status = await http.get("/v1/status", headers=auth(credential))

    assert wrong_device.status_code == 401
    assert listing.status_code == 401
    assert status.status_code == 401


async def test_device_credential_refreshes_only_its_own_session() -> None:
    app = create_app(settings())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
            credential = (
                await http.post(
                    "/v1/pairing/claim",
                    json={"pairing_code": code, "device_id": "glasses-01"},
                )
            ).json()["credential"]
            own = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-01"},
                    headers=auth(credential),
                )
            ).json()
            other = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-02"},
                    headers=auth(),
                )
            ).json()
            refreshed = await http.post(
                f"/v1/sessions/{own['session_id']}/token", headers=auth(credential)
            )
            refused = await http.post(
                f"/v1/sessions/{other['session_id']}/token", headers=auth(credential)
            )

    assert refreshed.status_code == 200
    assert refreshed.json()["session_id"] == own["session_id"]
    assert refused.status_code == 401


async def test_device_credential_can_delete_only_its_own_session() -> None:
    app = create_app(settings())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
            credential = (
                await http.post(
                    "/v1/pairing/claim",
                    json={"pairing_code": code, "device_id": "glasses-01"},
                )
            ).json()["credential"]
            own = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-01"},
                    headers=auth(credential),
                )
            ).json()
            other = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-02"},
                    headers=auth(),
                )
            ).json()
            refused = await http.delete(
                f"/v1/sessions/{other['session_id']}", headers=auth(credential)
            )
            deleted = await http.delete(
                f"/v1/sessions/{own['session_id']}", headers=auth(credential)
            )

    assert refused.status_code == 401
    assert deleted.status_code == 204


async def test_pairing_issue_requires_internal_auth_and_claim_honors_allowlist() -> None:
    app = create_app(settings(device_id_allowlist=("glasses-01",)))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            unauthenticated = await http.post("/v1/pairing")
            code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
            refused = await http.post(
                "/v1/pairing/claim",
                json={"pairing_code": code, "device_id": "glasses-02"},
            )

    assert unauthenticated.status_code == 401
    assert refused.status_code == 403


def test_expired_code_and_credential_are_refused() -> None:
    clock = {"now": dt.datetime(2026, 8, 12, tzinfo=dt.UTC)}

    def now() -> dt.datetime:
        return clock["now"]

    signer = DeviceCredentialSigner(secret=b"s" * 32, ttl_s=60, now=now)
    registry = PairingRegistry(ttl_s=30, max_pending=2, signer=signer, now=now)

    code = registry.issue()
    clock["now"] += dt.timedelta(seconds=31)
    with pytest.raises(UnauthorizedError):
        registry.claim(code=code.code, device_id="glasses-01")

    fresh = registry.issue()
    credential = registry.claim(code=fresh.code, device_id="glasses-01")
    clock["now"] += dt.timedelta(seconds=61)
    with pytest.raises(UnauthorizedError):
        signer.verify(credential.credential)
