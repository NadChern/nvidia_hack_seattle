"""Remote-assist request, listing, and accept."""

from __future__ import annotations

import asyncio
import datetime as dt
import json

import httpx
import pytest
import websockets

from media_gateway.config import Settings
from media_gateway.domain.assist import AssistRequestRegistry
from media_gateway.main import create_app
from tests.serving import serve

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


async def paired_session(http: httpx.AsyncClient) -> tuple[dict[str, object], str]:
    code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
    credential = (
        await http.post(
            "/v1/pairing/claim",
            json={"pairing_code": code, "device_id": "glasses-01"},
        )
    ).json()["credential"]
    session = (
        await http.post(
            "/v1/sessions",
            json={"device_id": "glasses-01"},
            headers=auth(credential),
        )
    ).json()
    return session, credential


async def pair_device(http: httpx.AsyncClient, *, device_id: str) -> str:
    """Claim a fresh device credential for a device with no session of its
    own -- a remote helper's phone, unlike the wearer's glasses, never calls
    `POST /v1/sessions`.
    """
    code = (await http.post("/v1/pairing", headers=auth())).json()["pairing_code"]
    claimed = await http.post(
        "/v1/pairing/claim", json={"pairing_code": code, "device_id": device_id}
    )
    return str(claimed.json()["credential"])


# --- Pure domain tests -------------------------------------------------
#
# Expiry needs an injected clock rather than a real 30s+ sleep -- the same
# reason test_device_events_api.py drives DeviceEventHub directly for its
# backpressure test instead of going through the HTTP app.


def test_expired_request_is_not_listed() -> None:
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    registry = AssistRequestRegistry(ttl_s=30, now=lambda: now)

    registry.request(session_id="sess_01", device_id="glasses-01")
    assert len(registry.pending()) == 1

    now += dt.timedelta(seconds=31)
    assert registry.pending() == []


def test_a_second_button_press_before_anyone_answers_does_not_reset_the_request() -> None:
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    registry = AssistRequestRegistry(ttl_s=30, now=lambda: now)

    first = registry.request(session_id="sess_01", device_id="glasses-01")
    second = registry.request(session_id="sess_01", device_id="glasses-01")

    assert first.request_id == second.request_id
    assert first.expires_at == second.expires_at


# --- HTTP tests ----------------------------------------------------------


async def test_request_creates_a_pending_request() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])

            requested = await http.post(
                f"/v1/assist/{session_id}/request", headers=auth(credential)
            )
            listed = await http.get("/v1/assist/requests", headers=auth())

    body = requested.json()
    assert requested.status_code == 201
    assert body["session_id"] == session_id
    assert body["state"] == "requested"
    assert body["request_id"]

    requests = listed.json()["requests"]
    assert len(requests) == 1
    assert requests[0]["request_id"] == body["request_id"]


async def test_request_for_unknown_session_is_404() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            response = await http.post("/v1/assist/sess_does_not_exist/request", headers=auth())

    assert response.status_code == 404


async def test_accept_is_consume_once() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            await http.post(f"/v1/assist/{session_id}/request", headers=auth(credential))

            first = await http.post(f"/v1/assist/{session_id}/accept", headers=auth())
            second = await http.post(f"/v1/assist/{session_id}/accept", headers=auth())
            listed = await http.get("/v1/assist/requests", headers=auth())

    assert first.status_code == 200
    assert first.json() == {"state": "accepted", "helper_identity": f"helper-{session_id}"}
    assert second.status_code == 409
    # Accepted, not pending -- it must not still show up as ringing.
    assert listed.json()["requests"] == []


async def test_accepting_with_no_request_raised_is_refused() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, _credential = await paired_session(http)
            session_id = str(session["session_id"])

            response = await http.post(f"/v1/assist/{session_id}/accept", headers=auth())

    assert response.status_code == 409


async def test_listing_and_accepting_require_authentication() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            await http.post(f"/v1/assist/{session_id}/request", headers=auth(credential))

            listed = await http.get("/v1/assist/requests")
            accepted = await http.post(f"/v1/assist/{session_id}/accept")

    assert listed.status_code == 401
    assert accepted.status_code == 401


async def test_a_paired_helper_device_can_list_and_accept_a_request() -> None:
    """A remote helper is never the wearer's own device -- unlike the
    request-assist step, listing and accepting must accept *any* paired
    device's credential, not just one matching a specific `device_id`.
    """
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, wearer_credential = await paired_session(http)
            session_id = str(session["session_id"])
            await http.post(f"/v1/assist/{session_id}/request", headers=auth(wearer_credential))

            helper_credential = await pair_device(http, device_id="helper-01")

            listed = await http.get("/v1/assist/requests", headers=auth(helper_credential))
            accepted = await http.post(
                f"/v1/assist/{session_id}/accept", headers=auth(helper_credential)
            )

    assert listed.status_code == 200
    assert len(listed.json()["requests"]) == 1
    assert accepted.status_code == 200
    assert accepted.json() == {"state": "accepted", "helper_identity": f"helper-{session_id}"}


async def test_an_invalid_device_credential_is_refused_for_listing_and_accepting() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, _wearer_credential = await paired_session(http)
            session_id = str(session["session_id"])

            listed = await http.get("/v1/assist/requests", headers=auth("v1.not-a-real-credential"))
            accepted = await http.post(
                f"/v1/assist/{session_id}/accept", headers=auth("v1.not-a-real-credential")
            )

    assert listed.status_code == 401
    assert accepted.status_code == 401


# --- Assist events WebSocket ---------------------------------------------


async def test_a_paired_helper_device_receives_request_and_accept_events() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, wearer_credential = await paired_session(http)
            session_id = str(session["session_id"])
            helper_credential = await pair_device(http, device_id="helper-01")

            url = f"ws://{host}/v1/assist/events"
            async with websockets.connect(url, additional_headers=auth(helper_credential)) as socket:
                hello = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

                requested = await http.post(
                    f"/v1/assist/{session_id}/request", headers=auth(wearer_credential)
                )
                on_requested = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

                accepted = await http.post(
                    f"/v1/assist/{session_id}/accept", headers=auth(helper_credential)
                )
                on_accepted = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

    assert hello["type"] == "hello"
    assert requested.status_code == 201
    assert on_requested["type"] == "assist_requested"
    assert on_requested["session_id"] == session_id
    assert accepted.status_code == 200
    assert on_accepted["type"] == "assist_accepted"
    assert on_accepted["session_id"] == session_id


async def test_a_helper_device_credential_is_refused_from_the_query_string() -> None:
    """Same rule as the device event stream: a long-lived credential must
    never travel where request lines are recorded, so only the internal
    token keeps the query-string concession.
    """
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            helper_credential = await pair_device(http, device_id="helper-01")

            async with websockets.connect(
                f"ws://{host}/v1/assist/events?token={helper_credential}"
            ) as refused:
                with pytest.raises(websockets.ConnectionClosed) as caught:
                    await asyncio.wait_for(refused.recv(), timeout=5)

            async with websockets.connect(
                f"ws://{host}/v1/assist/events", additional_headers=auth(helper_credential)
            ) as accepted:
                hello = json.loads(await asyncio.wait_for(accepted.recv(), timeout=5))

    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "unauthorized"
    assert hello["type"] == "hello"
