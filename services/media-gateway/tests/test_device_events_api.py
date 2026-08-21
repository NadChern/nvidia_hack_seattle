"""Gateway-owned HUD transcript and guarded-reply channel."""

import asyncio
import datetime as dt
import inspect
import json

import httpx
import pytest
import websockets

from media_gateway.api import device_events
from media_gateway.config import Settings
from media_gateway.domain.device_events import DeviceEventHub, TranscriptEvent
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


def test_event_endpoints_stay_on_the_event_loop() -> None:
    """Sync handlers run in Starlette's threadpool; these touch loop state.

    `hub.publish` reaches `asyncio.Queue.put_nowait` and `asyncio.Event.set` on
    a subscriber, which resolve loop-owned futures. Off-loop that is undefined
    and a HUD can miss the wakeup for an event already in its queue. The race is
    timing-dependent, so assert the property rather than trying to provoke it.
    """
    for endpoint in (
        device_events.publish_device_event,
        device_events.arm_manual_trigger,
        device_events.consume_manual_trigger,
        device_events.arm_register_trigger,
        device_events.consume_register_trigger,
    ):
        assert inspect.iscoroutinefunction(endpoint), f"{endpoint.__name__} must be async"


def test_slow_device_event_subscriber_is_closed_instead_of_growing_unbounded() -> None:
    hub = DeviceEventHub(queue_size=1, max_subscribers=1)
    subscriber = hub.subscribe("sess_01")
    event = TranscriptEvent(
        text="hello",
        epoch_id="TR_AUDIO_1",
        pts_samples_start=0,
        samples=16_000,
        sample_rate=16_000,
        occurred_at=dt.datetime.now(dt.UTC),
    )

    hub.publish("sess_01", event)
    hub.publish("sess_01", event)

    assert subscriber.closed.is_set()
    assert subscriber.close_reason == "event_backpressure"


async def test_device_receives_transcript_and_guarded_reply_for_its_session() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            url = f"ws://{host}/v1/device/{session_id}/events"
            async with websockets.connect(url, additional_headers=auth(credential)) as socket:
                hello = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
                transcript = {
                    "schema_version": "1.0",
                    "type": "transcript",
                    "text": "hey memory where are my keys",
                    "epoch_id": "TR_AUDIO_1",
                    "pts_samples_start": 0,
                    "samples": 16000,
                    "sample_rate": 16000,
                    "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                reply = {
                    "schema_version": "1.0",
                    "type": "reply",
                    "question": "where are my keys",
                    "reply": "I cannot safely confirm a location.",
                    "answer_status": "unknown",
                    "object_id": None,
                    "guard": "vetoed:3",
                    "latency_ms": 42,
                    "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                first = await http.post(
                    f"/v1/device/{session_id}/events", json=transcript, headers=auth()
                )
                second = await http.post(
                    f"/v1/device/{session_id}/events", json=reply, headers=auth()
                )
                received_transcript = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))
                received_reply = json.loads(await asyncio.wait_for(socket.recv(), timeout=5))

    assert hello["type"] == "hello"
    assert first.status_code == second.status_code == 202
    assert received_transcript["type"] == "transcript"
    assert received_transcript["session_id"] == session_id
    assert received_reply["type"] == "reply"
    assert received_reply["guard"] == "vetoed:3"
    assert received_reply["answer_status"] == "unknown"


async def test_manual_trigger_is_device_scoped_short_lived_and_consumed_once() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            armed = await http.post(
                f"/v1/device/{session_id}/manual-trigger",
                headers=auth(credential),
            )
            first = await http.post(
                f"/v1/device/{session_id}/manual-trigger/consume",
                headers=auth(),
            )
            second = await http.post(
                f"/v1/device/{session_id}/manual-trigger/consume",
                headers=auth(),
            )

    assert armed.status_code == 200
    assert first.json() == {"armed": True}
    assert second.json() == {"armed": False}


async def test_register_trigger_carries_a_label_and_is_consumed_once() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            armed = await http.post(
                f"/v1/device/{session_id}/register",
                json={"label": "keys"},
                headers=auth(credential),
            )
            first = await http.post(
                f"/v1/device/{session_id}/register/consume",
                headers=auth(),
            )
            second = await http.post(
                f"/v1/device/{session_id}/register/consume",
                headers=auth(),
            )

    assert armed.status_code == 200
    assert first.json() == {"armed": True, "label": "keys"}
    assert second.json() == {"armed": False, "label": None}


async def test_register_trigger_defaults_to_a_placeholder_label() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            await http.post(
                f"/v1/device/{session_id}/register",
                headers=auth(credential),
            )
            consumed = await http.post(
                f"/v1/device/{session_id}/register/consume",
                headers=auth(),
            )

    assert consumed.json() == {"armed": True, "label": None}


async def test_device_credential_cannot_publish_events_or_read_another_session() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            own, credential = await paired_session(http)
            other = (
                await http.post(
                    "/v1/sessions",
                    json={"device_id": "glasses-02"},
                    headers=auth(),
                )
            ).json()
            event = {
                "schema_version": "1.0",
                "type": "transcript",
                "text": "private transcript",
                "epoch_id": "TR_AUDIO_1",
                "pts_samples_start": 0,
                "samples": 16000,
                "sample_rate": 16000,
                "occurred_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            wrong_trigger = await http.post(
                f"/v1/device/{other['session_id']}/manual-trigger",
                headers=auth(credential),
            )
            publish = await http.post(
                f"/v1/device/{own['session_id']}/events",
                json=event,
                headers=auth(credential),
            )
            async with websockets.connect(
                f"ws://{host}/v1/device/{other['session_id']}/events",
                additional_headers=auth(credential),
            ) as socket:
                with pytest.raises(websockets.ConnectionClosed) as caught:
                    await asyncio.wait_for(socket.recv(), timeout=5)

    assert wrong_trigger.status_code == 401
    assert publish.status_code == 401
    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "unauthorized"


async def test_a_device_credential_is_refused_from_the_query_string() -> None:
    """A week-long secret must not travel where request lines are recorded.

    The operator token keeps the query-string concession because a browser
    cannot set headers on a WebSocket. The glasses can, and do.
    """
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            session_id = str(session["session_id"])
            async with websockets.connect(
                f"ws://{host}/v1/device/{session_id}/events?token={credential}"
            ) as refused:
                with pytest.raises(websockets.ConnectionClosed) as caught:
                    await asyncio.wait_for(refused.recv(), timeout=5)

            url = f"ws://{host}/v1/device/{session_id}/events"
            async with websockets.connect(url, additional_headers=auth(credential)) as accepted:
                hello = json.loads(await asyncio.wait_for(accepted.recv(), timeout=5))

    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "unauthorized"
    assert hello["type"] == "hello"


async def test_deleting_session_closes_its_event_stream() -> None:
    async with serve(create_app(settings())) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as http:
            session, credential = await paired_session(http)
            url = f"ws://{host}/v1/device/{session['session_id']}/events"
            async with websockets.connect(url, additional_headers=auth(credential)) as socket:
                await asyncio.wait_for(socket.recv(), timeout=5)
                deleted = await http.delete(f"/v1/sessions/{session['session_id']}", headers=auth())
                with pytest.raises(websockets.ConnectionClosed) as caught:
                    await asyncio.wait_for(socket.recv(), timeout=5)

    assert deleted.status_code == 204
    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "session_ended"
