"""`WS /v1/overlay`: what a console connects to.

Split deliberately between two styles. Auth, capacity and the greeting go
through `TestClient`, so real Starlette WebSocket handling and real close codes
are exercised. Delivery is driven against the route coroutine directly, because
`TestClient` runs the app on its own thread and publishing into the hub from the
test thread would race the `asyncio.Event` a waiting subscriber sits on -- a
flake dressed up as coverage. `tests/test_overlay_hub.py` covers fan-out in a
real event loop.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect, WebSocketState
from visual_memory_vision_contract.protocol import (
    SCHEMA_VERSION,
    BoundingBox,
    OverlayFrame,
    OverlayTrack,
)

from vision_worker.api import overlay
from vision_worker.config import Settings
from vision_worker.overlay.hub import OverlayHub

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 8, 7, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def an_overlay(sequence: int = 1) -> OverlayFrame:
    return OverlayFrame(
        session_id="sess_1",
        media_epoch_id="TR_VCaaa",
        sequence=sequence,
        captured_at=T0,
        relayed_at=T0,
        emitted_at=T0 + dt.timedelta(milliseconds=125),
        width=1280,
        height=720,
        tracks=(
            OverlayTrack(
                track_id="track-3",
                label="a set of keys",
                confidence=0.42,
                box=BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4),
                motion_state="settling",
                depth_m=None,
            ),
        ),
        pipeline_latency_ms=125.0,
    )


def an_app(*, token: str | None = None, max_viewers: int = 4) -> tuple[FastAPI, OverlayHub]:
    """A minimal app carrying only what the overlay route reads.

    Not `vision_worker.main.app`: that one's lifespan builds a detector and a
    relay client, none of which this endpoint touches.
    """
    app = FastAPI()
    app.include_router(overlay.router)
    hub = OverlayHub(max_subscribers=max_viewers)
    app.state.settings = Settings(
        internal_api_token=SecretStr(token) if token else None,
        detector_kind="fixture",
        source_fps=8.0,
    )
    app.state.overlay_hub = hub
    return app, hub


# --- Greeting ----------------------------------------------------------------


def test_a_viewer_is_told_what_it_is_watching_before_any_frames() -> None:
    """A pipeline receiving no video would otherwise leave the socket silent
    and indistinguishable from one that is broken."""
    app, _ = an_app()
    with TestClient(app).websocket_connect("/v1/overlay") as ws:
        hello = ws.receive_json()

    assert hello["type"] == "overlay_hello"
    assert hello["schema_version"] == SCHEMA_VERSION
    assert hello["detector_kind"] == "fixture"
    assert hello["source_fps"] == 8.0
    assert hello["session_id"] is None


def test_a_viewer_can_scope_the_stream_to_one_gateway_session() -> None:
    app, _ = an_app()
    with TestClient(app).websocket_connect("/v1/overlay?session_id=sess_2") as ws:
        hello = ws.receive_json()

    assert hello["session_id"] == "sess_2"


# --- Auth --------------------------------------------------------------------


def test_no_token_configured_leaves_the_stream_open() -> None:
    """The local development default, matching every other surface."""
    app, _ = an_app(token=None)
    with TestClient(app).websocket_connect("/v1/overlay") as ws:
        assert ws.receive_json()["type"] == "overlay_hello"


def test_a_service_may_authorize_with_a_bearer_header() -> None:
    app, _ = an_app(token="secret")
    with TestClient(app).websocket_connect(
        "/v1/overlay", headers={"authorization": "Bearer secret"}
    ) as ws:
        assert ws.receive_json()["type"] == "overlay_hello"


def test_a_browser_may_authorize_with_a_query_parameter() -> None:
    """A browser cannot set headers on a WebSocket handshake. Without this the
    console would work in development and be unreachable in deploy, which is
    exactly where a demo runs."""
    app, _ = an_app(token="secret")
    with TestClient(app).websocket_connect("/v1/overlay?token=secret") as ws:
        assert ws.receive_json()["type"] == "overlay_hello"


@pytest.mark.parametrize("path", ["/v1/overlay", "/v1/overlay?token=wrong"])
def test_a_missing_or_wrong_token_is_closed_with_a_reason(path: str) -> None:
    """Accepted and then closed, rather than rejected at the handshake -- a
    bare handshake failure gives the client nothing to act on."""
    app, _ = an_app(token="secret")
    client = TestClient(app)
    with client.websocket_connect(path) as ws, pytest.raises(WebSocketDisconnect) as caught:
        ws.receive_json()

    assert caught.value.code == overlay.CLOSE_UNAUTHORIZED
    assert caught.value.reason == "unauthorized"


def test_an_unauthorized_viewer_never_takes_a_slot() -> None:
    app, hub = an_app(token="secret")
    client = TestClient(app)
    with client.websocket_connect("/v1/overlay") as ws, pytest.raises(WebSocketDisconnect):
        ws.receive_json()

    assert hub.subscriber_count == 0


# --- Capacity ----------------------------------------------------------------


def test_a_viewer_beyond_the_limit_is_told_why() -> None:
    app, _ = an_app(max_viewers=1)
    client = TestClient(app)
    with client.websocket_connect("/v1/overlay") as first:
        assert first.receive_json()["type"] == "overlay_hello"
        with (
            client.websocket_connect("/v1/overlay") as second,
            pytest.raises(WebSocketDisconnect) as caught,
        ):
            second.receive_json()

    assert caught.value.code == overlay.CLOSE_CAPACITY
    assert caught.value.reason == "too_many_overlay_viewers"


def test_a_departed_viewer_frees_its_slot() -> None:
    app, hub = an_app(max_viewers=1)
    client = TestClient(app)
    with client.websocket_connect("/v1/overlay") as ws:
        ws.receive_json()
    assert hub.subscriber_count == 0

    with client.websocket_connect("/v1/overlay") as ws:
        assert ws.receive_json()["type"] == "overlay_hello"


# --- Delivery ----------------------------------------------------------------


class _FakeWebSocket:
    """Enough of a WebSocket to drive the route coroutine deterministically.

    Models `application_state` and `client_state` as two independent values,
    and refuses a send after a close, because that is exactly what Starlette
    does -- and a fake that tracked only one of them hid a real bug here once.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.sent: list[Any] = []
        self.application_state = WebSocketState.CONNECTED
        #: Deliberately left CONNECTED after we close. Only the peer's own
        #: close frame moves this, so on shutdown it stays CONNECTED while
        #: `application_state` does not -- the discrepancy the route must
        #: respect.
        self.client_state = WebSocketState.CONNECTED
        self.closed_with: int | None = None

    def _guard(self) -> None:
        if self.application_state is not WebSocketState.CONNECTED:
            raise RuntimeError('Cannot call "send" once a close message has been sent.')

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: Any) -> None:
        self._guard()
        self.sent.append(payload)

    async def send_text(self, payload: str) -> None:
        self._guard()
        self.sent.append(json.loads(payload))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del reason
        self._guard()
        self.closed_with = code
        self.application_state = WebSocketState.DISCONNECTED


async def test_a_published_overlay_reaches_the_viewer_as_contract_json() -> None:
    """The shape the console parses. Round-tripped back through the contract
    model so a field renamed on one side cannot pass unnoticed on the other."""
    app, hub = an_app()
    websocket = _FakeWebSocket(app)

    # Queue the frame first, then close, so the route sends exactly one overlay
    # and its loop terminates without needing to be cancelled.
    original = an_overlay(sequence=11)

    async def publish_then_close() -> None:
        hub.publish(original)
        hub.close()

    task = asyncio.ensure_future(overlay.overlay(websocket))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    await publish_then_close()
    await asyncio.wait_for(task, timeout=1)

    hello, payload = websocket.sent
    assert hello["type"] == "overlay_hello"

    restored = OverlayFrame.model_validate(payload)
    assert restored == original
    assert restored.tracks[0].motion_state == "settling"
    assert restored.pipeline_latency_ms == 125.0


async def test_the_viewer_is_released_when_the_stream_ends() -> None:
    app, hub = an_app()
    websocket = _FakeWebSocket(app)

    task = asyncio.ensure_future(overlay.overlay(websocket))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert hub.subscriber_count == 1

    hub.close()
    await asyncio.wait_for(task, timeout=1)

    assert hub.subscriber_count == 0


async def test_the_socket_is_not_closed_twice_on_shutdown() -> None:
    """Regression: the cleanup guarded on `client_state`, which only the peer's
    close frame moves. On shutdown the close has already been sent, so closing
    again raised `Cannot call "send" once a close message has been sent` out of
    the ASGI app -- harmless to the stream, but a traceback in the logs every
    time the server stopped.
    """
    app, hub = an_app()
    websocket = _FakeWebSocket(app)

    task = asyncio.ensure_future(overlay.overlay(websocket))  # type: ignore[arg-type]
    await asyncio.sleep(0)

    # What shutdown looks like: the app has already sent its close, while the
    # peer has not yet answered, so `client_state` still reads CONNECTED.
    await websocket.close(code=overlay.CLOSE_GOING_AWAY)
    hub.close()

    await asyncio.wait_for(task, timeout=1)

    assert websocket.client_state is WebSocketState.CONNECTED, "the peer never replied"
    assert hub.subscriber_count == 0, "the viewer is still released"
