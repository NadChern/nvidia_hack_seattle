"""The application actually starts: the lifespan connects to a relay, the
background consumer runs, and /v1/status reports real configuration.

This drives the *real* lifespan -- the relay task, the reasoner (fixture), the
embedder, the whole Pipeline -- against a fake gateway server, not just the
router in isolation. This is the test that backs the claim "the service is
runnable today", not merely "the pieces are unit-tested".
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from visual_memory_media_contract.framing import encode_message
from visual_memory_media_contract.protocol import StreamHello
from visual_memory_media_contract.testing import replay_server

from vision_worker.config import Settings
from vision_worker.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _settings(url: str, **kwargs: object) -> Settings:
    return Settings(
        environment="ci",
        gateway_video_url=url,
        memory_base_url="http://127.0.0.1:1",  # never reached in these tests
        reason_kind="fixture",  # no Cosmos, no GPU
        **kwargs,  # type: ignore[arg-type]
    )


async def get(app, path: str) -> tuple[int, dict[str, object]]:  # type: ignore[no-untyped-def]
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(path)
    return response.status_code, response.json()


async def test_the_service_starts_and_reports_ready() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        app = create_app(_settings(url))
        status_code, body = await get(app, "/health/ready")

    assert status_code == 200
    assert body["status"] == "ready"


async def test_status_reports_configuration_and_the_reasoner() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        app = create_app(_settings(url, detection_labels=("keys", "wallet")))
        status_code, body = await get(app, "/v1/status")

    assert status_code == 200
    assert body["ready"] is True
    assert body["config"]["reason_kind"] == "fixture"
    assert body["config"]["registration_labels"] == ["keys", "wallet"]
    # The reasoner window and the identity gate are reported so an evaluation
    # run can cite what was actually in effect, not what was merely configured.
    assert body["reasoner"]["window_seconds"] > 0
    assert body["reasoner"]["interval_seconds"] > 0
    assert body["reasoner"]["promote_motion_events"] is False
    assert 0.0 <= body["identity"]["min_cosine"] <= 1.0
    assert body["identity"]["gallery"]["gallery_objects"] == 0
    assert body["metrics"]["frames_processed"] == 0
    assert body["analysis"]["dropped"] == 0


async def test_clip_fps_defaults_to_the_source_rate() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        app = create_app(_settings(url, source_fps=3.0))
        status_code, body = await get(app, "/v1/status")

    assert status_code == 200
    assert body["config"]["clip_fps"] == 3.0


async def test_events_reports_an_empty_list_with_no_activity_yet() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        app = create_app(_settings(url))
        status_code, body = await get(app, "/v1/events")

    assert status_code == 200
    assert body["events"] == []


async def test_liveness_does_not_depend_on_the_relay() -> None:
    """A relay outage must not look like a dead process."""
    app = create_app(_settings("ws://127.0.0.1:1/v1/stream/video"))
    status_code, body = await get(app, "/health/live")

    assert status_code == 200
    assert body["status"] == "ok"
