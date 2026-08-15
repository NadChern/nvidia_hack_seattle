"""The application actually starts: the lifespan connects to a relay, the
background consumer runs, and /v1/status reports real configuration.

Unlike test_health.py, this drives the *real* lifespan -- the background
relay task, the fixture detector, the whole Pipeline -- against a fake
gateway server, not just the router in isolation. This is the test that
backs the claim "the service is runnable today", not merely "the pieces are
unit-tested".
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from visual_memory_media_contract.framing import encode_message
from visual_memory_media_contract.protocol import StreamHello
from visual_memory_media_contract.testing import replay_server

from vision_worker.config import Settings
from vision_worker.main import build_stability_config, create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def get(app, path: str) -> tuple[int, dict[str, object]]:  # type: ignore[no-untyped-def]
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(path)
    return response.status_code, response.json()


async def test_the_service_starts_and_reports_ready() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        settings = Settings(
            environment="ci",
            gateway_video_url=url,
            memory_base_url="http://127.0.0.1:1",  # never called -- see below
        )
        app = create_app(settings)

        status_code, body = await get(app, "/health/ready")

    assert status_code == 200
    assert body["status"] == "ready"


async def test_status_reports_configuration_and_thresholds() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        settings = Settings(
            environment="ci",
            gateway_video_url=url,
            memory_base_url="http://127.0.0.1:1",
            detection_labels="keys,wallet",  # type: ignore[arg-type]
        )
        app = create_app(settings)

        status_code, body = await get(app, "/v1/status")

    assert status_code == 200
    assert body["ready"] is True
    assert body["config"]["detector_kind"] == "fixture"
    assert body["config"]["detection_labels"] == ["keys", "wallet"]
    # Real thresholds, not just settings echoed back -- the plan's
    # requirement that an evaluation run can cite what was actually in
    # effect, not what was merely configured.
    assert body["stability_thresholds"]["dwell_frames"] > 0
    assert body["verifier_thresholds"]["min_confidence"] >= 0.0
    assert body["metrics"]["frames_processed"] == 0

    # Both halves of the frame-rate assumption. A frame count alone does not
    # say what duration it stands for, and the configured rate alone does not
    # say whether the relay agrees -- only the pair is citable.
    assert body["frame_rate"]["configured_fps"] == settings.source_fps
    assert body["frame_rate"]["observed_fps"] is None  # no frames yet
    assert body["stability_durations_s"]["dwell"] == settings.dwell_seconds
    assert body["stability_thresholds"]["dwell_frames"] == round(
        settings.dwell_seconds * settings.source_fps
    )


async def test_clip_fps_defaults_to_the_source_rate() -> None:
    """An evidence clip should play at real speed. Encoding a 2fps window at
    24 makes a 12x timelapse of an object being set down."""
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        settings = Settings(
            environment="ci",
            gateway_video_url=url,
            memory_base_url="http://127.0.0.1:1",
            source_fps=3.0,
        )
        assert settings.clip_fps is None
        app = create_app(settings)

        status_code, body = await get(app, "/v1/status")

    assert status_code == 200
    assert body["config"]["clip_fps"] == 3.0


async def test_events_reports_an_empty_list_with_no_activity_yet() -> None:
    hello = encode_message(StreamHello(gateway_version="test-1", stream_kind="video"))

    async with replay_server([hello], close_after=False) as url:
        settings = Settings(
            environment="ci",
            gateway_video_url=url,
            memory_base_url="http://127.0.0.1:1",
        )
        app = create_app(settings)

        status_code, body = await get(app, "/v1/events")

    assert status_code == 200
    assert body["events"] == []


async def test_liveness_does_not_depend_on_the_relay() -> None:
    """A relay outage must not look like a dead process."""
    settings = Settings(
        environment="ci",
        gateway_video_url="ws://127.0.0.1:1/v1/stream/video",  # nothing listens
        memory_base_url="http://127.0.0.1:1",
    )
    app = create_app(settings)

    status_code, body = await get(app, "/health/live")

    assert status_code == 200
    assert body["status"] == "ok"


async def test_build_stability_config_warns_when_a_duration_rounds_to_one_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At a source rate of 2fps a 0.5s dwell rounds to a
    single frame -- one stable sighting confirming a placement. Legitimate to
    run with, never something to arrive at by accident."""
    settings = Settings(
        environment="ci",
        memory_base_url="http://127.0.0.1:1",
        source_fps=2.0,
        dwell_seconds=0.5,
    )

    with caplog.at_level("WARNING", logger="vision_worker.main"):
        config = build_stability_config(settings)

    assert config.dwell_frames == 1
    assert any("rounds to a single frame" in record.message for record in caplog.records)


async def test_build_stability_config_is_quiet_when_the_rate_supports_the_durations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        environment="ci",
        memory_base_url="http://127.0.0.1:1",
        source_fps=24.0,
    )

    with caplog.at_level("WARNING", logger="vision_worker.main"):
        config = build_stability_config(settings)

    assert config.dwell_frames == 12
    assert config.passive_confirmation_frames == 90
    assert not [record for record in caplog.records if record.levelname == "WARNING"]
