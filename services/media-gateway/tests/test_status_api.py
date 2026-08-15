"""The status endpoint.

A dashboard reads this, so the shape is a contract. The dimension histogram is
the field that matters most: it is what distinguishes "no camera connected"
from "the camera is sending a size the guard rejects".
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.main import create_app
from media_gateway.transport.source import RawVideoFrame, utcnow

pytestmark = pytest.mark.anyio

TOKEN = "an-internal-token-of-at-least-32-chars"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def an_app(**overrides: object) -> FastAPI:
    base: dict[str, object] = {
        "environment": "ci",
        "media_source": "scripted",
        "scripted_frame_interval_s": 30.0,
    }
    base.update(overrides)
    return create_app(Settings(**base))  # type: ignore[arg-type]


async def status_of(app: FastAPI, **kwargs: object) -> dict[str, object]:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get("/v1/status", **kwargs)  # type: ignore[arg-type]
            return {"status_code": response.status_code, "body": response.json()}


async def test_status_reports_service_and_config() -> None:
    result = await status_of(an_app())
    body = result["body"]

    assert result["status_code"] == 200
    assert body["service"] == "media-gateway"  # type: ignore[index]
    assert body["media_source"] == "scripted"  # type: ignore[index]
    assert body["config"]["sample_fps"] == 8.0  # type: ignore[index]
    assert body["config"]["expected_video_size"] == "320x180"  # type: ignore[index]


async def test_the_shape_is_stable() -> None:
    """A dashboard depends on these keys existing."""
    body = (await status_of(an_app()))["body"]

    assert set(body) >= {  # type: ignore[arg-type]
        "service",
        "version",
        "environment",
        "media_source",
        "ready",
        "not_ready_reason",
        "uptime_s",
        "config",
        "sessions",
        "epochs",
        "relay",
        "metrics",
    }


async def test_a_scripted_gateway_is_ready_and_shows_its_synthetic_epochs() -> None:
    """Scripted mode starts producing at once, so epochs exist immediately.

    Sessions stay empty because the scripted source never calls the session
    API; the session registry only tracks sessions created through it.
    """
    body = (await status_of(an_app()))["body"]

    assert body["ready"] is True  # type: ignore[index]
    assert body["sessions"] == []  # type: ignore[index]
    epochs = body["epochs"]  # type: ignore[index]
    assert epochs, "the scripted source should have opened an epoch"
    assert {epoch["stream_kind"] for epoch in epochs} == {"video", "audio"}


async def test_privacy_knob_is_reported() -> None:
    """The privacy checklist needs something concrete to point at."""
    body = (await status_of(an_app()))["body"]

    assert body["config"]["raw_buffer_seconds"] == 0  # type: ignore[index]


async def test_epochs_expose_the_dimension_histogram() -> None:
    """The field that explains a silent pipeline."""
    app = an_app()
    async with app.router.lifespan_context(app):
        pipeline = app.state.pipeline
        pipeline.session_started(session_id="sess_1", device_id="glasses-01")
        pipeline.epoch_started(
            session_id="sess_1",
            stream_kind="video",
            track_sid="TR_VCaaa",
            participant_identity="glasses-01",
        )
        # One accepted size and one the guard rejects.
        for width, height in ((320, 180), (320, 180), (8, 8)):
            pipeline.video_frame(
                session_id="sess_1",
                frame=RawVideoFrame(
                    width=width, height=height, rgba=bytes(width * height * 4), captured_at=utcnow()
                ),
            )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            body = (await http.get("/v1/status")).json()

    epoch = body["epochs"][0]
    assert epoch["epoch_id"] == "TR_VCaaa"
    assert epoch["guard"]["dimensions"] == {"320x180": 2, "8x8": 1}
    assert epoch["guard"]["admitted"] == 2
    assert epoch["guard"]["rejected"] == 1


async def test_relay_subscribers_are_reported() -> None:
    app = an_app()
    async with app.router.lifespan_context(app):
        app.state.hub.subscribe(stream_kind="video", encoding="jpeg")
        app.state.hub.subscribe(stream_kind="audio")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            body = (await http.get("/v1/status")).json()

    assert body["relay"]["subscribers"] == 2
    assert len(body["relay"]["video"]) == 1
    assert body["relay"]["video"][0]["encoding"] == "jpeg"


async def test_status_requires_the_internal_token_when_configured() -> None:
    result = await status_of(an_app(internal_api_token=TOKEN))

    assert result["status_code"] == 401


async def test_status_accepts_the_configured_token() -> None:
    result = await status_of(
        an_app(internal_api_token=TOKEN), headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert result["status_code"] == 200


async def test_no_secret_appears_in_the_response() -> None:
    """Status is the surface most likely to be pasted into a chat."""
    secret = "a-livekit-secret-of-at-least-32-chars"
    result = await status_of(
        an_app(livekit_api_key="k", livekit_api_secret=secret, internal_api_token=TOKEN),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    rendered = str(result["body"])
    assert secret not in rendered
    assert TOKEN not in rendered
