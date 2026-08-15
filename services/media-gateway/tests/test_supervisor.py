"""Supervisor readiness and endpoint parsing.

Joining a real room needs a LiveKit server, so that lives in the opt-in
integration suite. What is testable here is the readiness contract, which is
the part most likely to be got wrong in a way nobody notices until a deploy
deadlocks.
"""

import pytest

from media_gateway.config import Settings
from media_gateway.domain.epoch import EpochRegistry
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.pipeline import MediaPipeline
from media_gateway.relay.hub import RelayHub
from media_gateway.transport.supervisor import SessionSupervisor, livekit_endpoint

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def a_supervisor(**overrides: object) -> SessionSupervisor:
    settings = Settings(environment="ci", **overrides)  # type: ignore[arg-type]
    hub = RelayHub(max_subscribers=4, audio_queue_chunks=8)
    pipeline = MediaPipeline(
        settings=settings,
        hub=hub,
        epochs=EpochRegistry(settings),
        metrics=MetricsRegistry(),
    )
    return SessionSupervisor(settings=settings, sink=pipeline)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("ws://127.0.0.1:7880", ("127.0.0.1", 7880)),
        ("wss://livekit.example.com", ("livekit.example.com", 443)),
        ("ws://livekit.example.com", ("livekit.example.com", 80)),
        ("http://10.0.0.5:7880", ("10.0.0.5", 7880)),
    ],
)
def test_endpoint_parsing(url: str, expected: tuple[str, int]) -> None:
    assert livekit_endpoint(url) == expected


async def test_scripted_mode_is_ready_without_livekit() -> None:
    """Scripted mode must not wait on a server it never uses."""
    supervisor = a_supervisor(media_source="scripted")

    await supervisor.start()
    try:
        assert supervisor.readiness() is None
    finally:
        await supervisor.stop()


async def test_livekit_mode_is_not_ready_before_the_first_probe() -> None:
    """Reporting ready before anything was checked would be a lie."""
    supervisor = a_supervisor(
        media_source="livekit",
        livekit_api_key="k",
        livekit_api_secret="a-secret-of-at-least-32-characters!!",
    )

    assert supervisor.readiness() == "probe pending"


async def test_an_unreachable_livekit_fails_readiness() -> None:
    supervisor = a_supervisor(
        media_source="livekit",
        # Nothing listens here; the probe must notice rather than hang.
        livekit_url="ws://127.0.0.1:1",
        livekit_connect_timeout_s=0.5,
        livekit_probe_interval_s=0.05,
        livekit_api_key="k",
        livekit_api_secret="a-secret-of-at-least-32-characters!!",
    )

    await supervisor.start()
    try:
        for _ in range(100):
            if supervisor.readiness() is not None and supervisor.readiness() != "probe pending":
                break
            await _tick()
        assert supervisor.readiness() == "livekit unreachable"
    finally:
        await supervisor.stop()


async def test_stopping_cancels_the_probe() -> None:
    supervisor = a_supervisor(
        media_source="livekit",
        livekit_url="ws://127.0.0.1:1",
        livekit_connect_timeout_s=0.2,
        livekit_probe_interval_s=0.05,
        livekit_api_key="k",
        livekit_api_secret="a-secret-of-at-least-32-characters!!",
    )
    await supervisor.start()

    await supervisor.stop()

    assert len(supervisor) == 0


async def _tick() -> None:
    import asyncio

    await asyncio.sleep(0.02)
