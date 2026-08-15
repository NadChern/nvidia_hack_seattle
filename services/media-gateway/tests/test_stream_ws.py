"""The relay endpoints, exercised through the real ASGI stack.

This is what a Vision or Speech developer will actually run: the gateway
binary, driven by the scripted source, with no LiveKit and no hardware.
"""

import asyncio

import pytest
import websockets
from visual_memory_media_contract import MediaClient, MediaClientError
from visual_memory_media_contract.protocol import (
    AudioChunk,
    EpochStarted,
    Keepalive,
    RelayMessage,
    StreamHello,
    VideoFrame,
)

from media_gateway.config import Settings
from media_gateway.main import create_app
from tests.serving import serve

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def fast_settings(**overrides: object) -> Settings:
    """A gateway that produces media quickly enough for a test to observe."""
    base: dict[str, object] = {
        "environment": "ci",
        "media_source": "scripted",
        "scripted_frame_interval_s": 0.01,
        "sample_fps": 50.0,
        "ws_keepalive_s": 10.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def take(url: str, count: int, *, token: str | None = None) -> list[RelayMessage]:
    """Read `count` messages from a stream, then disconnect."""
    received: list[RelayMessage] = []
    async with MediaClient(url, token=token, reconnect=False) as client:
        async for message in client:
            received.append(message)
            if len(received) >= count:
                break
    return received


async def take_until(url: str, predicate: object, *, limit: int = 200) -> list[RelayMessage]:
    """Read until `predicate` matches a message, or `limit` messages pass."""
    received: list[RelayMessage] = []
    async with MediaClient(url, reconnect=False) as client:
        async for message in client:
            received.append(message)
            if isinstance(message, predicate) or len(received) >= limit:  # type: ignore[arg-type]
                break
    return received


async def test_video_stream_opens_with_hello() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take(f"ws://{host}/v1/stream/video", 1)

    hello = messages[0]
    assert isinstance(hello, StreamHello)
    assert hello.stream_kind == "video"
    assert hello.encoding == "jpeg"


async def test_audio_stream_opens_with_hello() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take(f"ws://{host}/v1/stream/audio", 1)

    hello = messages[0]
    assert isinstance(hello, StreamHello)
    assert hello.stream_kind == "audio"


async def test_video_frames_reach_a_subscriber() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take_until(f"ws://{host}/v1/stream/video", VideoFrame)

    frames = [m for m in messages if isinstance(m, VideoFrame)]

    assert frames, "expected the scripted source to produce a frame"
    frame = frames[0]
    assert frame.width == 320
    assert frame.height == 180
    assert frame.rgb.shape == (180, 320, 3)


async def test_audio_chunks_reach_a_subscriber() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take_until(f"ws://{host}/v1/stream/audio", AudioChunk)

    chunks = [m for m in messages if isinstance(m, AudioChunk)]

    assert chunks
    assert chunks[0].sample_rate == 48_000
    assert chunks[0].pcm.shape[1] == 1


async def test_epoch_precedes_the_frames_it_governs() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take_until(f"ws://{host}/v1/stream/video", VideoFrame)

    seen: set[str] = set()
    for message in messages:
        if isinstance(message, EpochStarted):
            seen.add(message.epoch_id)
        if isinstance(message, VideoFrame):
            assert message.epoch_id in seen


async def test_a_late_subscriber_is_told_which_epochs_are_running() -> None:
    """Joining mid-epoch must still yield something to reset on."""
    app = create_app(fast_settings())
    async with serve(app) as host:
        url = f"ws://{host}/v1/stream/video"
        # Let the first subscriber establish an epoch.
        await take_until(url, VideoFrame)

        messages = await take(url, 2)

    assert isinstance(messages[0], StreamHello)
    assert isinstance(messages[1], EpochStarted)
    assert messages[0].active_epochs, "hello should list the running epoch"


async def test_rgba_raw_is_available_per_connection() -> None:
    async with serve(create_app(fast_settings())) as host:
        messages = await take_until(f"ws://{host}/v1/stream/video?encoding=rgba_raw", VideoFrame)

    frames = [m for m in messages if isinstance(m, VideoFrame)]

    assert frames
    assert frames[0].encoding == "rgba_raw"
    assert frames[0].rgba.shape == (180, 320, 4)


async def test_an_unknown_encoding_is_refused() -> None:
    """An unsupported encoding fails the handshake rather than half-working."""
    async with serve(create_app(fast_settings())) as host:
        with pytest.raises(websockets.InvalidStatus) as caught:
            await websockets.connect(f"ws://{host}/v1/stream/video?encoding=heif")

    assert caught.value.response.status_code == 403


async def test_hello_names_the_running_session() -> None:
    """A subscriber that missed `session_started` still learns the session.

    The scripted source announces a session once at startup, so anyone
    connecting later relies on the hello rather than on having been present.
    """
    app = create_app(fast_settings())
    async with serve(app) as host:
        url = f"ws://{host}/v1/stream/video"
        await take_until(url, VideoFrame)

        messages = await take(url, 1)

    hello = messages[0]
    assert isinstance(hello, StreamHello)
    assert hello.active_sessions == ["sess_scripted"]


async def test_idle_streams_send_keepalives() -> None:
    """A consumer must be able to tell "no publisher" from "dead socket"."""
    settings = fast_settings(scripted_frame_interval_s=30.0, ws_keepalive_s=0.1)
    async with serve(create_app(settings)) as host:
        messages = await take_until(f"ws://{host}/v1/stream/video", Keepalive, limit=20)

    assert any(isinstance(m, Keepalive) for m in messages)


async def test_a_missing_token_is_refused_when_one_is_configured() -> None:
    settings = fast_settings(internal_api_token="a-token-of-at-least-32-characters!!")
    async with serve(create_app(settings)) as host:
        async with websockets.connect(f"ws://{host}/v1/stream/video") as socket:
            with pytest.raises(websockets.ConnectionClosed) as caught:
                await asyncio.wait_for(socket.recv(), timeout=5)

    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "unauthorized"


async def test_a_wrong_token_is_refused_with_an_explanation() -> None:
    """A refusal must not look like "no publisher yet" to the consumer."""
    settings = fast_settings(internal_api_token="a-token-of-at-least-32-characters!!")
    async with serve(create_app(settings)) as host:
        with pytest.raises(MediaClientError, match="unauthorized"):
            await take(f"ws://{host}/v1/stream/video", 1, token="not-the-token")


async def test_the_configured_token_is_accepted() -> None:
    token = "a-token-of-at-least-32-characters!!"
    settings = fast_settings(internal_api_token=token)
    async with serve(create_app(settings)) as host:
        messages = await take(f"ws://{host}/v1/stream/video", 1, token=token)

    assert isinstance(messages[0], StreamHello)


async def test_the_subscriber_limit_is_enforced() -> None:
    settings = fast_settings(ws_max_subscribers=1)
    async with serve(create_app(settings)) as host:
        url = f"ws://{host}/v1/stream/video"
        async with websockets.connect(url) as first:
            await asyncio.wait_for(first.recv(), timeout=5)

            async with websockets.connect(url) as second:
                with pytest.raises(websockets.ConnectionClosed) as caught:
                    await asyncio.wait_for(second.recv(), timeout=5)

    assert caught.value.rcvd is not None
    assert caught.value.rcvd.reason == "capacity_exhausted"


async def test_disconnecting_releases_the_subscriber_slot() -> None:
    settings = fast_settings(ws_max_subscribers=1)
    app = create_app(settings)
    async with serve(app) as host:
        url = f"ws://{host}/v1/stream/video"
        await take(url, 1)
        # The slot must come back, or one dropped consumer would wedge the
        # gateway until restart.
        for _ in range(50):
            if len(app.state.hub) == 0:
                break
            await asyncio.sleep(0.02)

        messages = await take(url, 1)

    assert isinstance(messages[0], StreamHello)


async def test_video_and_audio_can_be_consumed_at_once() -> None:
    async with serve(create_app(fast_settings())) as host:
        video, audio = await asyncio.gather(
            take_until(f"ws://{host}/v1/stream/video", VideoFrame),
            take_until(f"ws://{host}/v1/stream/audio", AudioChunk),
        )

    assert any(isinstance(m, VideoFrame) for m in video)
    assert any(isinstance(m, AudioChunk) for m in audio)


async def test_an_idle_stream_notices_a_departed_consumer_at_once() -> None:
    """A one-way relay must still read the socket.

    With nothing to send, a disconnect would otherwise surface only when the
    next keepalive fails -- up to `ws_keepalive_s` later. That is exactly the
    state a gateway sits in with no glasses connected, and it would leave a
    dead consumer holding one of the hub's slots.
    """
    settings = fast_settings(
        # Both timers set well beyond the assertion window, so passing can only
        # mean the close was observed rather than waited out.
        ws_keepalive_s=30.0,
        scripted_frame_interval_s=30.0,
        sample_fps=0.1,
    )
    app = create_app(settings)
    async with serve(app) as host:
        async with MediaClient(f"ws://{host}/v1/stream/video", reconnect=False) as client:
            async for _ in client:
                break
            assert len(app.state.hub) == 1

        started = asyncio.get_running_loop().time()
        for _ in range(100):
            if len(app.state.hub) == 0:
                break
            await asyncio.sleep(0.02)
        elapsed = asyncio.get_running_loop().time() - started

    assert len(app.state.hub) == 0
    assert elapsed < 2.0, "the slot came back only when a keepalive failed"
