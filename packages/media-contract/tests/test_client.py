"""MediaClient against a real WebSocket serving the recorded fixtures.

This is the harness a consumer copies: no gateway, no LiveKit, no hardware.
"""

import numpy as np
import pytest

from visual_memory_media_contract.client import MediaClient, ReconnectPolicy
from visual_memory_media_contract.protocol import (
    AudioChunk,
    EpochEnded,
    EpochStarted,
    RelayMessage,
    SessionEnded,
    StreamHello,
    VideoFrame,
)
from visual_memory_media_contract.testing import (
    assert_matches_fixture,
    flaky_replay_server,
    replay_server,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def drain(url: str) -> list[RelayMessage]:
    """Read a stream to completion without reconnecting."""
    received: list[RelayMessage] = []
    async with MediaClient(url, reconnect=False) as client:
        async for message in client:
            received.append(message)
    return received


async def test_video_fixture_replays_end_to_end() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    assert_matches_fixture(received, "video_session_basic")


async def test_audio_fixture_replays_end_to_end() -> None:
    async with replay_server("audio_session_basic") as url:
        received = await drain(url)

    assert_matches_fixture(received, "audio_session_basic")


async def test_stream_opens_with_hello() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    assert isinstance(received[0], StreamHello)
    assert received[0].stream_kind == "video"


async def test_rejoin_produces_a_new_epoch_with_the_same_identity() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    starts = [m for m in received if isinstance(m, EpochStarted)]

    assert len(starts) == 2
    # The whole point: identity is stable across a rejoin, the track SID is not.
    assert starts[0].participant_identity == starts[1].participant_identity
    assert starts[0].epoch_id != starts[1].epoch_id
    assert all(start.epoch_id == start.track_sid for start in starts)


async def test_sequence_restarts_at_zero_in_each_epoch() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    by_epoch: dict[str, list[int]] = {}
    for message in received:
        if isinstance(message, VideoFrame):
            by_epoch.setdefault(message.epoch_id, []).append(message.sequence)

    assert len(by_epoch) == 2
    for sequences in by_epoch.values():
        assert sequences == list(range(len(sequences)))


async def test_every_epoch_start_is_matched_by_an_end() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    started = [m.epoch_id for m in received if isinstance(m, EpochStarted)]
    ended = [m.epoch_id for m in received if isinstance(m, EpochEnded)]

    assert started == ended
    assert isinstance(received[-1], SessionEnded)


async def test_frames_decode_to_declared_dimensions() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    frames = [m for m in received if isinstance(m, VideoFrame)]

    assert frames
    for frame in frames:
        assert frame.rgb.shape == (frame.height, frame.width, 3)
        assert frame.rgb.dtype == np.uint8


async def test_dropped_frames_are_reported_not_hidden() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    dropped = [m.dropped_since_previous for m in received if isinstance(m, VideoFrame)]

    assert sum(dropped) > 0, "fixture should exercise the latest-wins slot"


async def test_audio_gap_is_detectable_from_pts_not_sequence() -> None:
    async with replay_server("audio_session_basic") as url:
        received = await drain(url)

    chunks = [m for m in received if isinstance(m, AudioChunk)]
    sequences = [c.sequence for c in chunks]

    # Sequence is contiguous, so counting messages would miss the loss.
    assert sequences == list(range(len(chunks)))

    gaps = [
        (later.pts_samples - earlier.pts_samples - earlier.samples)
        for earlier, later in zip(chunks[:-1], chunks[1:], strict=True)
    ]
    assert sum(1 for gap in gaps if gap > 0) == 1
    assert max(gaps) == 4800 * 5


async def test_audio_chunks_decode_to_declared_sample_counts() -> None:
    async with replay_server("audio_session_basic") as url:
        received = await drain(url)

    chunks = [m for m in received if isinstance(m, AudioChunk)]

    assert chunks
    for chunk in chunks:
        assert chunk.pcm.shape == (chunk.samples, chunk.channels)
        assert chunk.pcm.dtype == np.int16


async def test_client_reconnects_after_a_mid_stream_drop() -> None:
    policy = ReconnectPolicy(initial_seconds=0.01, max_seconds=0.05, jitter=0.0)
    received: list[RelayMessage] = []

    async with flaky_replay_server("video_session_basic", drop_after=5) as url:
        client = MediaClient(url, policy=policy)
        async for message in client:
            received.append(message)
            if isinstance(message, SessionEnded):
                await client.aclose()
                break

    # The first connection was cut after 5 frames; the retry replayed the whole
    # fixture, so a consumer sees a second hello and re-sees the first epoch.
    hellos = [m for m in received if isinstance(m, StreamHello)]
    assert len(hellos) == 2
    assert len(received) > 5


async def test_assert_matches_fixture_reports_a_mismatch() -> None:
    async with replay_server("video_session_basic") as url:
        received = await drain(url)

    with pytest.raises(AssertionError, match="messages, observed"):
        assert_matches_fixture(received[:-1], "video_session_basic")


def test_reconnect_policy_backs_off_and_caps() -> None:
    policy = ReconnectPolicy(initial_seconds=1.0, max_seconds=8.0, multiplier=2.0, jitter=0.0)

    assert [policy.delay_for(n) for n in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_reconnect_policy_jitter_stays_in_range() -> None:
    policy = ReconnectPolicy(initial_seconds=1.0, max_seconds=1.0, jitter=0.25)

    delays = [policy.delay_for(1) for _ in range(200)]

    assert all(0.75 <= delay <= 1.25 for delay in delays)
    assert len(set(delays)) > 1, "jitter should spread retries"
