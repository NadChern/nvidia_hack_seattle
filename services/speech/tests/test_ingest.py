"""Tests for `speech.ingest`.

Everything here runs with no gateway, no LiveKit, and no model, per
role-prompts/Speech.md's no-hardware start path.

`audio_session_basic` has exactly one epoch and one internal `pts_samples`
gap (confirmed by decoding the fixture directly: one `epoch_started`, 30
`audio_chunk` messages, one `epoch_ended`), so it cannot on its own exercise
a genuine second, distinct epoch. `flaky_replay_server` *can* legitimately
exercise `MediaClient`'s real reconnect path -- it severs the connection and
replays the whole fixture again on the next attempt -- so the reconnect test
below uses that rather than faking coverage.

NOT tested here, and not fakeable with what currently exists: the `1011
audio_backpressure` close. Neither the fixture/replay tooling nor
`MediaClient`'s public API can produce or expose that specific close code to
a consumer -- see the module docstring in `speech/ingest.py` for the full
explanation and the open question raised for Alex.
"""

import pytest
from visual_memory_media_contract.client import ReconnectPolicy
from visual_memory_media_contract.testing import flaky_replay_server, replay_server

from speech.ingest import AudioSegment, ingest_segments


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _collect(
    url: str,
    *,
    reconnect: bool = True,
    policy: ReconnectPolicy | None = None,
) -> list[AudioSegment]:
    return [segment async for segment in ingest_segments(url, reconnect=reconnect, policy=policy)]


@pytest.mark.anyio
async def test_ingest_splits_into_two_contiguous_segments_at_the_gap() -> None:
    async with replay_server("audio_session_basic") as url:
        segments = await _collect(url, reconnect=False)

    assert len(segments) == 2

    first, second = segments
    assert first.pts_samples_start == 0
    assert first.samples == 48_000  # 10 chunks * 4800 samples, before the gap
    assert second.pts_samples_start == 72_000  # 48_000 + the 24_000-sample (500ms) gap
    assert second.samples == 96_000  # 20 chunks * 4800 samples, after the gap

    # Each segment's PCM is exactly its sample count worth of s16le mono.
    assert len(first.pcm) == first.samples * 2 * first.channels
    assert len(second.pcm) == second.samples * 2 * second.channels


@pytest.mark.anyio
async def test_ingest_resets_cleanly_across_a_reconnect() -> None:
    """A drop mid-epoch must not leak partial state into the resumed stream.

    `flaky_replay_server` severs the first connection after `drop_after`
    frames, then serves the *entire* fixture again on the next attempt,
    genuinely exercising `MediaClient`'s reconnect logic rather than
    simulating it. Dropping at frame 5 lands after two chunks have already
    been buffered but well before the gap at chunk 10, so this specifically
    catches a real bug this test caught once already: if `epoch_started`
    handling *finishes* a leftover builder instead of discarding it, those
    two interrupted chunks turn into a spurious third segment.
    """
    policy = ReconnectPolicy(initial_seconds=0.01, max_seconds=0.05, jitter=0.0)

    async with flaky_replay_server("audio_session_basic", drop_after=5) as url:
        segments = await _collect(url, policy=policy)

    # Same shape as the clean run above -- the interrupted first attempt
    # contributed nothing to the result.
    assert len(segments) == 2
    assert segments[0].samples == 48_000
    assert segments[1].samples == 96_000
