"""Tests for `speech.resample`. Fully offline: no gateway, no LiveKit, no
model, no network -- synthetic PCM plus the recorded `audio_session_basic`
fixture only.
"""

import numpy as np
import pytest
from visual_memory_media_contract.testing import replay_server

from speech.ingest import ingest_segments
from speech.resample import resample_pcm, resample_segment


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_resample_pcm_produces_exact_expected_sample_count_and_length() -> None:
    """48 kHz -> 16 kHz on synthetic PCM: exact 3:1 downsample.

    Uses a plain sine wave rather than the fixture so this test is a pure,
    isolated check of `resample_pcm`'s own arithmetic -- no `AudioSegment`,
    no fixture, no async machinery involved.
    """
    source_sample_rate = 48_000
    target_sample_rate = 16_000
    duration_seconds = 1.0
    input_samples = int(source_sample_rate * duration_seconds)

    t = np.arange(input_samples) / source_sample_rate
    tone = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype("<i2")
    pcm = tone.tobytes()

    resampled = resample_pcm(
        pcm,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        channels=1,
    )

    expected_samples = 16_000  # 48_000 samples in at a clean 3:1 ratio
    assert len(resampled) == expected_samples * 2  # s16le = 2 bytes/sample/channel


def test_resample_pcm_is_a_noop_when_rates_already_match() -> None:
    pcm = (np.arange(100, dtype="<i2")).tobytes()
    assert (
        resample_pcm(pcm, source_sample_rate=16_000, target_sample_rate=16_000, channels=1) == pcm
    )


def test_resample_pcm_rejects_an_unsupported_sample_format() -> None:
    with pytest.raises(ValueError, match="s16le"):
        resample_pcm(
            b"",
            source_sample_rate=48_000,
            target_sample_rate=16_000,
            channels=1,
            sample_format="unsupported",  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_resample_segment_on_real_ingested_segments_from_the_fixture() -> None:
    """Resamples the two real `AudioSegment`s `audio_session_basic` produces.

    `audio_session_basic` splits into a 48_000-sample segment and a
    96_000-sample segment (see `tests/test_ingest.py`); at a clean 3:1 ratio
    those become exactly 16_000 and 32_000 samples at 16 kHz.
    """
    async with replay_server("audio_session_basic") as url:
        segments = [segment async for segment in ingest_segments(url, reconnect=False)]

    assert len(segments) == 2

    resampled = [resample_segment(segment, target_sample_rate=16_000) for segment in segments]

    assert resampled[0].samples == 16_000
    assert resampled[1].samples == 32_000
    for original, converted in zip(segments, resampled, strict=True):
        assert converted.sample_rate == 16_000
        assert len(converted.pcm) == converted.samples * 2 * converted.channels
        # Identity/location fields must survive resampling unchanged --
        # they point back to where this audio came from on the relay.
        assert converted.session_id == original.session_id
        assert converted.epoch_id == original.epoch_id
        assert converted.pts_samples_start == original.pts_samples_start
