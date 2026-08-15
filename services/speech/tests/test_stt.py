"""Tests for `speech.stt`, including the full offline pipeline
(ingest -> resample -> transcribe) with the stub. Fully offline: no gateway,
no LiveKit, no model, no network.
"""

import pytest
from visual_memory_media_contract.testing import replay_server

from speech.config import get_settings
from speech.ingest import ingest_segments
from speech.resample import resample_segment
from speech.stt import StubSpeechToText, Transcript


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _one_real_segment() -> object:
    async with replay_server("audio_session_basic") as url:
        segments = [segment async for segment in ingest_segments(url, reconnect=False)]
    return segments[0]


@pytest.mark.anyio
async def test_stub_speech_to_text_returns_a_populated_transcript() -> None:
    segment = await _one_real_segment()

    transcript = await StubSpeechToText().transcribe(segment)

    assert isinstance(transcript, Transcript)
    assert transcript.text  # non-empty; content is a placeholder, not asserted
    # Source-location fields must match the segment that was transcribed.
    assert transcript.session_id == segment.session_id
    assert transcript.epoch_id == segment.epoch_id
    assert transcript.pts_samples_start == segment.pts_samples_start
    assert transcript.samples == segment.samples
    assert transcript.sample_rate == segment.sample_rate


@pytest.mark.anyio
async def test_offline_pipeline_ingest_resample_transcribe() -> None:
    """The full Stage B Part 1 pipeline, with no real model anywhere in it.

    `audio_session_basic` -> `ingest_segments` -> `resample_segment` ->
    `StubSpeechToText.transcribe`, asserting one `Transcript` per ingested
    segment. This is the shape real Parakeet wiring will eventually replace
    `StubSpeechToText` inside, without changing anything upstream of it.
    """
    target_rate = get_settings().stt_target_sample_rate
    stt = StubSpeechToText()

    async with replay_server("audio_session_basic") as url:
        segments = [segment async for segment in ingest_segments(url, reconnect=False)]

    assert len(segments) == 2  # audio_session_basic's one documented gap

    resampled = [resample_segment(segment, target_sample_rate=target_rate) for segment in segments]
    transcripts = [await stt.transcribe(segment) for segment in resampled]

    assert len(transcripts) == len(segments)
    for transcript, original in zip(transcripts, segments, strict=True):
        assert isinstance(transcript, Transcript)
        assert transcript.sample_rate == target_rate
        # Even after resampling, the transcript still points back to the
        # original segment's place in the source epoch.
        assert transcript.session_id == original.session_id
        assert transcript.epoch_id == original.epoch_id
        assert transcript.pts_samples_start == original.pts_samples_start
