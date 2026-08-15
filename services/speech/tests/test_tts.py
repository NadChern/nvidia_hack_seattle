"""Tests for `speech.tts`. Fully offline: no model, no network."""

import pytest

from speech.tts import SpeechAudio, StubTextToSpeech


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_stub_text_to_speech_returns_well_formed_speech_audio() -> None:
    tts = StubTextToSpeech()

    audio = await tts.synthesize("hello")

    assert isinstance(audio, SpeechAudio)
    assert audio.text == "hello"
    assert audio.channels == 1
    assert audio.sample_format == "s16le"
    assert audio.sample_rate > 0
    assert audio.pcm  # non-empty
    # s16le = 2 bytes/sample/channel; must be a whole number of samples.
    assert len(audio.pcm) % 2 == 0
