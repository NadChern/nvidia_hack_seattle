"""Return audio framing and the endpoints that feed it.

Publishing into a real room needs LiveKit, so that is integration territory.
What is testable here is the framing, which is where silent corruption would
come from, and the endpoint guards.
"""

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from media_gateway.config import Settings
from media_gateway.main import create_app
from media_gateway.transport.return_audio import FRAME_MS, ReturnAudio, silence

pytestmark = pytest.mark.anyio

SAMPLE_RATE = 48_000


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeSource:
    """Stands in for rtc.AudioSource, recording what it was given."""

    def __init__(self) -> None:
        self.frames: list[object] = []

    async def capture_frame(self, frame: object) -> None:
        self.frames.append(frame)

    async def aclose(self) -> None:
        pass


def an_audio(channels: int = 1) -> tuple[ReturnAudio, FakeSource]:
    audio = ReturnAudio(
        sample_rate=SAMPLE_RATE,
        channels=channels,
        queue_size_ms=200,
        track_name="assistant-tts",
    )
    source = FakeSource()
    audio._source = source  # type: ignore[assignment]  # noqa: SLF001
    return audio, source


async def test_pcm_is_split_into_whole_frames() -> None:
    audio, source = an_audio()
    per_frame_bytes = SAMPLE_RATE * FRAME_MS // 1000 * 2

    frames = await audio.feed(bytes(per_frame_bytes * 5))

    assert frames == 5
    assert len(source.frames) == 5


async def test_a_trailing_partial_frame_is_padded_not_dropped() -> None:
    """Dropping it would clip the end of every utterance."""
    audio, source = an_audio()
    per_frame_bytes = SAMPLE_RATE * FRAME_MS // 1000 * 2

    frames = await audio.feed(bytes(per_frame_bytes + 10))

    assert frames == 2
    assert len(source.frames) == 2


async def test_feeding_before_publishing_is_an_error() -> None:
    audio = ReturnAudio(
        sample_rate=SAMPLE_RATE, channels=1, queue_size_ms=200, track_name="assistant-tts"
    )

    with pytest.raises(RuntimeError, match="not been published"):
        await audio.feed(b"\x00\x00")


async def test_a_tone_produces_the_expected_frame_count() -> None:
    audio, _ = an_audio()

    frames = await audio.play_tone(hz=440, seconds=1.0)

    assert frames == 1000 // FRAME_MS
    assert audio.frames_sent == frames


async def test_tone_phase_carries_across_calls() -> None:
    """A phase reset between utterances is an audible click."""
    audio, source = an_audio()

    await audio.play_tone(hz=440, seconds=0.1)
    first_phase = audio._phase  # noqa: SLF001
    await audio.play_tone(hz=440, seconds=0.1)

    assert first_phase != 0
    assert audio._phase != first_phase  # noqa: SLF001
    assert len(source.frames) == 10


def test_silence_is_the_right_length() -> None:
    data = silence(SAMPLE_RATE, 1, 20)

    assert len(data) == SAMPLE_RATE * 20 // 1000 * 2
    assert set(np.frombuffer(data, dtype="<i2")) == {0}


# --- Endpoints ------------------------------------------------------------


def an_app() -> FastAPI:
    return create_app(
        Settings(environment="ci", media_source="scripted", scripted_frame_interval_s=30.0)
    )


async def call(app: FastAPI, method: str, path: str):  # type: ignore[no-untyped-def]
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            return await http.request(method, path)


async def test_a_tone_for_an_unknown_session_is_explicit() -> None:
    response = await call(an_app(), "POST", "/v1/return-audio/sess_missing/tone")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize(
    "query",
    ["hz=0", "hz=99999", "seconds=0", "seconds=120"],
)
async def test_tone_parameters_are_bounded(query: str) -> None:
    """An unbounded tone would hold the device's speaker indefinitely."""
    response = await call(an_app(), "POST", f"/v1/return-audio/sess_x/tone?{query}")

    assert response.status_code == 422
