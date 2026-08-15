"""Tests for `POST /v1/synthesize`. Forces the stub backend, so this passes
with no mlx installed regardless of what happens to be on the machine
running it -- exactly what a Linux/CI run will actually have.
"""

from __future__ import annotations

import wave
from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient

from speech.config import get_settings
from speech.main import app
from speech.tts import StubTextToSpeech


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _force_stub_backend() -> None:
    """Deterministic regardless of whether mlx happens to be installed
    wherever this test runs -- this test is about the endpoint's plumbing
    (request handling, WAV framing), not about which backend produced the
    bytes.
    """
    app.state.tts_backend = StubTextToSpeech()


@pytest.mark.anyio
async def test_synthesize_returns_a_valid_wav() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/synthesize", json={"text": "hello there"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    settings = get_settings()
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getframerate() == settings.tts_output_sample_rate
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2  # s16le
        assert wav_file.getnframes() > 0


@pytest.mark.anyio
async def test_synthesize_rejects_empty_text() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/synthesize", json={"text": ""})

    assert response.status_code == 422  # FastAPI's request-validation error


@pytest.mark.anyio
async def test_synthesize_accepts_voice_and_lang_code_overrides() -> None:
    """The stub ignores these, but the request shape must accept them --
    this is the "optional voice, lang overrides" surface the endpoint is
    supposed to expose.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/synthesize",
            json={"text": "hello there", "voice": "af_heart", "lang_code": "a"},
        )

    assert response.status_code == 200
