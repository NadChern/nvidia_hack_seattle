"""Real-Kokoro test for `POST /v1/synthesize`. Skips entirely -- not a
failure -- when `mlx`/`mlx-audio`/`misaki` aren't installed, same as
`tests/test_kokoro_backend.py`. Separate file from `test_api_synthesize.py`
on purpose: that file's `autouse` fixture forces the stub backend for every
test in it, which would defeat this one.
"""

from __future__ import annotations

import wave
from io import BytesIO

import pytest

pytest.importorskip("mlx", reason="mlx is Apple Silicon only; optional dependency group")
pytest.importorskip("mlx_audio", reason="mlx-audio is an optional dependency group")
pytest.importorskip("misaki", reason="misaki is an optional dependency group (Kokoro's G2P)")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from speech.kokoro_backend import KokoroMlxTextToSpeech  # noqa: E402
from speech.main import app  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_synthesize_with_real_kokoro_backend() -> None:
    app.state.tts_backend = KokoroMlxTextToSpeech()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/synthesize", json={"text": "Testing one two three."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnframes() > 0
