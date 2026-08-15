"""Real-model test for `speech.kokoro_backend`.

Skips entirely -- not a failure -- when `mlx`/`mlx-audio`/`misaki` aren't
installed. That's the default state: they're an optional dependency group
(`pyproject.toml`'s `mlx` group), not part of the base install, so the rest
of this service's test suite never needs them. This is the one test in the
suite that needs the actual ~339 MB pinned Kokoro checkpoint and does real
inference -- everything else stays fully offline and model-free, per
role-prompts/Speech.md's no-hardware start path.

Run explicitly, once the model is available:
    uv sync --group mlx && uv run pytest tests/test_kokoro_backend.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx", reason="mlx is Apple Silicon only; optional dependency group")
pytest.importorskip("mlx_audio", reason="mlx-audio is an optional dependency group")
pytest.importorskip("misaki", reason="misaki is an optional dependency group (Kokoro's G2P)")

from speech.config import get_settings  # noqa: E402
from speech.kokoro_backend import KokoroMlxTextToSpeech  # noqa: E402
from speech.tts import SpeechAudio  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_kokoro_mlx_synthesizes_a_short_phrase(tmp_path: Path) -> None:
    settings = get_settings()
    tts = KokoroMlxTextToSpeech()

    audio = await tts.synthesize("Testing one two three.")

    assert isinstance(audio, SpeechAudio)
    assert audio.pcm, "Kokoro produced no audio"
    assert len(audio.pcm) % 2 == 0  # whole number of s16le samples
    assert audio.sample_rate == settings.tts_output_sample_rate
    assert audio.channels == 1
    assert audio.sample_format == "s16le"

    # Write to tmp_path only, never into the service directory -- this test
    # produces a real file purely so a human can spot-check it, not as part
    # of the suite's own state.
    wav_path = tmp_path / "kokoro_check.wav"
    import wave

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(audio.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.pcm)
    assert wav_path.stat().st_size > 0
