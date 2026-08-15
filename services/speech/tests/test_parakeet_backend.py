"""Real-model test for `speech.parakeet_backend`.

Skips entirely -- not a failure -- when `mlx`/`parakeet-mlx` aren't
installed. That's the default state: they're an optional dependency group
(`pyproject.toml`'s `mlx` group), not part of the base install, so the rest
of this service's test suite never needs them. This is the one test in the
suite that needs the actual ~2.3 GB pinned checkpoint and does real
inference -- everything else stays fully offline and model-free, per
role-prompts/Speech.md's no-hardware start path.

Run explicitly, once the model is available:
    uv sync --group mlx && uv run pytest tests/test_parakeet_backend.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

pytest.importorskip("mlx", reason="mlx is Apple Silicon only; optional dependency group")
pytest.importorskip("parakeet_mlx", reason="parakeet-mlx is an optional dependency group")

from speech.config import get_settings  # noqa: E402
from speech.ingest import AudioSegment  # noqa: E402
from speech.parakeet_backend import ParakeetMlxSpeechToText  # noqa: E402


def _say_to_wav(text: str, path: Path, *, sample_rate: int) -> None:
    """Generate a known WAV clip with macOS `say`, skipping the AIFF trap.

    `say`'s default output format silently produces a file `ffmpeg` (and
    therefore parakeet-mlx) can't read -- 0 audio frames, no error until much
    later. `--file-format=WAVE --data-format=LEI16@<rate>` avoids it
    entirely. See `SY-Knowledge.md`'s session note on this exact gotcha.
    """
    subprocess.run(
        [
            "say",
            text,
            "--file-format=WAVE",
            f"--data-format=LEI16@{sample_rate}",
            "-o",
            str(path),
        ],
        check=True,
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_parakeet_mlx_transcribes_a_known_clip(tmp_path: Path) -> None:
    if shutil.which("say") is None:
        pytest.skip("macOS 'say' is not available to generate a known test clip")

    settings = get_settings()
    sample_rate = settings.stt_target_sample_rate
    wav_path = tmp_path / "known_clip.wav"
    known_phrase = "This is a test of the Parakeet speech recognition system."
    _say_to_wav(known_phrase, wav_path, sample_rate=sample_rate)

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == sample_rate
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2  # s16le
        pcm = wav_file.readframes(wav_file.getnframes())

    assert pcm, (
        "say produced an empty clip -- exactly the AIFF/empty-audio trap "
        "this fixed-format invocation exists to avoid"
    )

    segment = AudioSegment(
        session_id="sess-test",
        epoch_id="epoch-test",
        sample_rate=sample_rate,
        channels=1,
        sample_format="s16le",
        pts_samples_start=0,
        samples=len(pcm) // 2,
        first_sample_captured_at="2026-01-01T00:00:00.000Z",
        pcm=pcm,
    )

    stt = ParakeetMlxSpeechToText()
    transcript = await stt.transcribe(segment)

    assert transcript.text.strip()
    # Loose on purpose -- exact ASR wording/punctuation isn't guaranteed --
    # but at least one real word from the known phrase should survive.
    assert "test" in transcript.text.lower() or "parakeet" in transcript.text.lower()
    assert transcript.session_id == segment.session_id
    assert transcript.epoch_id == segment.epoch_id
    assert transcript.pts_samples_start == segment.pts_samples_start
