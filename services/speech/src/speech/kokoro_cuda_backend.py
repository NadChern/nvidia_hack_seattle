"""Real TTS backend: Kokoro on CUDA, via the `kokoro` package (torch).

Implements the same `TextToSpeech` interface `StubTextToSpeech` does, and
produces the same shape `kokoro_backend.py` (the Apple Silicon path) does.

**The same logical model as the MLX path**: Kokoro-82M either way, so moving
between a Mac and the GN100 changes the runtime and not the voice.
`hexgrad/Kokoro-82M` is the original; `mlx-community/Kokoro-82M-bf16` is a
conversion of it.

Kokoro's English G2P chain is a hard requirement, not an extra --
`misaki[en]`, and through it the `en_core_web_sm` spaCy model -- exactly as
`kokoro_backend.py` documents for the MLX path. Both are declared in the
`cuda` dependency group rather than left to download at first use.

`espeak-ng` is needed as a *system* package for out-of-dictionary words.
Without it Kokoro still speaks, falling back for words `misaki` cannot look
up, so this is a quality dependency rather than a hard one -- the Dockerfile
installs it, and a dev box without it gets slightly worse pronunciation and no
error.
"""

# `kokoro` and `misaki` ship no type stubs, so under strict mode everything
# touching them is Unknown -- the same situation `kokoro_backend.py`
# documents. `reportMissingImports`: the `cuda` group is genuinely
# uninstalled on a Mac and on CI, not merely unstubbed.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from speech.config import get_settings
from speech.tts import SpeechAudio

logger = logging.getLogger(__name__)


class KokoroCudaTextToSpeech:
    """Kokoro-82M on a CUDA device, loaded once and kept warm."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._device: str = "pending"
        self._lock = asyncio.Lock()

    @property
    def device(self) -> str:
        return self._device

    async def initialize(self) -> None:
        """Load the pipeline and default voice before the first spoken reply."""
        settings = get_settings()
        pipeline = await self._get_pipeline(settings.tts_lang_code)
        await asyncio.to_thread(
            self._synthesize_sync,
            pipeline,
            "Ready.",
            settings.tts_cuda_default_voice,
        )

    async def _get_pipeline(self, lang_code: str) -> Any:
        async with self._lock:
            if self._pipeline is None:
                self._pipeline = await asyncio.to_thread(self._load_blocking, lang_code)
            return self._pipeline

    def _load_blocking(self, lang_code: str) -> Any:
        import torch
        from kokoro import KPipeline

        settings = get_settings()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "loading kokoro",
            extra={
                "device": self._device,
                "source": settings.tts_cuda_model_source,
                "lang_code": lang_code,
            },
        )
        # `repo_id` is passed explicitly: KPipeline warns and guesses a default
        # otherwise, and a model this service pins should never be chosen by a
        # library's fallback.
        return KPipeline(
            lang_code=lang_code,
            repo_id=settings.tts_cuda_model_source,
            device=self._device,
        )

    async def synthesize(
        self, text: str, *, voice: str | None = None, lang_code: str | None = None
    ) -> SpeechAudio:
        settings = get_settings()
        effective_lang_code = lang_code if lang_code is not None else settings.tts_lang_code
        effective_voice = voice if voice is not None else settings.tts_cuda_default_voice

        pipeline = await self._get_pipeline(effective_lang_code)
        pcm, native_rate = await asyncio.to_thread(
            self._synthesize_sync, pipeline, text, effective_voice
        )

        target_rate = settings.tts_output_sample_rate
        if target_rate != native_rate:
            from speech.resample import resample_pcm

            pcm = resample_pcm(
                pcm,
                source_sample_rate=native_rate,
                target_sample_rate=target_rate,
                channels=1,
            )
            output_rate = target_rate
        else:
            output_rate = native_rate

        return SpeechAudio(
            text=text,
            pcm=pcm,
            sample_rate=output_rate,
            channels=1,
            sample_format="s16le",
        )

    @staticmethod
    def _synthesize_sync(pipeline: Any, text: str, voice: str) -> tuple[bytes, int]:
        # KPipeline yields one result per chunk it split the text into, each
        # carrying `.audio` as a torch tensor of float32 samples in [-1, 1].
        # Long text becomes several chunks, so they are concatenated rather
        # than taking the first -- the failure that would otherwise truncate
        # every sentence after the first.
        chunks = [
            result.audio for result in pipeline(text, voice=voice) if result.audio is not None
        ]
        if not chunks:
            raise ValueError("Kokoro produced no audio for the given text")

        audio = np.concatenate(
            [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks]
        )
        samples = np.clip(audio, -1.0, 1.0)
        pcm = (samples * 32767.0).astype(np.int16).tobytes()
        # Kokoro's own output rate, fixed by the model rather than configured.
        return pcm, 24_000


__all__ = ["KokoroCudaTextToSpeech"]
