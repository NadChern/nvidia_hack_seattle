"""Real TTS backend: Kokoro via `mlx-audio`, Apple Silicon only (MLX).

Implements the same `TextToSpeech` interface `StubTextToSpeech` does, so it
can be swapped in without touching anything upstream, mirroring how
`parakeet_backend.py` relates to `stt.py`.

Everything mlx-audio/mlx-specific is imported lazily, inside methods, not at
module import time -- for the same reason as `parakeet_backend.py`: this
whole service and every offline test must keep importing and passing with no
`mlx` installed at all (Linux ARM64 / the GN100 deploy target / CI).
`mlx-audio` and its Kokoro text-processing dependencies (`misaki[en]`, the
`en_core_web_sm` spaCy model) are all in the optional, Darwin-only `mlx`
dependency group, not base dependencies.

Two real things found by reading mlx-audio's actual installed source
(`mlx_audio/tts/utils.py`, `mlx_audio/utils.py`,
`mlx_audio/tts/models/kokoro/kokoro.py`) rather than assumed:

1. **Revision pinning works here, unlike parakeet-mlx.** `mlx_audio.tts.
   utils.load` accepts `revision=` directly and threads it through to
   `huggingface_hub.snapshot_download` via `get_model_path` in
   `mlx_audio/utils.py`. No need to bypass the public loader the way
   `parakeet_backend.py` had to for Parakeet -- confirmed by reading
   `base_load_model`'s source, not assumed from the docstring alone.
2. **`Model.generate()` already returns in-memory audio**, not a file. It is
   a generator of `GenerationResult` objects, each carrying `.audio` (an
   `mx.array` of float32 samples, empirically observed in roughly `[-1, 1]`)
   and `.sample_rate`. The CLI writes these to a `.wav` file separately;
   this module never does. One real gotcha caught by actually running it,
   not by reading the source: `GenerationResult.samples` is **not** the
   real sample count -- it consistently reported `1` regardless of audio
   length in manual testing (looks like a batch-dimension bug in mlx-audio
   itself). This module reads `len(result.audio)` instead, which matches
   the audio actually returned.
"""

# `mlx`, `mlx_audio`, and their text-processing dependencies ship no type
# stubs, so under strict mode every value that touches them comes back
# Unknown, the same situation as `parakeet_backend.py`. Disabling exactly the
# stub-related rules for this one file keeps strict mode intact everywhere
# else in the service.
#
# `reportMissingImports`/`reportUnknownParameterType` are disabled too: `mlx`
# is an optional, Darwin-only dependency group (`pyproject.toml`), so CI's
# Linux pyright run has it genuinely uninstalled, not just unstubbed --
# `mlx_audio.tts.models.kokoro.kokoro` can't be resolved at all there, which
# cascades into "unknown" parameter/return types on every function that
# names `MlxModule`. Runtime is unaffected (the import is lazy, inside
# `_load_model_sync`, and `main.py` falls back to `StubTextToSpeech` when the
# package isn't importable) -- this is a static-analysis gap only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportUnknownParameterType=false

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from speech.config import get_settings
from speech.tts import SpeechAudio

if TYPE_CHECKING:
    # Only for static typing. Never imported at runtime unless mlx is
    # installed, and even then only from inside `_load_model_sync`.
    # `mlx_audio.tts.utils.load`'s own declared return type is the generic
    # `nn.Module`, but that base class doesn't itself declare `.generate` or
    # `.sample_rate` -- those are Kokoro-specific. Typing against the
    # concrete class (like `parakeet_backend.py` does with `BaseParakeet`,
    # not a generic base) is what lets pyright actually resolve them.
    from mlx_audio.tts.models.kokoro.kokoro import Model as MlxModule


class KokoroMlxTextToSpeech:
    """Real TTS via `mlx-audio`'s Kokoro model. Loads the pinned model once, lazily."""

    def __init__(self) -> None:
        self._model: MlxModule | None = None
        self._load_lock = asyncio.Lock()

    async def _get_model(self) -> MlxModule:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            # Re-check: another caller may have finished loading while this
            # one was waiting for the lock.
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    def _load_model_sync(self) -> MlxModule:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten
        from mlx_audio.tts.models.kokoro.kokoro import Model
        from mlx_audio.tts.utils import load

        settings = get_settings()
        loaded = load(settings.tts_model_source, revision=settings.tts_model_revision)
        # `load()` is generically typed to return the base `nn.Module` --
        # correct, since it loads any mlx-audio TTS model, not only Kokoro.
        # This narrows it to the concrete type `synthesize`/`generate` rely
        # on, and doubles as a real check: if `tts_model_source` is ever
        # pointed at a non-Kokoro model, this fails clearly here rather than
        # with a confusing `AttributeError` deep inside `synthesize`.
        if not isinstance(loaded, Model):
            raise TypeError(
                f"expected a Kokoro model from {settings.tts_model_source!r}, "
                f"got {type(loaded).__name__}"
            )
        model = loaded

        dtype = mx.bfloat16 if settings.tts_model_dtype == "bfloat16" else mx.float32
        weights = dict(tree_flatten(model.parameters()))
        weights = [(k, v.astype(dtype)) for k, v in weights.items()]
        model.update(tree_unflatten(weights))

        return model

    async def synthesize(
        self, text: str, *, voice: str | None = None, lang_code: str | None = None
    ) -> SpeechAudio:
        model = await self._get_model()
        settings = get_settings()
        effective_voice = voice if voice is not None else settings.tts_voice
        effective_lang_code = lang_code if lang_code is not None else settings.tts_lang_code

        pcm, native_rate = await asyncio.to_thread(
            self._synthesize_sync, model, text, effective_lang_code, effective_voice
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
    def _synthesize_sync(
        model: MlxModule, text: str, lang_code: str, voice: str | None
    ) -> tuple[bytes, int]:
        import mlx.core as mx

        # Kokoro's own `generate(voice: str = None, ...)` contradicts its own
        # annotation -- the real default is `None`, which it then resolves
        # to "af_heart" internally (confirmed by reading kokoro.py). Passing
        # `None` through is correct; the annotation is wrong, not this call.
        chunks = [
            result.audio
            for result in model.generate(
                text,
                voice=voice,  # pyright: ignore[reportArgumentType]
                lang_code=lang_code,
            )
        ]
        if not chunks:
            raise ValueError("Kokoro produced no audio for the given text")
        audio = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        samples_float = np.clip(np.array(audio), -1.0, 1.0)
        samples_int16 = (samples_float * 32767.0).round().astype("<i2")
        return samples_int16.tobytes(), model.sample_rate


__all__ = ["KokoroMlxTextToSpeech"]
