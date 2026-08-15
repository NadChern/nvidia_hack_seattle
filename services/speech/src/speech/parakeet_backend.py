"""Real STT backend: parakeet-mlx, Apple Silicon only (MLX).

Implements the same `SpeechToText` interface `StubSpeechToText` does, so it
can be swapped in without touching anything upstream (`ingest.py`,
`resample.py`, or whatever eventually calls `SpeechToText.transcribe`).

Everything MLX/parakeet-mlx-specific is imported lazily, inside methods, not
at module import time. That is what lets this whole service -- including
`stt.py`, the FastAPI app, and every offline test -- keep importing and
passing with no `mlx` installed at all, which is the normal case on Linux
ARM64 (the GN100 deploy target has no Apple GPU) and in CI. `parakeet-mlx` is
an optional dependency group (`uv sync --group mlx`), not a base dependency,
for the same reason -- see `pyproject.toml`'s comment on the `mlx` group.

Two real API constraints shaped this file, found by reading parakeet-mlx's
actual installed source (`parakeet_mlx/utils.py`, `parakeet_mlx/parakeet.py`,
`parakeet_mlx/audio.py`) rather than assumed:

1. **No revision pinning in the public API.** `parakeet_mlx.from_pretrained`
   calls `huggingface_hub.hf_hub_download` with no `revision` argument, so it
   always fetches whatever the repo's default branch currently has -- exactly
   what hard rule 7 (pin the exact revision, never "latest") forbids. This
   module bypasses `from_pretrained` and calls `hf_hub_download` directly
   with `revision=`, then builds the model with
   `parakeet_mlx.utils.from_config` -- a real function, just not one
   `parakeet_mlx.__init__` re-exports, which is why the dependency is pinned
   tighter than usual in `pyproject.toml`.
2. **`transcribe()` only accepts a file path**, not in-memory audio.
   `BaseParakeet.transcribe`'s only real job before calling the model is
   `load_audio()`, which shells out to `ffmpeg` and produces
   `mx.array(int16_samples).astype(mx.float32) / 32768.0`. This service's
   `AudioSegment.pcm` is already s16le PCM at the model's target sample rate
   after `resample.py` -- already exactly what `load_audio` would produce
   from a file -- so this module builds that same array directly from the
   segment's bytes and calls the lower-level `get_logmel` + `model.generate`
   instead, skipping both `ffmpeg` and a temp file entirely.
"""

# `mlx`, `parakeet_mlx`, and `huggingface_hub` ship no type stubs, so under
# strict mode every value that touches them -- not just the import lines --
# comes back Unknown (~15 call sites: hf_hub_download, tree_flatten,
# mx.float32, get_logmel, model.generate, ...). This file's entire job is
# bridging to that untyped library, so scattering a dozen-plus per-line
# ignores would be noise, not signal. Disabling exactly the stub-related
# rules for this one file keeps strict mode intact everywhere else in the
# service (config.py, ingest.py, resample.py, stt.py, main.py are all still
# fully strict) rather than weakening `[tool.pyright]` repo-wide.
#
# `reportMissingImports`/`reportUnknownParameterType` are disabled too: `mlx`
# is an optional, Darwin-only dependency group (`pyproject.toml`), so CI's
# Linux pyright run has it genuinely uninstalled, not just unstubbed --
# `parakeet_mlx.parakeet` can't be resolved at all there, which cascades into
# "unknown" parameter/return types on every function that names
# `BaseParakeet`. Runtime is unaffected (the import is lazy, inside
# `_load_model_sync`, and `main.py` falls back to `StubSpeechToText` when the
# package isn't importable) -- this is a static-analysis gap only.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingImports=false, reportUnknownParameterType=false

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import numpy as np

from speech.config import get_settings
from speech.ingest import AudioSegment
from speech.stt import Transcript

if TYPE_CHECKING:
    # Only for static typing. Never imported at runtime unless mlx is
    # installed, and even then only from inside `_load_model_sync`.
    from parakeet_mlx.parakeet import BaseParakeet


class ParakeetMlxSpeechToText:
    """Real STT via `parakeet-mlx`. Loads the pinned model once, lazily."""

    def __init__(self) -> None:
        self._model: BaseParakeet | None = None
        self._load_lock = asyncio.Lock()

    async def _get_model(self) -> BaseParakeet:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            # Re-check: another caller may have finished loading while this
            # one was waiting for the lock.
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model_sync)
        return self._model

    def _load_model_sync(self) -> BaseParakeet:
        import mlx.core as mx
        from huggingface_hub import hf_hub_download
        from mlx.utils import tree_flatten, tree_unflatten
        from parakeet_mlx.utils import from_config

        settings = get_settings()
        config_path = hf_hub_download(
            settings.stt_model_source,
            "config.json",
            revision=settings.stt_model_revision,
        )
        weights_path = hf_hub_download(
            settings.stt_model_source,
            "model.safetensors",
            revision=settings.stt_model_revision,
        )
        with open(config_path) as f:
            config = json.load(f)

        model = from_config(config)  # also calls model.eval()
        model.load_weights(weights_path)

        dtype = mx.float32 if settings.stt_model_dtype == "float32" else mx.bfloat16
        weights = dict(tree_flatten(model.parameters()))
        weights = [(k, v.astype(dtype)) for k, v in weights.items()]
        model.update(tree_unflatten(weights))

        return model

    async def transcribe(self, segment: AudioSegment) -> Transcript:
        if segment.channels != 1:
            raise ValueError(
                f"ParakeetMlxSpeechToText only supports mono audio, got {segment.channels} channels"
            )

        model = await self._get_model()
        expected_rate = model.preprocessor_config.sample_rate
        if segment.sample_rate != expected_rate:
            raise ValueError(
                f"segment sample_rate {segment.sample_rate} does not match what this "
                f"model expects ({expected_rate}); check config.py's stt_target_sample_rate"
            )

        text = await asyncio.to_thread(self._transcribe_sync, model, segment.pcm)

        return Transcript(
            text=text,
            session_id=segment.session_id,
            epoch_id=segment.epoch_id,
            pts_samples_start=segment.pts_samples_start,
            samples=segment.samples,
            sample_rate=segment.sample_rate,
        )

    @staticmethod
    def _transcribe_sync(model: BaseParakeet, pcm: bytes) -> str:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        samples_int16 = np.frombuffer(pcm, dtype="<i2")
        audio_data = mx.array(samples_int16).astype(mx.float32) / 32768.0
        mel = get_logmel(audio_data, model.preprocessor_config)
        result = model.generate(mel)[0]
        return result.text


__all__ = ["ParakeetMlxSpeechToText"]
