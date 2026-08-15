"""Real STT backend: Parakeet on CUDA, via `transformers`.

Implements the same `SpeechToText` interface `StubSpeechToText` does, so it
swaps in without touching anything upstream -- exactly as
`parakeet_backend.py` (the Apple Silicon path) does.

**The same logical model as the MLX path**, which is the point: both load
`parakeet-tdt-0.6b-v3`, so switching hardware changes the runtime and not what
the assistant hears. `mlx-community/parakeet-tdt-0.6b-v3` is a conversion of
`nvidia/parakeet-tdt-0.6b-v3`; this loads the original.

**No NeMo.** NVIDIA publishes v3 with `library_name: transformers` --
`config.json`, `model.safetensors`, a tokenizer, and `ParakeetForTDT` in
transformers 5.6+. NeMo would work too and is how v2 had to be loaded, but it
is an enormous dependency tree (hydra, lightning, and their transitive
closure) for one model that the standard loader now handles. Checked against
the model card's own metadata rather than assumed.

Everything torch- and transformers-specific is imported lazily inside methods,
never at module import time, for the same reason as the MLX backends: this
service and every offline test must keep importing and passing with none of it
installed. The `cuda` dependency group is Linux-only and optional.
"""

# `transformers` ships partial stubs and its `from_pretrained` returns
# `Unknown` under strict mode, the same situation the MLX backends document.
# Disabling exactly the stub-related rules for this one file keeps strict mode
# intact everywhere else in the service. `reportMissingImports` too: the
# `cuda` group is genuinely uninstalled on a Mac and on CI, not merely
# unstubbed -- which is what the runtime probe in `main.py` handles.
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from speech.config import get_settings
from speech.stt import Transcript

if TYPE_CHECKING:
    from speech.ingest import AudioSegment

logger = logging.getLogger(__name__)


class ParakeetCudaSpeechToText:
    """Parakeet TDT 0.6B v3 on a CUDA device, loaded once and kept warm."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str = "pending"
        self._lock = asyncio.Lock()

    @property
    def device(self) -> str:
        return self._device

    async def _get_model(self) -> tuple[Any, Any]:
        """Load on first use, exactly once.

        A lock rather than a bare check: two transcripts can arrive together
        from a single relay burst, and loading a 0.6B model twice would
        duplicate it in VRAM on a device that has to share with a detector.
        """
        async with self._lock:
            if self._model is None or self._processor is None:
                self._model, self._processor = await asyncio.to_thread(self._load_blocking)
            return self._model, self._processor

    def _load_blocking(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        settings = get_settings()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "loading parakeet",
            extra={
                "device": self._device,
                "source": settings.stt_cuda_model_source,
                "revision": settings.stt_cuda_model_revision,
            },
        )

        # `revision=` is pinned, never "latest" -- the same rule
        # `model-manifest.toml` states and `parakeet_backend.py` had to bypass
        # its package's public loader to honour. The standard transformers
        # loader takes it directly, so nothing here reaches past a public API.
        common = {
            "revision": settings.stt_cuda_model_revision,
        }
        processor = AutoProcessor.from_pretrained(settings.stt_cuda_model_source, **common)
        # `AutoModelForTDT`, not `AutoModelForCTC`. Parakeet v3 is a
        # token-and-duration transducer; the CTC auto class refuses its
        # config outright, which is how this was caught.
        model = AutoModelForTDT.from_pretrained(
            settings.stt_cuda_model_source,
            dtype=getattr(torch, settings.stt_cuda_model_dtype),
            **common,
        )
        model.to(self._device)
        model.eval()
        return model, processor

    async def transcribe(self, segment: AudioSegment) -> Transcript:
        # The same two preconditions `parakeet_backend.py` enforces, and for
        # the same reason: silently mixing down or resampling here would hide
        # a misconfigured `stt_target_sample_rate` behind a slightly worse
        # transcript rather than reporting it.
        if segment.channels != 1:
            raise ValueError(
                f"ParakeetCudaSpeechToText only supports mono audio, "
                f"got {segment.channels} channels"
            )
        expected_rate = get_settings().stt_target_sample_rate
        if segment.sample_rate != expected_rate:
            raise ValueError(
                f"segment sample_rate {segment.sample_rate} does not match what this "
                f"model expects ({expected_rate}); check config.py's stt_target_sample_rate"
            )

        model, processor = await self._get_model()
        text = await asyncio.to_thread(
            self._transcribe_sync, model, processor, segment.pcm, expected_rate
        )
        return Transcript(
            text=text,
            session_id=segment.session_id,
            epoch_id=segment.epoch_id,
            pts_samples_start=segment.pts_samples_start,
            samples=segment.samples,
            sample_rate=segment.sample_rate,
        )

    @staticmethod
    def _transcribe_sync(model: Any, processor: Any, pcm: bytes, sample_rate: int) -> str:
        import torch

        # The model wants float32 in [-1, 1]. `<i2` rather than `np.int16` to
        # state the endianness explicitly, matching `parakeet_backend.py` --
        # the relay's PCM is little-endian by contract, not by host luck.
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

        inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
        # Move to the device, and match the model's dtype -- but only for the
        # floating-point tensors. The processor returns features as float32
        # regardless of how the model was loaded, so a bf16 model rejects them
        # ("Input type (float) and bias type (c10::BFloat16) should be the
        # same"); casting everything instead would corrupt the integer masks
        # and lengths alongside them.
        inputs = {
            key: (
                value.to(device=model.device, dtype=model.dtype)
                if value.is_floating_point()
                else value.to(model.device)
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model.generate(**inputs)

        # `.sequences`, not the output object. `generate` returns a
        # `ParakeetRNNTGenerateOutput` dataclass; handing that straight to
        # `batch_decode` iterates its *field names* and fails with
        # "'str' object cannot be interpreted as an integer".
        decoded = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""


__all__ = ["ParakeetCudaSpeechToText"]
