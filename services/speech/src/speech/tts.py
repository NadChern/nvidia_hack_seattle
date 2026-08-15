"""Typed TTS boundary: the `SpeechAudio` result shape, the `TextToSpeech`
interface, and a deterministic stub implementation with no real model.

Scope, deliberately minimal, mirroring `stt.py`: this carries only the
synthesis result -- raw PCM plus enough to know how to interpret or resample
it. It says nothing about where the audio goes next. The return-audio
transport back through the gateway is a separate, currently blocked stage:
`MediaClient` has no return-audio send method yet (raised with Alex), so
nothing in this module should presuppose how that will eventually work.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict
from visual_memory_media_contract.protocol import SampleFormat


class SpeechAudio(BaseModel):
    """One piece of synthesized speech: raw PCM plus its own encoding.

    Carries the source `text` alongside the audio so a caller doesn't need
    to correlate this back to a request separately.
    """

    model_config = ConfigDict(frozen=True)

    #: What the assistant is about to say. Never pass this to a logger --
    #: docs/07 treats transcript/spoken content the same as raw media.
    text: str
    #: Raw audio. Never pass this to a logger -- `logging.py`'s redaction
    #: filter would reduce it to a byte count as defense in depth, but the
    #: intent is to never hand it to a log call at all.
    pcm: bytes
    sample_rate: int
    channels: int
    sample_format: SampleFormat


class TextToSpeech(Protocol):
    """Interface every TTS backend (stub or real) implements.

    `voice`/`lang_code` are per-call overrides of whatever `config.py`
    defaults to (`tts_voice`, `tts_lang_code`) -- `None` means "use the
    configured default," not "no voice"/"no language."
    """

    async def synthesize(
        self, text: str, *, voice: str | None = None, lang_code: str | None = None
    ) -> SpeechAudio: ...


class StubTextToSpeech:
    """Deterministic placeholder -- no real model.

    Returns real, well-formed PCM (a short burst of silence at the
    configured output rate), not synthesized speech -- exists so a
    TTS-consuming pipeline is testable offline before Kokoro is wired in,
    the same role `StubSpeechToText` plays for the STT side.
    """

    async def synthesize(
        self, text: str, *, voice: str | None = None, lang_code: str | None = None
    ) -> SpeechAudio:
        # `voice`/`lang_code` are accepted, not used -- the stub's output
        # never depends on either, only on the configured output rate. They
        # exist here purely for interface parity with the real backend.
        from speech.config import get_settings

        settings = get_settings()
        sample_rate = settings.tts_output_sample_rate
        duration_seconds = 0.1
        samples = int(sample_rate * duration_seconds)

        return SpeechAudio(
            text=text,
            pcm=b"\x00\x00" * samples,  # s16le silence
            sample_rate=sample_rate,
            channels=1,
            sample_format="s16le",
        )


__all__ = ["SpeechAudio", "StubTextToSpeech", "TextToSpeech"]
