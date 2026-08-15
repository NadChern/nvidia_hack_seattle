"""Text-to-speech synthesis.

`POST /v1/synthesize` turns text into audio a caller can play or forward
immediately -- a self-describing WAV file, not raw PCM the caller has to
separately know how to interpret. Which `TextToSpeech` backend actually does
the synthesis was decided once at startup (`main.py`'s
`_select_tts_backend`), not per request; this router only asks `deps.py` for
whatever was decided.

Out of scope here, deliberately: the STT endpoint (a later part), the
return-audio transport back through the gateway (blocked -- `MediaClient` has
no send method yet), and Memory/query wiring. This endpoint only synthesizes
and hands back audio; it doesn't send it anywhere.
"""

from __future__ import annotations

import io
import wave

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from speech.deps import tts_backend_of

router = APIRouter(tags=["synthesize"])


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    #: Per-call overrides of `config.py`'s `tts_voice`/`tts_lang_code`.
    #: `None` means "use the configured default," not "no voice"/"no
    #: language" -- see `TextToSpeech.synthesize`.
    voice: str | None = None
    lang_code: str | None = None


def _wav_bytes(*, pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw s16le PCM in a real RIFF/WAVE header.

    Self-describing on purpose: whoever consumes this response -- Alex, a
    browser `<audio>` tag, a future return-audio transport -- can read the
    sample rate and channel count straight off the file instead of needing
    this service's config or a side channel to know how to interpret the
    bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # s16le
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


@router.post("/v1/synthesize")
async def synthesize(body: SynthesizeRequest, request: Request) -> Response:
    tts = tts_backend_of(request)
    audio = await tts.synthesize(body.text, voice=body.voice, lang_code=body.lang_code)
    wav = _wav_bytes(pcm=audio.pcm, sample_rate=audio.sample_rate, channels=audio.channels)
    return Response(content=wav, media_type="audio/wav")
