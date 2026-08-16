"""What this service is actually running.

`main.py` picks a backend per capability at startup -- the real MLX models when
their packages import, a deterministic stub otherwise -- and until now nothing
reported which. That silence is a problem rather than an omission: the stub TTS
returns a tenth of a second of **pure silence**, with a 200 and a valid WAV
header, so a caller that cannot ask has no way to tell "synthesis is not
installed here" from "synthesis is broken". Both look like pressing a button
and hearing nothing.

So the backend name is reported, and `real` states plainly whether a model is
behind it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Request

from speech import __version__
from speech.deps import settings_of
from speech.stt import StubSpeechToText
from speech.tts import StubTextToSpeech

router = APIRouter(tags=["status"])


@router.get("/v1/status")
def status(request: Request) -> dict[str, Any]:
    """Report configuration and which backends are in force."""
    settings = settings_of(request)
    tts = request.app.state.tts_backend
    stt = request.app.state.stt_backend

    return {
        "service": "speech",
        "version": __version__,
        "now": dt.datetime.now(dt.UTC),
        "config": {
            "gateway_audio_url": settings.gateway_audio_url,
            "expected_audio_sample_rate": settings.expected_audio_sample_rate,
            "warm_models_on_startup": settings.warm_models_on_startup,
        },
        "backends": {
            "tts": {
                "name": type(tts).__name__,
                # The distinction a caller needs. A stub is a working endpoint
                # with no model behind it, which is a different thing from a
                # broken one and should never be mistaken for a real voice.
                "real": not isinstance(tts, StubTextToSpeech),
            },
            "stt": {
                "name": type(stt).__name__,
                "real": not isinstance(stt, StubSpeechToText),
            },
        },
    }


__all__ = ["router"]
