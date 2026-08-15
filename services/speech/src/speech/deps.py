"""Request-scoped helpers shared by the routers.

Mirrors `application_memory/deps.py`'s shape: plain functions that pull
whatever `main.py` put on `app.state`, called directly from a route handler
rather than through FastAPI's `Depends()` injection -- matching the rest of
this repository's services, not just a local preference.
"""

from __future__ import annotations

from fastapi import Request, WebSocket
from starlette.requests import HTTPConnection

from speech.config import Settings
from speech.stt import SpeechToText
from speech.tts import TextToSpeech


def settings_of(connection: HTTPConnection) -> Settings:
    """Typed against `HTTPConnection`, the real base class `Request` and
    `WebSocket` share, since reading settings has no reason to differ
    between an HTTP route (`synthesize.py`) and a WebSocket one (`stt.py`).
    """
    settings: Settings = connection.app.state.settings
    return settings


def tts_backend_of(request: Request) -> TextToSpeech:
    backend: TextToSpeech = request.app.state.tts_backend
    return backend


def stt_backend_of(websocket: WebSocket) -> SpeechToText:
    backend: SpeechToText = websocket.app.state.stt_backend
    return backend


__all__ = ["settings_of", "stt_backend_of", "tts_backend_of"]
