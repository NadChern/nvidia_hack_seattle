"""Real-Parakeet test for `WS /v1/stt/{session_id}`. Skips entirely -- not a
failure -- when `mlx`/`parakeet-mlx` aren't installed, same as
`tests/test_parakeet_backend.py`.

Assertions here are structural (message count, session_id, segment
boundaries), not "the text says X": `audio_session_basic` is a recorded PCM
fixture built to exercise the gap-detection path, not a known speech
phrase -- there's no guarantee Parakeet transcribes it to anything
meaningful, so asserting real word content here would be testing the
fixture's audio content, not this endpoint. `test_parakeet_backend.py`
already covers real transcription accuracy against a known `say`-generated
phrase; this test is about the WebSocket plumbing end to end with a real
model behind it, not about accuracy.

Plain synchronous test function, `replay_server` entered through
`anyio.from_thread.start_blocking_portal()` -- see `test_api_stt.py`'s
module docstring for why the obvious `async def test_...(): async with
replay_server(...): ... TestClient ...` form deadlocks.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx", reason="mlx is Apple Silicon only; optional dependency group")
pytest.importorskip("parakeet_mlx", reason="parakeet-mlx is an optional dependency group")

from anyio.from_thread import start_blocking_portal  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from visual_memory_media_contract.testing import replay_server  # noqa: E402

from speech.config import Settings  # noqa: E402
from speech.main import app  # noqa: E402
from speech.parakeet_backend import ParakeetMlxSpeechToText  # noqa: E402

_FIXTURE_SESSION_ID = "sess_01JAB000000000000000000"


def test_stt_streams_real_transcripts_via_parakeet() -> None:
    with start_blocking_portal(backend="asyncio") as portal:
        with portal.wrap_async_context_manager(replay_server("audio_session_basic")) as url:
            app.state.settings = Settings(gateway_audio_url=url)
            app.state.stt_backend = ParakeetMlxSpeechToText()

            with (
                TestClient(app) as client,
                client.websocket_connect(f"/v1/stt/{_FIXTURE_SESSION_ID}") as websocket,
            ):
                first = websocket.receive_json()
                second = websocket.receive_json()

    assert first["session_id"] == _FIXTURE_SESSION_ID
    assert second["session_id"] == _FIXTURE_SESSION_ID
    assert first["pts_samples_start"] == 0
    assert second["pts_samples_start"] == 72_000
    # `text` is a string either way (possibly empty for non-speech PCM) --
    # just confirm the field exists and the pipeline actually ran.
    assert isinstance(first["text"], str)
    assert isinstance(second["text"], str)
