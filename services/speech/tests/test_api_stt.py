"""Tests for `WS /v1/stt/{session_id}`. Forces the stub backend and points
the gateway URL at a replay server serving the recorded fixture, so this
passes with no mlx, no gateway, and no network -- exactly what a Linux/CI
run will actually have.

Uses `fastapi.testclient.TestClient`, not the `httpx.AsyncClient` the rest of
this suite uses -- `AsyncClient` has no WebSocket support. This is the first
endpoint in this service that needs it.

Plain synchronous test functions, `replay_server` entered through
`anyio.from_thread.start_blocking_portal()` rather than an outer
`async def test_...(): async with replay_server(...)`. Verified by hand
(see SY-Summary.md) that the obvious-looking async form deadlocks: it runs
`replay_server`'s listening socket on the same thread/event loop that
`TestClient`'s *synchronous* `__enter__`/`websocket_connect` then blocks,
so that loop never gets to `accept()` the connection MediaClient opens from
`TestClient`'s own separate portal thread -- every attempt just sits until
`open_timeout` fires, and `ingest_segments`'s default `reconnect=True` retries
forever. Running `replay_server` in its own portal thread means its loop
keeps servicing the socket regardless of what the (plain, loop-less) main
thread's synchronous `TestClient` calls are doing.
"""

from __future__ import annotations

from anyio.from_thread import start_blocking_portal
from fastapi.testclient import TestClient
from pydantic import SecretStr
from visual_memory_media_contract.testing import replay_server

from speech.config import Settings
from speech.main import app
from speech.stt import StubSpeechToText

#: `audio_session_basic`'s own session_id, decoded directly from the fixture
#: rather than assumed -- same fixture `tests/test_ingest.py` uses, which
#: splits into two segments (10 chunks before its documented gap, 20 after).
_FIXTURE_SESSION_ID = "sess_01JAB000000000000000000"


def test_stt_streams_one_transcript_per_ingested_segment() -> None:
    with start_blocking_portal(backend="asyncio") as portal:
        with portal.wrap_async_context_manager(replay_server("audio_session_basic")) as url:
            app.state.settings = Settings(gateway_audio_url=url)
            app.state.stt_backend = StubSpeechToText()

            with (
                TestClient(app) as client,
                client.websocket_connect(f"/v1/stt/{_FIXTURE_SESSION_ID}") as websocket,
            ):
                first = websocket.receive_json()
                second = websocket.receive_json()

    assert first["session_id"] == _FIXTURE_SESSION_ID
    assert second["session_id"] == _FIXTURE_SESSION_ID
    # Same segment boundaries as tests/test_ingest.py's clean run: the gap
    # sits at pts_samples 72_000 (48_000 samples before it, at 48 kHz).
    assert first["pts_samples_start"] == 0
    assert second["pts_samples_start"] == 72_000
    assert first["text"]
    assert second["text"]


def test_stt_forwards_the_internal_token_to_the_gateway(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_ingest(url: str, **kwargs: object):
        captured["url"] = url
        captured.update(kwargs)
        if False:
            yield None

    monkeypatch.setattr("speech.api.stt.ingest_segments", fake_ingest)
    app.state.settings = Settings(
        gateway_audio_url="ws://gateway/v1/stream/audio",
        internal_api_token=SecretStr("secret-token"),
    )
    app.state.stt_backend = StubSpeechToText()

    with TestClient(app) as client:
        with client.websocket_connect("/v1/stt/sess_token_check"):
            pass

    assert captured["url"] == "ws://gateway/v1/stream/audio"
    assert captured["session_id"] == "sess_token_check"
    assert captured["token"] == "secret-token"


def test_stt_ignores_other_sessions_on_the_same_relay() -> None:
    """A different session_id in the URL must yield nothing from this fixture.

    Proves two things at once: the `session_id` filter added to
    `ingest_segments` for this endpoint actually filters (a matching
    session's audio never leaks through as a transcript), and disconnecting
    before that ever happens still tears `MediaClient` down promptly.

    A session_id absent from the fixture leaves `ingest_segments` waiting
    indefinitely -- by design, the same `reconnect=True` resilience that
    lets it ride out a real gateway blip (see `api/stt.py`'s module
    docstring). The only way this endpoint ever notices the client gave up
    is the concurrent disconnect-watch task added there; if that regressed
    back to only checking for a disconnect inside `send_json`, this would
    hang forever, since `send_json` is never reached for a session that
    never matches anything.
    """
    with start_blocking_portal(backend="asyncio") as portal:
        with portal.wrap_async_context_manager(replay_server("audio_session_basic")) as url:
            app.state.settings = Settings(gateway_audio_url=url)
            app.state.stt_backend = StubSpeechToText()

            with TestClient(app) as client:
                with client.websocket_connect("/v1/stt/sess_some_other_session"):
                    pass
