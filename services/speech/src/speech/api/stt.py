"""Speech-to-text, streamed over a WebSocket.

`WS /v1/stt/{session_id}` streams one JSON `Transcript` per contiguous audio
segment, as the pipeline produces them: `ingest_segments` (this session's
audio, scoped out of the shared relay stream) -> `resample_segment` (to
`config.py`'s `stt_target_sample_rate`) -> `SpeechToText.transcribe`. All
three stages are the existing, already-tested pipeline (`ingest.py`,
`resample.py`, `stt.py`) -- this router only wires them together and puts
the result on the wire, per `docs/11-Engineering-Standards.md`'s "keep
business logic outside FastAPI route handlers."

The relay has no per-session URL (`docs/12-Media-Relay-Contract.md`): every
consumer connects to the same `ws://<gateway>/v1/stream/audio`, and messages
carry `session_id` as a field. `ingest_segments`'s `session_id` parameter
(added in `ingest.py` alongside this endpoint) does that filtering, so two
sessions active on the same relay connection at once never bleed into each
other's transcript stream.

Which `SpeechToText` backend actually transcribes was decided once at
startup (`main.py`'s `_select_stt_backend`), not per connection -- mirrors
`synthesize.py`'s `_select_tts_backend` exactly.

Out of scope here, deliberately: the return-audio transport (blocked --
`MediaClient` has no send method yet), Memory/query wiring, and a shared
speech-contract package. This endpoint only transcribes and streams text
back to whoever opened the socket; it doesn't send anything to Memory or the
gateway.

A disconnect can arrive at any time, including before the first segment is
ever produced -- e.g. a `session_id` that never appears on the relay leaves
`ingest_segments` waiting indefinitely (by design: `MediaClient`'s default
`reconnect=True` is what makes it resilient to the *gateway* blipping, and
there is no separate signal to distinguish "gone for good" from "not started
yet"). The producing loop alone would never notice such a disconnect --
it only reaches a `websocket` call (`send_json`) *after* a segment exists.
So it runs concurrently with a second task that does nothing but call
`websocket.receive()` in a loop -- Starlette's own way of surfacing a
disconnect -- and whichever of the two finishes first cancels the other,
via `anyio.create_task_group`'s cancel scope. That is what makes "the
`MediaClient` must be torn down cleanly on disconnect" (this stage's own
requirement) true regardless of when the disconnect happens, not just when
it happens to land inside a `send_json` call.

Both tasks swallow `WebSocketDisconnect` themselves rather than letting it
escape the task group: `anyio.create_task_group.__aexit__` always reports a
child's exception wrapped in an `ExceptionGroup`, even for one child, so a
plain `except WebSocketDisconnect` around the `async with` would never match
it -- disconnecting is this endpoint's normal, expected way to end, not a
failure to report as one.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from speech.deps import settings_of, stt_backend_of
from speech.ingest import ingest_segments
from speech.resample import resample_segment
from speech.utterance import SileroBoundary

router = APIRouter(tags=["stt"])


@router.websocket("/v1/stt/{session_id}")
async def stt(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    stt_backend = stt_backend_of(websocket)
    settings = settings_of(websocket)

    segments = ingest_segments(
        settings.gateway_audio_url,
        session_id=session_id,
        token=settings.gateway_token,
        # Without this a segment ends only on a gap, an epoch, or the session
        # -- none of which happen while a microphone is publishing, so the
        # socket would stay silent for the whole call and then deliver one
        # enormous transcript. See `utterance.py`.
        boundary=SileroBoundary(
            silence_ms=settings.stt_utterance_silence_ms,
            max_seconds=settings.stt_utterance_max_seconds,
            threshold=settings.stt_vad_threshold,
        ),
        # Must cover the detector's attack delay, or the leading-silence trim
        # takes the first phoneme with it and the wake prefix never matches.
        preroll_seconds=settings.stt_preroll_ms / 1000,
    )
    try:
        async with anyio.create_task_group() as tg:

            async def _produce() -> None:
                try:
                    async for segment in segments:
                        resampled = resample_segment(
                            segment, target_sample_rate=settings.stt_target_sample_rate
                        )
                        transcript = await stt_backend.transcribe(resampled)
                        # A segment can legitimately contain no words -- the
                        # trailing silence flushed when a stream ends is one,
                        # and so is a burst of noise the VAD let through.
                        # Sending it puts an empty line in front of whoever is
                        # reading, which says nothing except that something
                        # arrived.
                        if not transcript.text.strip():
                            continue
                        await websocket.send_json(transcript.model_dump(mode="json"))
                except WebSocketDisconnect:
                    pass
                finally:
                    # Upstream ended (or errored, or the client left) --
                    # nothing left to watch the socket for.
                    tg.cancel_scope.cancel()

            async def _watch_for_disconnect() -> None:
                try:
                    while True:
                        # The raw, low-level `receive()` -- not
                        # `receive_text()`/`receive_json()` -- because
                        # those raise on their own `websocket.disconnect`
                        # check but then assume a `"text"`/`"bytes"` key
                        # exists, which is only true for a message this
                        # endpoint never expects the client to send (it
                        # only ever listens). Checking the type explicitly
                        # tolerates any message the client sends without
                        # crashing on one this endpoint doesn't care about.
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                finally:
                    # Either the client disconnected, or `_produce` finishing
                    # cancelled this task itself -- either way, done.
                    tg.cancel_scope.cancel()

            tg.start_soon(_watch_for_disconnect)
            tg.start_soon(_produce)
    finally:
        # `ingest_segments` owns a `MediaClient` internally (its own
        # `async with` block). Closing the generator explicitly -- rather
        # than relying on it being garbage-collected eventually -- is what
        # tears that connection down promptly, whichever side ended first.
        await segments.aclose()
