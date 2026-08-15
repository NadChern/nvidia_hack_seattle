"""The media ingress boundary.

Everything downstream of a `MediaSource` is written against these types rather
than against LiveKit, which is what lets the entire relay, epoch, metrics, and
WebSocket path run in CI with no server. `ScriptedMediaSource` is the other
implementation of the same boundary.

Sink callbacks are synchronous on purpose. The ingest path only ever puts into
a bounded slot or queue, so introducing await points would add cancellation
edge cases for no benefit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

from visual_memory_media_contract.protocol import (
    EpochEndReason,
    SessionEndReason,
    StreamKind,
)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class RawVideoFrame:
    """One decoded frame, before the dimension guard has seen it."""

    width: int
    height: int
    #: Tightly packed RGBA, `width * height * 4` bytes.
    rgba: bytes
    captured_at: dt.datetime


@dataclass(frozen=True, slots=True)
class RawAudioFrame:
    """One decoded audio frame: interleaved little-endian int16."""

    pcm: bytes
    samples: int
    sample_rate: int
    channels: int
    captured_at: dt.datetime


class MediaSink(Protocol):
    """What a source reports into."""

    def session_started(self, *, session_id: str, device_id: str) -> None: ...

    def epoch_started(
        self,
        *,
        session_id: str,
        stream_kind: StreamKind,
        track_sid: str,
        participant_identity: str,
    ) -> None: ...

    def video_frame(self, *, session_id: str, frame: RawVideoFrame) -> None: ...

    def audio_frame(self, *, session_id: str, frame: RawAudioFrame) -> None: ...

    def epoch_ended(
        self, *, session_id: str, stream_kind: StreamKind, reason: EpochEndReason
    ) -> None: ...

    def session_ended(self, *, session_id: str, reason: SessionEndReason) -> None: ...


class MediaSource(Protocol):
    """Produces media into a sink until stopped."""

    async def run(self, sink: MediaSink) -> None: ...

    async def aclose(self) -> None: ...
