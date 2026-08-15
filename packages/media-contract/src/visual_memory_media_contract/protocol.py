"""Message models for the media relay protocol.

`docs/12-Media-Relay-Contract.md` is the normative definition. These models are
the executable form of it and the single source of truth for field names.

Every message is a Pydantic model discriminated on `type`. Models are frozen
and ignore unknown fields, so a consumer pinned to an older minor version keeps
working when the gateway starts emitting an additional field.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Annotated, Literal, Self, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    TypeAdapter,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

#: Bumped for any change to the wire format. Additive optional fields are a
#: minor bump; removals, renames, and semantic changes are a major bump.
PROTOCOL_VERSION = "media-relay/1.0"


def _utc_iso(value: dt.datetime) -> str:
    """Render a timestamp as UTC ISO-8601 with a Z suffix and millisecond precision.

    `docs/01-Recommended-Architecture.md` requires UTC ISO-8601 everywhere. Fixed
    precision keeps golden fixtures byte-stable.
    """
    return value.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


UtcTimestamp: TypeAlias = Annotated[AwareDatetime, PlainSerializer(_utc_iso, return_type=str)]

StreamKind: TypeAlias = Literal["video", "audio"]
VideoEncoding: TypeAlias = Literal["jpeg", "rgba_raw"]
PixelFormat: TypeAlias = Literal["rgb", "rgba"]
SampleFormat: TypeAlias = Literal["s16le"]

EpochEndReason: TypeAlias = Literal[
    "track_unsubscribed",
    "participant_disconnected",
    "room_disconnected",
    "session_ended",
    "gateway_shutdown",
]
SessionEndReason: TypeAlias = Literal[
    "participant_disconnected",
    "room_disconnected",
    "session_deleted",
    "session_ttl_expired",
    "gateway_shutdown",
]
LifecycleAction: TypeAlias = Literal["track_lost", "session_ended"]


class _Message(BaseModel):
    """Shared configuration for every relay message."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    protocol_version: str = PROTOCOL_VERSION


class _PayloadMessage(_Message):
    """A message whose binary payload travels in the same wire frame.

    The payload is a private attribute so it stays out of the serialized JSON
    header while remaining reachable through `.payload` and the decoding
    helpers. Pydantic does compare private attributes, so two messages are
    equal only when their payloads match as well.
    """

    _payload: bytes = PrivateAttr(default=b"")

    @property
    def payload(self) -> bytes:
        """Raw payload bytes, empty when the message was built by hand."""
        return self._payload

    def attach_payload(self, payload: bytes) -> Self:
        """Return a copy of this message carrying `payload`."""
        carried = self.model_copy()
        carried._payload = payload
        return carried


class ActiveEpoch(BaseModel):
    """An epoch already in progress when a subscriber connects."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    session_id: str
    epoch_id: str
    stream_kind: StreamKind
    started_at: UtcTimestamp


class StreamHello(_Message):
    """First message on every stream. Describes what is already in flight."""

    type: Literal["stream_hello"] = "stream_hello"
    gateway_version: str
    stream_kind: StreamKind
    encoding: VideoEncoding | None = None
    active_sessions: list[str] = Field(default_factory=list[str])
    active_epochs: list[ActiveEpoch] = Field(default_factory=list[ActiveEpoch])


class SessionStarted(_Message):
    type: Literal["session_started"] = "session_started"
    session_id: str
    device_id: str
    started_at: UtcTimestamp


class SessionEnded(_Message):
    type: Literal["session_ended"] = "session_ended"
    session_id: str
    ended_at: UtcTimestamp
    reason: SessionEndReason


class EpochStarted(_Message):
    """A new media epoch. Consumers MUST reset per-track state on this message.

    `epoch_id` is the LiveKit track SID. The S01 spike established that a rejoin
    produces a new track SID even when the participant identity is unchanged, so
    the SID -- not the identity -- is the media-epoch boundary.
    """

    type: Literal["epoch_started"] = "epoch_started"
    session_id: str
    epoch_id: str
    stream_kind: StreamKind
    track_sid: str
    participant_identity: str
    started_at: UtcTimestamp
    width: int | None = None
    height: int | None = None
    encoding: VideoEncoding | None = None
    pixel_format: PixelFormat | None = None
    sample_rate: int | None = None
    channels: int | None = None


class EpochEnded(_Message):
    type: Literal["epoch_ended"] = "epoch_ended"
    session_id: str
    epoch_id: str
    ended_at: UtcTimestamp
    reason: EpochEndReason


class VideoFrame(_PayloadMessage):
    """One sampled, dimension-guarded video frame. Payload is the encoded image."""

    type: Literal["video_frame"] = "video_frame"
    session_id: str
    epoch_id: str
    sequence: int
    captured_at: UtcTimestamp
    received_at: UtcTimestamp
    relayed_at: UtcTimestamp
    width: int
    height: int
    encoding: VideoEncoding
    pixel_format: PixelFormat
    payload_bytes: int
    sha256: str
    dropped_since_previous: int = 0

    def decode(self) -> NDArray[np.uint8]:
        """Decode the payload at its natural channel count."""
        from visual_memory_media_contract.images import decode_video_payload

        return decode_video_payload(
            self._payload,
            encoding=self.encoding,
            width=self.width,
            height=self.height,
            pixel_format=self.pixel_format,
        )

    @property
    def rgb(self) -> NDArray[np.uint8]:
        """Frame as an `(H, W, 3)` uint8 array."""
        from visual_memory_media_contract.images import to_rgb

        return to_rgb(self.decode())

    @property
    def rgba(self) -> NDArray[np.uint8]:
        """Frame as an `(H, W, 4)` uint8 array, alpha filled opaque if needed."""
        from visual_memory_media_contract.images import to_rgba

        return to_rgba(self.decode())


class AudioChunk(_PayloadMessage):
    """Coalesced PCM audio. Payload is raw interleaved samples, never dropped.

    `pts_samples` is the cumulative sample count since the epoch started, so a
    consumer can detect a gap arithmetically rather than by guessing.
    """

    type: Literal["audio_chunk"] = "audio_chunk"
    session_id: str
    epoch_id: str
    sequence: int
    pts_samples: int
    samples: int
    sample_rate: int
    channels: int
    sample_format: SampleFormat
    first_sample_captured_at: UtcTimestamp
    payload_bytes: int

    @property
    def pcm(self) -> NDArray[np.int16]:
        """Payload as an `(samples, channels)` int16 array."""
        import numpy as np

        flat = np.frombuffer(self._payload, dtype="<i2")
        return flat.reshape(-1, self.channels)


class LifecycleScope(BaseModel):
    """Blast radius of a lifecycle signal.

    A gateway cannot know about objects, so it scopes by media epoch: the
    transition applies to every object whose in-transit state originated there.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    media_epoch_id: str | None = None
    object_id: str | None = None
    track_id: str | None = None


class LifecycleDetail(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    action: LifecycleAction
    source: str = "media_gateway"
    occurred_at: UtcTimestamp
    reason: EpochEndReason | SessionEndReason


class LifecycleProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    component: str = "media-gateway"
    version: str
    protocol_version: str = PROTOCOL_VERSION


class LifecycleEnvelope(BaseModel):
    """What the gateway posts to the Memory Service.

    Deliberately not an observation: it carries no object, location, confidence,
    or evidence, because the gateway observes none of those. Pending Person 3
    sign-off on the corresponding `docs/06-Data-Contract.md` amendment.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: str = "1.0"
    signal_id: str
    idempotency_key: str
    session_id: str
    device_id: str
    signal: LifecycleDetail
    scope: LifecycleScope
    provenance: LifecycleProvenance


class LifecycleSignal(_Message):
    """In-band copy of a lifecycle envelope, ordered against the media stream."""

    type: Literal["lifecycle_signal"] = "lifecycle_signal"
    envelope: LifecycleEnvelope


class Keepalive(_Message):
    """Emitted while idle so a consumer can tell "no publisher" from "dead socket"."""

    type: Literal["keepalive"] = "keepalive"
    sent_at: UtcTimestamp


RelayMessage: TypeAlias = Annotated[
    StreamHello
    | SessionStarted
    | SessionEnded
    | EpochStarted
    | EpochEnded
    | VideoFrame
    | AudioChunk
    | LifecycleSignal
    | Keepalive,
    Field(discriminator="type"),
]

RELAY_MESSAGE_ADAPTER: TypeAdapter[RelayMessage] = TypeAdapter(RelayMessage)

#: Message types that carry a non-empty binary payload.
PAYLOAD_MESSAGE_TYPES = frozenset({"video_frame", "audio_chunk"})

__all__ = [
    "PAYLOAD_MESSAGE_TYPES",
    "RELAY_MESSAGE_ADAPTER",
    "ActiveEpoch",
    "AudioChunk",
    "EpochEndReason",
    "EpochEnded",
    "EpochStarted",
    "Keepalive",
    "LifecycleAction",
    "LifecycleDetail",
    "LifecycleEnvelope",
    "LifecycleProvenance",
    "LifecycleScope",
    "LifecycleSignal",
    "PixelFormat",
    "RelayMessage",
    "SampleFormat",
    "SessionEndReason",
    "SessionEnded",
    "SessionStarted",
    "StreamHello",
    "StreamKind",
    "UtcTimestamp",
    "VideoEncoding",
    "VideoFrame",
]
