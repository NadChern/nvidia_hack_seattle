"""Binary framing for the media relay protocol.

One logical message is one WebSocket binary frame:

    magic       4 bytes   b"VMA1"
    header_len  4 bytes   uint32 big-endian
    header      N bytes   UTF-8 JSON, validates into a RelayMessage
    payload     rest      raw bytes, empty for control messages

Header and payload travel together rather than as two WebSocket messages so a
consumer cannot mis-pair them, and so control messages stay strictly ordered
against media: an `epoch_started` can never be processed after a frame that
belongs to the epoch it starts.
"""

from __future__ import annotations

import hashlib
import struct

from visual_memory_media_contract.protocol import (
    RELAY_MESSAGE_ADAPTER,
    AudioChunk,
    RelayMessage,
    VideoFrame,
)

MAGIC = b"VMA1"
_PREFIX = struct.Struct(">4sI")
PREFIX_BYTES = _PREFIX.size

#: Headers are a few hundred bytes in practice. The cap bounds the damage from a
#: corrupt or hostile length prefix before any allocation happens.
MAX_HEADER_BYTES = 64 * 1024


class FramingError(ValueError):
    """Base class for malformed relay frames."""


class BadMagicError(FramingError):
    """The frame does not start with the protocol magic."""


class TruncatedFrameError(FramingError):
    """The frame ended before the declared header or prefix was complete."""


class HeaderTooLargeError(FramingError):
    """The declared header length exceeds MAX_HEADER_BYTES."""


class PayloadMismatchError(FramingError):
    """Payload length or digest disagrees with the header."""


def payload_digest(payload: bytes) -> str:
    """Return the lowercase hex SHA-256 of a payload."""
    return hashlib.sha256(payload).hexdigest()


def _declared_payload_bytes(message: RelayMessage) -> int | None:
    """Return the payload length the message declares, if it declares one."""
    if isinstance(message, VideoFrame | AudioChunk):
        return message.payload_bytes
    return None


def encode_message(message: RelayMessage, payload: bytes = b"") -> bytes:
    """Serialize a message and its payload into one wire frame.

    Raises PayloadMismatchError when the message declares a payload length that
    disagrees with the bytes supplied, so a producer bug surfaces here rather
    than as silent corruption at the far end.
    """
    declared = _declared_payload_bytes(message)
    if declared is None:
        if payload:
            raise PayloadMismatchError(f"{message.type} does not carry a payload")
    elif declared != len(payload):
        raise PayloadMismatchError(
            f"{message.type} declares payload_bytes={declared} but got {len(payload)}"
        )

    header = message.model_dump_json(exclude_none=True).encode("utf-8")
    if len(header) > MAX_HEADER_BYTES:
        raise HeaderTooLargeError(f"header is {len(header)} bytes, limit is {MAX_HEADER_BYTES}")
    return _PREFIX.pack(MAGIC, len(header)) + header + payload


def decode_message(frame: bytes) -> RelayMessage:
    """Parse one wire frame into a message with its payload attached.

    Verifies the payload length and, for video frames, the SHA-256 digest, so
    corruption surfaces at the boundary rather than as a strange detection
    result several services downstream. Reach the bytes through
    `message.payload`, or the decoded pixels through `message.rgb`.
    """
    if len(frame) < PREFIX_BYTES:
        raise TruncatedFrameError(f"frame is {len(frame)} bytes, need at least {PREFIX_BYTES}")

    magic, header_len = _PREFIX.unpack_from(frame)
    if magic != MAGIC:
        raise BadMagicError(f"expected magic {MAGIC!r}, got {magic!r}")
    if header_len > MAX_HEADER_BYTES:
        raise HeaderTooLargeError(
            f"header declares {header_len} bytes, limit is {MAX_HEADER_BYTES}"
        )

    header_end = PREFIX_BYTES + header_len
    if len(frame) < header_end:
        raise TruncatedFrameError(f"header declares {header_len} bytes, frame holds {len(frame)}")

    message = RELAY_MESSAGE_ADAPTER.validate_json(frame[PREFIX_BYTES:header_end])
    payload = frame[header_end:]

    declared = _declared_payload_bytes(message)
    if declared is None:
        if payload:
            raise PayloadMismatchError(f"{message.type} carries {len(payload)} unexpected bytes")
    elif declared != len(payload):
        raise PayloadMismatchError(
            f"{message.type} declares payload_bytes={declared} but frame holds {len(payload)}"
        )

    if isinstance(message, VideoFrame):
        actual = payload_digest(payload)
        if actual != message.sha256:
            raise PayloadMismatchError(
                f"video_frame digest mismatch: header {message.sha256}, payload {actual}"
            )

    if isinstance(message, VideoFrame | AudioChunk):
        return message.attach_payload(payload)
    return message


__all__ = [
    "MAGIC",
    "MAX_HEADER_BYTES",
    "PREFIX_BYTES",
    "BadMagicError",
    "FramingError",
    "HeaderTooLargeError",
    "PayloadMismatchError",
    "TruncatedFrameError",
    "decode_message",
    "encode_message",
    "payload_digest",
]
