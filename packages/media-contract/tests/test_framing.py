import datetime as dt
import struct

import pytest

from visual_memory_media_contract.framing import (
    MAGIC,
    MAX_HEADER_BYTES,
    PREFIX_BYTES,
    BadMagicError,
    HeaderTooLargeError,
    PayloadMismatchError,
    TruncatedFrameError,
    decode_message,
    encode_message,
    payload_digest,
)
from visual_memory_media_contract.protocol import (
    ActiveEpoch,
    AudioChunk,
    EpochEnded,
    EpochStarted,
    Keepalive,
    LifecycleDetail,
    LifecycleEnvelope,
    LifecycleProvenance,
    LifecycleScope,
    LifecycleSignal,
    RelayMessage,
    SessionEnded,
    SessionStarted,
    StreamHello,
    VideoFrame,
)

T0 = dt.datetime(2026, 7, 30, 18, 4, 11, 21000, tzinfo=dt.UTC)
SESSION = "sess_01JAB000000000000000000"
EPOCH = "TR_VCabc123"


def a_video_frame(payload: bytes, *, sequence: int = 0) -> VideoFrame:
    return VideoFrame(
        session_id=SESSION,
        epoch_id=EPOCH,
        sequence=sequence,
        captured_at=T0,
        received_at=T0,
        relayed_at=T0,
        width=320,
        height=180,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
    )


def an_audio_chunk(payload: bytes) -> AudioChunk:
    return AudioChunk(
        session_id=SESSION,
        epoch_id="TR_ACdef456",
        sequence=7,
        pts_samples=33600,
        samples=len(payload) // 2,
        sample_rate=48000,
        channels=1,
        sample_format="s16le",
        first_sample_captured_at=T0,
        payload_bytes=len(payload),
    )


def a_lifecycle_signal() -> LifecycleSignal:
    return LifecycleSignal(
        envelope=LifecycleEnvelope(
            signal_id="lc_01JABC0000000000000000000",
            idempotency_key=f"glasses-01/{SESSION}/{EPOCH}/track_lost",
            session_id=SESSION,
            device_id="glasses-01",
            signal=LifecycleDetail(
                action="track_lost",
                occurred_at=T0,
                reason="track_unsubscribed",
            ),
            scope=LifecycleScope(media_epoch_id=EPOCH),
            provenance=LifecycleProvenance(version="0.1.0"),
        )
    )


CONTROL_MESSAGES: list[RelayMessage] = [
    StreamHello(
        gateway_version="0.1.0",
        stream_kind="video",
        encoding="jpeg",
        active_sessions=[SESSION],
        active_epochs=[
            ActiveEpoch(session_id=SESSION, epoch_id=EPOCH, stream_kind="video", started_at=T0)
        ],
    ),
    SessionStarted(session_id=SESSION, device_id="glasses-01", started_at=T0),
    SessionEnded(session_id=SESSION, ended_at=T0, reason="session_deleted"),
    EpochStarted(
        session_id=SESSION,
        epoch_id=EPOCH,
        stream_kind="video",
        track_sid=EPOCH,
        participant_identity="glasses-01",
        started_at=T0,
        width=320,
        height=180,
        encoding="jpeg",
        pixel_format="rgb",
    ),
    EpochEnded(session_id=SESSION, epoch_id=EPOCH, ended_at=T0, reason="track_unsubscribed"),
    a_lifecycle_signal(),
    Keepalive(sent_at=T0),
]


@pytest.mark.parametrize("message", CONTROL_MESSAGES, ids=lambda m: m.type)
def test_control_messages_round_trip_without_payload(message: RelayMessage) -> None:
    decoded = decode_message(encode_message(message))

    assert decoded == message


def test_video_frame_round_trips_with_payload() -> None:
    payload = b"\xff\xd8\xff\xe0 not really a jpeg"
    message = a_video_frame(payload)

    decoded = decode_message(encode_message(message, payload))

    assert decoded.payload == payload
    assert decoded == message.attach_payload(payload)


def test_audio_chunk_round_trips_with_payload() -> None:
    payload = bytes(9600)
    message = an_audio_chunk(payload)

    decoded = decode_message(encode_message(message, payload))

    assert decoded.payload == payload
    assert decoded == message.attach_payload(payload)


def test_timestamps_serialize_as_utc_z() -> None:
    frame = encode_message(Keepalive(sent_at=T0))

    assert b'"sent_at":"2026-07-30T18:04:11.021Z"' in frame


def test_non_utc_timestamp_is_normalized_to_utc() -> None:
    tokyo = dt.timezone(dt.timedelta(hours=9))
    message = Keepalive(sent_at=T0.astimezone(tokyo))

    assert b'"sent_at":"2026-07-30T18:04:11.021Z"' in encode_message(message)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError):
        Keepalive(sent_at=dt.datetime(2026, 7, 30, 18, 4, 11))  # noqa: DTZ001


def test_unknown_header_fields_are_ignored_for_forward_compatibility() -> None:
    header = b'{"type":"keepalive","protocol_version":"media-relay/1.0",'
    header += b'"sent_at":"2026-07-30T18:04:11.021Z","future_field":123}'
    frame = struct.pack(">4sI", MAGIC, len(header)) + header

    decoded = decode_message(frame)

    assert decoded.type == "keepalive"


def test_control_message_with_payload_is_rejected_on_encode() -> None:
    with pytest.raises(PayloadMismatchError):
        encode_message(Keepalive(sent_at=T0), b"unexpected")


def test_declared_length_mismatch_is_rejected_on_encode() -> None:
    payload = b"twenty-four bytes long!!"
    message = a_video_frame(payload)

    with pytest.raises(PayloadMismatchError):
        encode_message(message, payload + b"extra")


def test_corrupted_payload_fails_the_digest_check() -> None:
    payload = b"original bytes"
    frame = bytearray(encode_message(a_video_frame(payload), payload))
    frame[-1] ^= 0xFF

    with pytest.raises(PayloadMismatchError, match="digest mismatch"):
        decode_message(bytes(frame))


def test_bad_magic_is_rejected() -> None:
    frame = bytearray(encode_message(Keepalive(sent_at=T0)))
    frame[0:4] = b"XXXX"

    with pytest.raises(BadMagicError):
        decode_message(bytes(frame))


def test_short_frame_is_rejected() -> None:
    with pytest.raises(TruncatedFrameError):
        decode_message(b"VMA")


def test_truncated_header_is_rejected() -> None:
    frame = encode_message(Keepalive(sent_at=T0))

    with pytest.raises(TruncatedFrameError):
        decode_message(frame[:-5])


def test_oversized_declared_header_is_rejected_before_allocation() -> None:
    frame = struct.pack(">4sI", MAGIC, MAX_HEADER_BYTES + 1)

    with pytest.raises(HeaderTooLargeError):
        decode_message(frame)


def test_prefix_is_eight_bytes() -> None:
    assert PREFIX_BYTES == 8
    assert encode_message(Keepalive(sent_at=T0))[:4] == MAGIC
