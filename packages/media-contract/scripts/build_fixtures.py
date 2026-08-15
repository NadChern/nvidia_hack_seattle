#!/usr/bin/env python3
"""Regenerate the contract fixtures.

    uv run python scripts/build_fixtures.py

Fixtures are deterministic: a fixed clock, a fixed image pattern, and fixed
ids, so regenerating without a protocol change produces byte-identical files
and `git diff` stays empty.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from visual_memory_media_contract.fixtures import (  # noqa: E402
    FIXTURES_DIR,
    at,
    build_frames,
    write_fixture,
)
from visual_memory_media_contract.framing import payload_digest  # noqa: E402
from visual_memory_media_contract.protocol import (  # noqa: E402
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

DEVICE = "glasses-01"
SESSION = "sess_01JAB000000000000000000"
VIDEO_EPOCH_1 = "TR_VCaaaaaaaaaaaa"
VIDEO_EPOCH_2 = "TR_VCbbbbbbbbbbbb"
AUDIO_EPOCH = "TR_ACcccccccccccc"
GATEWAY_VERSION = "0.1.0"
WIDTH, HEIGHT = 320, 180
SAMPLE_RATE = 48_000
CHUNK_SAMPLES = 4_800  # 100 ms at 48 kHz


def frame_image(sequence: int) -> bytes:
    """A deterministic frame whose content changes with the sequence number."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = (np.arange(WIDTH, dtype=np.uint16) + sequence * 7).astype(np.uint8)
    image[:, :, 1] = np.arange(HEIGHT, dtype=np.uint16)[:, None].astype(np.uint8)
    image[:, :, 2] = np.uint8((sequence * 17) % 256)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92, subsampling=0)
    return buffer.getvalue()


def video_frame(
    *, epoch_id: str, sequence: int, offset_ms: int, dropped: int = 0
) -> tuple[RelayMessage, bytes]:
    payload = frame_image(sequence)
    message = VideoFrame(
        session_id=SESSION,
        epoch_id=epoch_id,
        sequence=sequence,
        captured_at=at(offset_ms),
        received_at=at(offset_ms + 8),
        relayed_at=at(offset_ms + 13),
        width=WIDTH,
        height=HEIGHT,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
        dropped_since_previous=dropped,
    )
    return message, payload


def build_video_session_basic() -> list[bytes]:
    """Hello, a session, two epochs across a rejoin, a lifecycle signal, end.

    The epoch change is the important part: a consumer that fails to reset
    tracker state on `epoch_started` will carry identities across the rejoin.
    """
    messages: list[tuple[RelayMessage, bytes]] = [
        (StreamHello(gateway_version=GATEWAY_VERSION, stream_kind="video", encoding="jpeg"), b""),
        (SessionStarted(session_id=SESSION, device_id=DEVICE, started_at=at(0)), b""),
        (
            EpochStarted(
                session_id=SESSION,
                epoch_id=VIDEO_EPOCH_1,
                stream_kind="video",
                track_sid=VIDEO_EPOCH_1,
                participant_identity=DEVICE,
                started_at=at(120),
                width=WIDTH,
                height=HEIGHT,
                encoding="jpeg",
                pixel_format="rgb",
            ),
            b"",
        ),
    ]

    for sequence in range(6):
        # A dropped frame partway through proves the latest-wins slot is
        # reported rather than hidden.
        dropped = 2 if sequence == 3 else 0
        messages.append(
            video_frame(
                epoch_id=VIDEO_EPOCH_1,
                sequence=sequence,
                offset_ms=500 + sequence * 500,
                dropped=dropped,
            )
        )

    messages.append(
        (
            EpochEnded(
                session_id=SESSION,
                epoch_id=VIDEO_EPOCH_1,
                ended_at=at(3600),
                reason="track_unsubscribed",
            ),
            b"",
        )
    )
    messages.append(
        (
            LifecycleSignal(
                envelope=LifecycleEnvelope(
                    signal_id="lc_01JABC0000000000000000001",
                    idempotency_key=f"{DEVICE}/{SESSION}/{VIDEO_EPOCH_1}/track_lost",
                    session_id=SESSION,
                    device_id=DEVICE,
                    signal=LifecycleDetail(
                        action="track_lost",
                        occurred_at=at(3600),
                        reason="track_unsubscribed",
                    ),
                    scope=LifecycleScope(media_epoch_id=VIDEO_EPOCH_1),
                    provenance=LifecycleProvenance(version=GATEWAY_VERSION),
                )
            ),
            b"",
        )
    )
    messages.append((Keepalive(sent_at=at(4000)), b""))

    # Same participant identity, new track SID: a rejoin, and a new epoch.
    messages.append(
        (
            EpochStarted(
                session_id=SESSION,
                epoch_id=VIDEO_EPOCH_2,
                stream_kind="video",
                track_sid=VIDEO_EPOCH_2,
                participant_identity=DEVICE,
                started_at=at(4500),
                width=WIDTH,
                height=HEIGHT,
                encoding="jpeg",
                pixel_format="rgb",
            ),
            b"",
        )
    )
    for sequence in range(3):
        messages.append(
            video_frame(epoch_id=VIDEO_EPOCH_2, sequence=sequence, offset_ms=5000 + sequence * 500)
        )

    messages.append(
        (
            EpochEnded(
                session_id=SESSION,
                epoch_id=VIDEO_EPOCH_2,
                ended_at=at(6800),
                reason="session_ended",
            ),
            b"",
        )
    )
    messages.append(
        (
            SessionEnded(session_id=SESSION, ended_at=at(6900), reason="session_deleted"),
            b"",
        )
    )
    return build_frames(messages)


def build_audio_session_basic() -> list[bytes]:
    """Three seconds of 48 kHz mono with one deliberate gap.

    `pts_samples` jumps across the gap while `sequence` stays contiguous, so a
    consumer must use pts, not message counting, to detect lost audio.
    """
    messages: list[tuple[RelayMessage, bytes]] = [
        (StreamHello(gateway_version=GATEWAY_VERSION, stream_kind="audio"), b""),
        (SessionStarted(session_id=SESSION, device_id=DEVICE, started_at=at(0)), b""),
        (
            EpochStarted(
                session_id=SESSION,
                epoch_id=AUDIO_EPOCH,
                stream_kind="audio",
                track_sid=AUDIO_EPOCH,
                participant_identity=DEVICE,
                started_at=at(120),
                sample_rate=SAMPLE_RATE,
                channels=1,
            ),
            b"",
        ),
    ]

    gap_after = 10
    pts = 0
    for sequence in range(30):
        if sequence == gap_after:
            pts += CHUNK_SAMPLES * 5  # 500 ms of audio never arrived
        time = np.arange(pts, pts + CHUNK_SAMPLES, dtype=np.float64) / SAMPLE_RATE
        tone = np.sin(2.0 * np.pi * 440.0 * time) * 0.18 * 32767.0
        payload = tone.astype("<i2").tobytes()
        messages.append(
            (
                AudioChunk(
                    session_id=SESSION,
                    epoch_id=AUDIO_EPOCH,
                    sequence=sequence,
                    pts_samples=pts,
                    samples=CHUNK_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    channels=1,
                    sample_format="s16le",
                    first_sample_captured_at=at(200 + sequence * 100),
                    payload_bytes=len(payload),
                ),
                payload,
            )
        )
        pts += CHUNK_SAMPLES

    messages.append(
        (
            EpochEnded(
                session_id=SESSION,
                epoch_id=AUDIO_EPOCH,
                ended_at=at(3400),
                reason="session_ended",
            ),
            b"",
        )
    )
    messages.append(
        (SessionEnded(session_id=SESSION, ended_at=at(3500), reason="session_deleted"), b"")
    )
    return build_frames(messages)


def main() -> int:
    builders = {
        "video_session_basic": build_video_session_basic,
        "audio_session_basic": build_audio_session_basic,
    }
    for name, builder in builders.items():
        frames = builder()
        path = FIXTURES_DIR / f"{name}.bin"
        write_fixture(path, frames)
        total = sum(len(frame) for frame in frames)
        print(f"{name}: {len(frames)} messages, {total:,} bytes -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
