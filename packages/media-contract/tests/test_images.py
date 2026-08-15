"""Payload decoding, exercised through the message API consumers actually use."""

import datetime as dt
import io

import numpy as np
import pytest
from PIL import Image

from visual_memory_media_contract.framing import decode_message, encode_message, payload_digest
from visual_memory_media_contract.images import ImageDecodeError, to_rgb, to_rgba
from visual_memory_media_contract.protocol import AudioChunk, VideoFrame

T0 = dt.datetime(2026, 7, 30, 18, 4, 11, 21000, tzinfo=dt.UTC)
WIDTH, HEIGHT = 8, 4


def a_gradient(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A deterministic RGB image with a distinct value in every channel."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(width, dtype=np.uint8) * 8
    image[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 16
    image[:, :, 2] = 128
    return image


def as_jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92, subsampling=0)
    return buffer.getvalue()


def a_video_frame(payload: bytes, *, encoding: str, pixel_format: str) -> VideoFrame:
    return VideoFrame(
        session_id="sess_01JAB000000000000000000",
        epoch_id="TR_VCabc123",
        sequence=0,
        captured_at=T0,
        received_at=T0,
        relayed_at=T0,
        width=WIDTH,
        height=HEIGHT,
        encoding=encoding,  # type: ignore[arg-type]
        pixel_format=pixel_format,  # type: ignore[arg-type]
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
    )


def test_rgba_raw_round_trips_pixel_exactly() -> None:
    original = to_rgba(a_gradient())
    payload = original.tobytes()
    message = a_video_frame(payload, encoding="rgba_raw", pixel_format="rgba")

    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    np.testing.assert_array_equal(decoded.rgba, original)
    np.testing.assert_array_equal(decoded.rgb, original[:, :, :3])


def test_jpeg_decodes_to_declared_shape() -> None:
    payload = as_jpeg(a_gradient())
    message = a_video_frame(payload, encoding="jpeg", pixel_format="rgb")

    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    assert decoded.rgb.shape == (HEIGHT, WIDTH, 3)
    assert decoded.rgb.dtype == np.uint8


def test_jpeg_at_quality_92_is_close_to_the_original() -> None:
    original = a_gradient()
    payload = as_jpeg(original)
    message = a_video_frame(payload, encoding="jpeg", pixel_format="rgb")

    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    # 4:4:4 subsampling keeps chroma edges; a small residual is expected.
    assert np.abs(decoded.rgb.astype(int) - original.astype(int)).max() <= 12


def test_rgba_property_fills_opaque_alpha_for_jpeg() -> None:
    payload = as_jpeg(a_gradient())
    message = a_video_frame(payload, encoding="jpeg", pixel_format="rgb")

    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    assert decoded.rgba.shape == (HEIGHT, WIDTH, 4)
    assert np.all(decoded.rgba[:, :, 3] == 255)


def test_raw_payload_of_wrong_length_is_rejected() -> None:
    payload = bytes(WIDTH * HEIGHT * 4 - 1)
    message = a_video_frame(payload, encoding="rgba_raw", pixel_format="rgba")
    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    with pytest.raises(ImageDecodeError, match="expected"):
        _ = decoded.rgb


def test_jpeg_disagreeing_with_declared_dimensions_is_rejected() -> None:
    payload = as_jpeg(a_gradient(width=WIDTH * 2, height=HEIGHT))
    message = a_video_frame(payload, encoding="jpeg", pixel_format="rgb")
    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, VideoFrame)

    with pytest.raises(ImageDecodeError, match="header declares"):
        _ = decoded.rgb


def test_audio_chunk_exposes_int16_pcm() -> None:
    samples = np.array([0, 1, -1, 32767, -32768, 5, 6, 7], dtype="<i2")
    payload = samples.tobytes()
    message = AudioChunk(
        session_id="sess_01JAB000000000000000000",
        epoch_id="TR_ACdef456",
        sequence=0,
        pts_samples=0,
        samples=len(samples),
        sample_rate=48000,
        channels=1,
        sample_format="s16le",
        first_sample_captured_at=T0,
        payload_bytes=len(payload),
    )

    decoded = decode_message(encode_message(message, payload))
    assert isinstance(decoded, AudioChunk)

    assert decoded.pcm.shape == (len(samples), 1)
    np.testing.assert_array_equal(decoded.pcm[:, 0], samples)


def test_to_rgb_and_to_rgba_are_idempotent() -> None:
    rgb = a_gradient()
    rgba = to_rgba(rgb)

    np.testing.assert_array_equal(to_rgb(to_rgb(rgb)), rgb)
    np.testing.assert_array_equal(to_rgba(rgba), rgba)
    np.testing.assert_array_equal(to_rgb(rgba), rgb)


def test_payload_is_absent_from_the_json_header() -> None:
    payload = as_jpeg(a_gradient())
    message = a_video_frame(payload, encoding="jpeg", pixel_format="rgb")

    frame = encode_message(message, payload)
    header_end = frame.index(b"}", 8) + 1

    assert b"_payload" not in frame[:header_end]
    assert b"payload_bytes" in frame[:header_end]
