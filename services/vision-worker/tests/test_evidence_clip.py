"""evidence/clip.py: encoding a confirmed candidate's window into an mp4,
and the still-frame fallback."""

from __future__ import annotations

import datetime as dt
import io

import av
import numpy as np
import pytest
from PIL import Image

from vision_worker.evidence.clip import (
    CLIP_MEDIA_TYPE,
    ClipEncodeError,
    encode_clip,
    select_still_frame,
)
from vision_worker.evidence.ring import BufferedFrame

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)


def a_jpeg_frame(
    offset_seconds: float, *, width: int = 32, height: int = 24, shade: int = 0
) -> BufferedFrame:
    array = np.full((height, width, 3), shade, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG")
    return BufferedFrame(
        captured_at=T0 + dt.timedelta(seconds=offset_seconds),
        payload=buffer.getvalue(),
        width=width,
        height=height,
    )


def test_encoding_a_window_produces_a_playable_clip_with_the_right_frame_count() -> None:
    frames = [a_jpeg_frame(i, shade=i * 30) for i in range(5)]

    clip_bytes = encode_clip(frames, fps=24.0)

    assert len(clip_bytes) > 0
    container = av.open(io.BytesIO(clip_bytes))
    try:
        decoded = list(container.decode(video=0))
    finally:
        container.close()
    assert len(decoded) == 5


def test_an_empty_window_is_refused() -> None:
    with pytest.raises(ClipEncodeError, match="empty"):
        encode_clip([], fps=24.0)


def test_mismatched_frame_dimensions_are_refused() -> None:
    frames = [a_jpeg_frame(0, width=32, height=24), a_jpeg_frame(1, width=64, height=48)]

    with pytest.raises(ClipEncodeError, match="mixed frame sizes"):
        encode_clip(frames, fps=24.0)


def test_select_still_frame_returns_the_last_one() -> None:
    frames = [a_jpeg_frame(0), a_jpeg_frame(1), a_jpeg_frame(2)]

    still = select_still_frame(frames)

    assert still is frames[-1]


def test_select_still_frame_on_an_empty_window_is_refused() -> None:
    with pytest.raises(ClipEncodeError, match="empty"):
        select_still_frame([])


def test_the_clip_media_type_matches_the_memory_contract_field() -> None:
    assert CLIP_MEDIA_TYPE == "video/mp4"
