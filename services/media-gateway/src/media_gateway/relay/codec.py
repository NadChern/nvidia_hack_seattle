"""Encode decoded frames for the relay.

JPEG at quality 92 with 4:4:4 subsampling is the default. At 320x180 raw RGBA
would be harmless, but real glasses at 720p are 3.7 MB per frame and buffering
that for several subscribers is pure waste. 4:4:4 keeps chroma edges intact for
segmentation, and `image/jpeg` is already the canonical evidence media type in
docs/06-Data-Contract.md.

`rgba_raw` passes the decoded buffer through untouched, for pixel-exact work.
"""

from __future__ import annotations

import io

from PIL import Image
from visual_memory_media_contract.protocol import PixelFormat, VideoEncoding

RGBA_CHANNELS = 4


class EncodeError(ValueError):
    """A frame could not be encoded."""


def encode_video(
    rgba: bytes,
    *,
    width: int,
    height: int,
    encoding: VideoEncoding,
    quality: int = 92,
    subsampling: int = 0,
) -> tuple[bytes, PixelFormat]:
    """Encode a decoded RGBA buffer, returning the payload and its format."""
    expected = width * height * RGBA_CHANNELS
    if len(rgba) != expected:
        raise EncodeError(
            f"buffer is {len(rgba)} bytes, expected {expected} for {width}x{height} RGBA"
        )

    if encoding == "rgba_raw":
        return rgba, "rgba"

    image = Image.frombuffer("RGBA", (width, height), rgba, "raw", "RGBA", 0, 1)
    buffer = io.BytesIO()
    # LiveKit camera video is opaque, so dropping alpha loses nothing.
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
    return buffer.getvalue(), "rgb"


__all__ = ["RGBA_CHANNELS", "EncodeError", "encode_video"]
