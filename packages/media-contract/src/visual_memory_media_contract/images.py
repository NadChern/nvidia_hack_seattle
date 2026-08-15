"""Decode relay video payloads into numpy arrays.

Pillow is an optional dependency. The Speech Service consumes the same relay
and has no use for an image library, so decoding lives behind an extra:

    uv add "visual-memory-media-contract[images]"

`rgba_raw` payloads decode with numpy alone and need no extra.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

RgbArray = "NDArray[np.uint8]"


class ImageDecodeError(RuntimeError):
    """A video payload could not be decoded."""


class MissingImageDependencyError(ImageDecodeError):
    """Pillow is required to decode this payload but is not installed."""


def decode_video_payload(
    payload: bytes,
    *,
    encoding: str,
    width: int,
    height: int,
    pixel_format: str,
) -> NDArray[np.uint8]:
    """Decode a relay video payload to an ``(H, W, C)`` uint8 array.

    Returns the frame's natural channel count: 3 for JPEG, 4 for ``rgba_raw``.
    Callers wanting a guaranteed layout should use ``VideoFrame.rgb`` or
    ``VideoFrame.rgba`` instead.
    """
    if encoding == "rgba_raw":
        return _decode_raw(payload, width=width, height=height, pixel_format=pixel_format)
    if encoding == "jpeg":
        return _decode_jpeg(payload, width=width, height=height)
    raise ImageDecodeError(f"unsupported encoding {encoding!r}")


def _decode_raw(payload: bytes, *, width: int, height: int, pixel_format: str) -> NDArray[np.uint8]:
    channels = 4 if pixel_format == "rgba" else 3
    expected = width * height * channels
    if len(payload) != expected:
        raise ImageDecodeError(
            f"rgba_raw payload is {len(payload)} bytes, "
            f"expected {expected} for {width}x{height}x{channels}"
        )
    array = np.frombuffer(payload, dtype=np.uint8)
    return array.reshape(height, width, channels)


def _decode_jpeg(payload: bytes, *, width: int, height: int) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by extras
        raise MissingImageDependencyError(
            "decoding jpeg payloads requires Pillow; install visual-memory-media-contract[images]"
        ) from exc

    with Image.open(io.BytesIO(payload)) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)

    if decoded.shape[:2] != (height, width):
        raise ImageDecodeError(
            f"jpeg decoded to {decoded.shape[1]}x{decoded.shape[0]}, "
            f"header declares {width}x{height}"
        )
    return decoded


def to_rgb(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Drop an alpha plane if present."""
    if array.shape[2] == 3:
        return array
    return np.ascontiguousarray(array[:, :, :3])


def to_rgba(array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Append an opaque alpha plane if absent.

    LiveKit camera video is always opaque, so JPEG loses nothing by dropping
    alpha and this restores the four-channel layout callers may expect.
    """
    if array.shape[2] == 4:
        return array
    height, width, _ = array.shape
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = array
    rgba[:, :, 3] = 255
    return rgba


__all__ = [
    "ImageDecodeError",
    "MissingImageDependencyError",
    "decode_video_payload",
    "to_rgb",
    "to_rgba",
]
