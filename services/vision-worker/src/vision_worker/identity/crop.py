"""The single mask-to-crop transform shared by enrollment and matching."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from visual_memory_vision_contract.protocol import BoundingBox

from vision_worker.identity.base import MaskedCrop

_CONTEXT_SCALE = 1.15
_BACKGROUND = 127


def box_to_mask(
    box: BoundingBox, height: int, width: int, *, padding: float = 0.0
) -> NDArray[np.bool_]:
    """A rectangular boolean mask for `box`, expanded by `padding` on each side.

    The rectangle is a **measured decision, not a shortcut awaiting a segmenter.**
    The enrollment-redesign spikes tested real SAM2 masks against this padded box
    and masking lost: it made identity *worse* on the probe set (F1 0.898 vs
    0.941), and on a same-class twin it collapsed ranking 5/5 -> 1/5, because
    masking two near-identical keyrings leaves two identical silhouettes and the
    silhouette is the part they share. See `docs/19-Post-Spike-Build-Plan.md`
    (do-not-build list, spikes 1 and 8). The real identity gap was pixels on
    target, not background.

    So both enrollment and matching treat the reasoner's box as the object
    region: this mask is the padded box itself, `prepare_masked_crop` grays
    everything outside it and pools over it, and the same rule is applied on both
    sides so crop-parity holds. Do not replace this with a segmenter to "finish"
    it -- that regresses identity.
    """
    pad_x = (box.x_max - box.x_min) * padding
    pad_y = (box.y_max - box.y_min) * padding
    x0 = int(np.floor(max(0.0, box.x_min - pad_x) * width))
    y0 = int(np.floor(max(0.0, box.y_min - pad_y) * height))
    x1 = int(np.ceil(min(1.0, box.x_max + pad_x) * width))
    y1 = int(np.ceil(min(1.0, box.y_max + pad_y) * height))
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[y0:y1, x0:x1] = True
    return mask


def encode_jpeg(image: NDArray[np.uint8]) -> bytes:
    """Encode an RGB crop for a model adapter without persisting it."""
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=92)
    return output.getvalue()


def prepare_masked_crop(
    frame_rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    box: BoundingBox,
    *,
    output_size: int = 512,
    context_scale: float = _CONTEXT_SCALE,
) -> MaskedCrop:
    """Mask, add fixed context margin, square-pad, and resize.

    The exact function is called by both registration and matching. A second
    implementation with slightly different padding would create a domain gap
    no cosine threshold can repair.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape HxWx3")
    height, width = frame_rgb.shape[:2]
    if mask.shape != (height, width):
        raise ValueError("mask must have the same HxW shape as frame_rgb")
    if output_size < 16:
        raise ValueError("output_size must be at least 16")
    if context_scale < 1.0:
        raise ValueError("context_scale must be at least 1.0")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("cannot crop an empty mask")

    mask_x1, mask_x2 = int(xs.min()), int(xs.max()) + 1
    mask_y1, mask_y2 = int(ys.min()), int(ys.max()) + 1
    box_x1 = round(max(0.0, min(1.0, box.x_min)) * width)
    box_y1 = round(max(0.0, min(1.0, box.y_min)) * height)
    box_x2 = round(max(0.0, min(1.0, box.x_max)) * width)
    box_y2 = round(max(0.0, min(1.0, box.y_max)) * height)

    x1, y1 = min(mask_x1, box_x1), min(mask_y1, box_y1)
    x2, y2 = max(mask_x2, box_x2), max(mask_y2, box_y2)
    crop_width, crop_height = max(1, x2 - x1), max(1, y2 - y1)
    side = max(crop_width, crop_height) * context_scale
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = max(0, int(np.floor(center_x - side / 2.0)))
    top = max(0, int(np.floor(center_y - side / 2.0)))
    right = min(width, int(np.ceil(center_x + side / 2.0)))
    bottom = min(height, int(np.ceil(center_y + side / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError("box and mask produce an empty crop")

    local_image = frame_rgb[top:bottom, left:right].copy()
    local_mask = mask[top:bottom, left:right]
    # Masking is load-bearing for instance identity. Neutral gray outside the
    # object prevents a desk or a hand from becoming the easiest identifier.
    local_image[~local_mask] = _BACKGROUND

    square_side = max(local_image.shape[:2])
    square_image = np.full((square_side, square_side, 3), _BACKGROUND, dtype=np.uint8)
    square_mask = np.zeros((square_side, square_side), dtype=np.uint8)
    offset_y = (square_side - local_image.shape[0]) // 2
    offset_x = (square_side - local_image.shape[1]) // 2
    square_image[
        offset_y : offset_y + local_image.shape[0],
        offset_x : offset_x + local_image.shape[1],
    ] = local_image
    square_mask[
        offset_y : offset_y + local_mask.shape[0],
        offset_x : offset_x + local_mask.shape[1],
    ] = local_mask.astype(np.uint8) * 255

    resized_image = np.asarray(
        Image.fromarray(square_image).resize(
            (output_size, output_size), resample=Image.Resampling.BICUBIC
        ),
        dtype=np.uint8,
    )
    resized_mask = (
        np.asarray(
            Image.fromarray(square_mask).resize(
                (output_size, output_size), resample=Image.Resampling.NEAREST
            )
        )
        > 0
    )
    return MaskedCrop(image=resized_image, mask=resized_mask)


__all__ = ["box_to_mask", "encode_jpeg", "prepare_masked_crop"]
