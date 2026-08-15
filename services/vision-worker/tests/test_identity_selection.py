"""Registration frame quality and farthest-point diversity selection."""

from __future__ import annotations

import numpy as np
from visual_memory_vision_contract.protocol import BoundingBox

from vision_worker.identity.base import EmbeddingVectors, MaskedCrop
from vision_worker.identity.selection import (
    EmbeddedCandidate,
    QualityConfig,
    apply_relative_sharpness,
    score_quality,
    select_diverse,
)

BOX = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)


def textured_frame(*, blurred: bool = False) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((64, 64, 3), 127, dtype=np.uint8)
    if blurred:
        frame[13:51, 13:51] = 150
    else:
        checker = (np.indices((38, 38)).sum(axis=0) % 2 * 255).astype(np.uint8)
        frame[13:51, 13:51] = checker[..., None]
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[13:51, 13:51] = True
    return frame, mask


def test_sharpness_is_gated_relative_to_the_capture_window() -> None:
    sharp_frame, mask = textured_frame()
    blurred_frame, _ = textured_frame(blurred=True)
    raw = (
        score_quality(sharp_frame, mask, BOX, 0.9),
        score_quality(blurred_frame, mask, BOX, 0.9),
    )

    sharp, blurred = apply_relative_sharpness(raw)

    assert sharp.accepted is True
    assert blurred.accepted is False
    assert blurred.reason == "below_relative_sharpness"


def test_edge_touch_and_optional_gyro_gate_are_honest_rejections() -> None:
    frame, mask = textured_frame()
    edge = score_quality(
        frame,
        mask,
        BoundingBox(x_min=0.0, y_min=0.2, x_max=0.8, y_max=0.8),
        0.9,
    )
    gyro = score_quality(frame, mask, BOX, 0.9, angular_velocity=3.0)
    no_imu = score_quality(frame, mask, BOX, 0.9, angular_velocity=None)

    assert edge.reason == "touches_frame_edge"
    assert gyro.reason == "gyro_blur_risk"
    assert no_imu.accepted is True


def candidate(vector: tuple[float, float], quality: float) -> EmbeddedCandidate:
    summary = np.asarray(vector, dtype=np.float32)
    summary /= np.linalg.norm(summary)
    frame, mask = textured_frame()
    score = score_quality(frame, mask, BOX, quality).with_window_median(
        1.0, config=QualityConfig(relative_sharpness_floor=0.0)
    )
    return EmbeddedCandidate(
        crop=MaskedCrop(image=frame, mask=mask),
        embedding=EmbeddingVectors(
            embedder_id="fixture",
            pooling="shared",
            summary=summary,
            pooled_spatial=summary,
        ),
        quality=score,
    )


def test_diversity_prefers_a_different_view_and_drops_near_duplicates() -> None:
    front = candidate((1.0, 0.0), 0.99)
    duplicate = candidate((0.999, 0.001), 0.9)
    side = candidate((0.0, 1.0), 0.8)

    selected = select_diverse((front, duplicate, side), k=4, dedup_threshold=0.95)

    assert len(selected) == 2
    assert selected[0] is front
    assert selected[1] is side
