"""Pure-numpy enrollment quality filtering and diversity selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import BoundingBox

from vision_worker.identity.base import EmbeddingVectors, MaskedCrop


@dataclass(frozen=True, slots=True)
class QualityConfig:
    min_detection_confidence: float = 0.5
    min_scale: float = 0.01
    min_mask_box_ratio: float = 0.4
    max_mask_box_ratio: float = 1.05
    relative_sharpness_floor: float = 0.5
    max_angular_velocity_rad_s: float = 2.5


@dataclass(frozen=True, slots=True)
class QualityScore:
    accepted: bool
    reason: str
    detection_confidence: float
    box_area_fraction: float
    sharpness_score: float
    mask_box_ratio: float
    quality_score: float
    angular_velocity_rad_s: float | None = None

    def with_window_median(self, median_sharpness: float, *, config: QualityConfig) -> QualityScore:
        relative = self.sharpness_score / max(median_sharpness, np.finfo(np.float32).eps)
        accepted = self.accepted and relative >= config.relative_sharpness_floor
        reason = (
            self.reason if self.reason != "accepted" or accepted else "below_relative_sharpness"
        )
        quality = (
            0.35 * self.detection_confidence
            + 0.25 * min(1.0, relative)
            + 0.4 * min(1.0, self.box_area_fraction / max(config.min_scale * 4.0, 1e-9))
        )
        return replace(self, accepted=accepted, reason=reason, quality_score=quality)


@dataclass(frozen=True, slots=True)
class EmbeddedCandidate:
    crop: MaskedCrop
    embedding: EmbeddingVectors
    quality: QualityScore
    crop_sha256: str = ""


def score_quality(
    crop: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    box: BoundingBox,
    confidence: float,
    *,
    angular_velocity: float | None = None,
    config: QualityConfig | None = None,
) -> QualityScore:
    """Score one frame without fixed-lighting sharpness constants.

    The returned sharpness is compared with the window median in a second
    pass through `with_window_median`.
    """
    resolved = config or QualityConfig()
    height, width = mask.shape
    if crop.shape[:2] != (height, width):
        raise ValueError("crop and mask must share HxW")
    area = max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)
    x1 = max(0, min(width, round(box.x_min * width)))
    x2 = max(0, min(width, round(box.x_max * width)))
    y1 = max(0, min(height, round(box.y_min * height)))
    y2 = max(0, min(height, round(box.y_max * height)))
    box_pixels = max(1, (x2 - x1) * (y2 - y1))
    mask_ratio = float(mask[y1:y2, x1:x2].sum()) / box_pixels

    object_pixels = crop[y1:y2, x1:x2]
    gray = (
        object_pixels[..., 0].astype(np.float32) * 0.299
        + object_pixels[..., 1].astype(np.float32) * 0.587
        + object_pixels[..., 2].astype(np.float32) * 0.114
    )
    if min(gray.shape, default=0) < 2:
        sharpness = 0.0
    else:
        gradient_y, gradient_x = np.gradient(gray)
        magnitude = gradient_x * gradient_x + gradient_y * gradient_y
        sharpness = float(np.var(magnitude))

    reason = "accepted"
    if confidence < resolved.min_detection_confidence:
        reason = "low_detection_confidence"
    elif area < resolved.min_scale:
        reason = "too_small"
    elif box.x_min <= 0.0 or box.y_min <= 0.0 or box.x_max >= 1.0 or box.y_max >= 1.0:
        reason = "touches_frame_edge"
    elif not resolved.min_mask_box_ratio <= mask_ratio <= resolved.max_mask_box_ratio:
        reason = "fragmented_or_invalid_mask"
    elif angular_velocity is not None and angular_velocity > resolved.max_angular_velocity_rad_s:
        reason = "gyro_blur_risk"

    quality = 0.35 * confidence + 0.4 * min(1.0, area / max(resolved.min_scale * 4.0, 1e-9))
    return QualityScore(
        accepted=reason == "accepted",
        reason=reason,
        detection_confidence=confidence,
        box_area_fraction=area,
        sharpness_score=sharpness,
        mask_box_ratio=mask_ratio,
        quality_score=quality,
        angular_velocity_rad_s=angular_velocity,
    )


def apply_relative_sharpness(
    scores: Sequence[QualityScore], *, config: QualityConfig | None = None
) -> tuple[QualityScore, ...]:
    resolved = config or QualityConfig()
    candidates = [score.sharpness_score for score in scores if score.accepted]
    if not candidates:
        return tuple(scores)
    median = float(np.median(np.asarray(candidates, dtype=np.float64)))
    return tuple(score.with_window_median(median, config=resolved) for score in scores)


def select_diverse(
    candidates: Sequence[EmbeddedCandidate],
    *,
    k: int = 4,
    dedup_threshold: float = 0.95,
    summary_weight: float = 0.5,
) -> tuple[EmbeddedCandidate, ...]:
    """Greedy farthest-point sampling in the shared embedding space."""
    accepted = [candidate for candidate in candidates if candidate.quality.accepted]
    if not accepted or k < 1:
        return ()
    seed = max(accepted, key=lambda candidate: candidate.quality.quality_score)
    selected = [seed]
    remaining = [candidate for candidate in accepted if candidate is not seed]
    while remaining and len(selected) < k:
        similarities = [
            max(_cosine(candidate, chosen, summary_weight) for chosen in selected)
            for candidate in remaining
        ]
        farthest_index = int(np.argmin(np.asarray(similarities)))
        if similarities[farthest_index] > dedup_threshold:
            break
        selected.append(remaining.pop(farthest_index))
    return tuple(selected)


def _cosine(a: EmbeddedCandidate, b: EmbeddedCandidate, summary_weight: float) -> float:
    return summary_weight * float(np.dot(a.embedding.summary, b.embedding.summary)) + (
        1.0 - summary_weight
    ) * float(np.dot(a.embedding.pooled_spatial, b.embedding.pooled_spatial))


__all__ = [
    "EmbeddedCandidate",
    "QualityConfig",
    "QualityScore",
    "apply_relative_sharpness",
    "score_quality",
    "select_diverse",
]
