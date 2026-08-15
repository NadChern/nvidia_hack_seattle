"""Adapter boundaries for personal-object segmentation and embedding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import BoundingBox, Detection, IdentityMatch


@dataclass(frozen=True, slots=True)
class SegmentedDetection:
    detection: Detection
    #: Full-frame boolean mask in the same HxW coordinates as the RGB frame.
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MaskedCrop:
    #: Square RGB crop, uint8.
    image: NDArray[np.uint8]
    #: Square object mask aligned pixel-for-pixel with `image`.
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EmbeddingVectors:
    embedder_id: str
    pooling: str
    summary: NDArray[np.float32]
    pooled_spatial: NDArray[np.float32]

    @property
    def dim(self) -> int:
        return int(self.summary.shape[0])


@dataclass(frozen=True, slots=True)
class IdentityFrame:
    frame_rgb: NDArray[np.uint8]
    detection: Detection


class SegmentingDetector(Protocol):
    async def segment(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[SegmentedDetection]: ...


class ObjectEmbedder(Protocol):
    @property
    def is_ready(self) -> bool: ...

    @property
    def embedder_id(self) -> str: ...

    @property
    def pooling(self) -> str: ...

    async def initialize(self) -> None: ...

    async def embed(self, crops: Sequence[MaskedCrop]) -> Sequence[EmbeddingVectors]: ...

    def readiness_payload(self) -> Mapping[str, object]: ...

    async def aclose(self) -> None: ...


class IdentityResolverProtocol(Protocol):
    def accepts_label(self, label: str) -> bool: ...

    async def resolve(self, frames: Sequence[IdentityFrame]) -> IdentityMatch: ...

    def status_payload(self) -> Mapping[str, object]: ...

    async def aclose(self) -> None: ...


class IdentityEscalator(Protocol):
    async def verify(
        self,
        *,
        object_id: str,
        label: str,
        query_crops: Sequence[MaskedCrop],
        crop_references: Sequence[str],
    ) -> bool | None: ...


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x_min, b.x_min), max(a.y_min, b.y_min)
    x2, y2 = min(a.x_max, b.x_max), min(a.y_max, b.y_max)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a.x_max - a.x_min) * max(0.0, a.y_max - a.y_min)
    area_b = max(0.0, b.x_max - b.x_min) * max(0.0, b.y_max - b.y_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


__all__ = [
    "EmbeddingVectors",
    "IdentityEscalator",
    "IdentityFrame",
    "IdentityResolverProtocol",
    "MaskedCrop",
    "ObjectEmbedder",
    "SegmentedDetection",
    "SegmentingDetector",
    "box_iou",
]
