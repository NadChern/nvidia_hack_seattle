"""Deterministic CPU identity embedder for tests and no-GPU demos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vision_worker.identity.base import EmbeddingVectors, MaskedCrop


class FixtureEmbedder:
    embedder_id = "fixture-object-embedder-v1"
    pooling = "masked-color-summary-v1"

    @property
    def is_ready(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def embed(self, crops: Sequence[MaskedCrop]) -> Sequence[EmbeddingVectors]:
        return tuple(self.embed_one(crop) for crop in crops)

    def embed_one(self, crop: MaskedCrop) -> EmbeddingVectors:
        pixels = crop.image[crop.mask].astype(np.float32) / 255.0
        if not len(pixels):
            raise ValueError("fixture embedder cannot embed an empty mask")
        summary = np.concatenate((pixels.mean(axis=0), pixels.std(axis=0))).astype(np.float32)

        midpoint = crop.image.shape[0] // 2
        upper_mask = crop.mask[:midpoint]
        lower_mask = crop.mask[midpoint:]
        upper = crop.image[:midpoint][upper_mask].astype(np.float32) / 255.0
        lower = crop.image[midpoint:][lower_mask].astype(np.float32) / 255.0
        upper_mean = upper.mean(axis=0) if len(upper) else pixels.mean(axis=0)
        lower_mean = lower.mean(axis=0) if len(lower) else pixels.mean(axis=0)
        spatial = np.concatenate((upper_mean, lower_mean)).astype(np.float32)
        return EmbeddingVectors(
            embedder_id=self.embedder_id,
            pooling=self.pooling,
            summary=_unit(summary),
            pooled_spatial=_unit(spatial),
        )

    def readiness_payload(self) -> Mapping[str, object]:
        return {"identity_embedder": "fixture", "ready": True, "dim": 6}

    async def aclose(self) -> None:
        return None


def _unit(vector: np.ndarray[tuple[int], np.dtype[np.float32]]) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, np.finfo(np.float32).eps)


__all__ = ["FixtureEmbedder"]
