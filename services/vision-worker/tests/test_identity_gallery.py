"""Pure-numpy gallery scoring: max over views, mean over query frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vision_worker.identity.base import EmbeddingVectors
from vision_worker.identity.gallery import GalleryView, score_gallery


def vector(x: float, y: float) -> np.ndarray:
    value = np.asarray((x, y), dtype=np.float32)
    return value / np.linalg.norm(value)


def view(object_id: str, index: int, value: np.ndarray) -> GalleryView:
    return GalleryView(
        object_id=object_id,
        label="keys",
        view_id=f"{object_id}-{index}",
        embedder_id="fixture-v1",
        pooling="shared-v1",
        dim=2,
        summary=value,
        pooled_spatial=value,
        crop_reference=f"/v1/objects/{object_id}/views/view-{index}/crop",
    )


def query(value: np.ndarray) -> EmbeddingVectors:
    return EmbeddingVectors(
        embedder_id="fixture-v1",
        pooling="shared-v1",
        summary=value,
        pooled_spatial=value,
    )


def test_query_pose_uses_the_best_reference_view_not_the_average() -> None:
    views = (
        view("object_mine", 0, vector(1.0, 0.0)),
        view("object_mine", 1, vector(0.0, 1.0)),
        view("object_other", 0, vector(-1.0, 0.0)),
        view("object_other", 1, vector(0.0, -1.0)),
    )

    result = score_gallery(
        views,
        (query(vector(1.0, 0.0)), query(vector(0.0, 1.0))),
        label="keys",
        summary_weight=0.5,
    )

    assert result is not None
    assert result.object_id == "object_mine"
    assert result.score == pytest.approx(1.0)
    assert result.margin == pytest.approx(1.0)


def test_incompatible_embedder_views_are_invalidated_loudly() -> None:
    stale = replace(view("object_stale", 0, vector(1.0, 0.0)), embedder_id="old-model")

    result = score_gallery(
        (view("object_mine", 0, vector(1.0, 0.0)), stale),
        (query(vector(1.0, 0.0)),),
        label="keys",
        summary_weight=0.5,
    )

    assert result is not None
    assert result.object_id == "object_mine"
    assert result.stale_views == 1
