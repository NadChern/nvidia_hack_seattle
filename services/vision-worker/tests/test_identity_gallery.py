"""Pure-numpy gallery scoring: max over views, mean over query frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vision_worker.identity.base import EmbeddingVectors
from vision_worker.identity.gallery import GalleryView, object_thresholds, score_gallery


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


def test_a_lone_object_degrades_to_the_floor() -> None:
    # No same-label sibling to be confused with, so the bar is just the floor --
    # this is why a single global cosine sufficed for the one-object demo.
    views = (
        view("object_mine", 0, vector(1.0, 0.0)),
        view("object_mine", 1, vector(0.9, 0.4359)),
    )

    result = score_gallery(
        views,
        (query(vector(1.0, 0.0)),),
        label="keys",
        summary_weight=0.5,
        floor=0.6,
        confusion_margin=0.05,
    )

    assert result is not None
    assert result.object_id == "object_mine"
    assert result.threshold == pytest.approx(0.6)


def test_a_same_label_sibling_raises_the_bar_to_its_confusable_similarity() -> None:
    # object_other's reference sits at cosine 0.8 from object_mine's, so a query
    # must clear 0.8 + margin to be accepted as mine -- the second keyring the
    # global threshold could not survive.
    views = (
        view("object_mine", 0, vector(1.0, 0.0)),
        view("object_other", 0, vector(0.8, 0.6)),
    )

    result = score_gallery(
        views,
        (query(vector(1.0, 0.0)),),
        label="keys",
        summary_weight=0.5,
        floor=0.6,
        confusion_margin=0.04,
    )

    assert result is not None
    assert result.object_id == "object_mine"
    assert result.threshold == pytest.approx(0.84)


def test_object_thresholds_reports_each_objects_bar() -> None:
    views = (
        view("object_mine", 0, vector(1.0, 0.0)),
        view("object_other", 0, vector(0.8, 0.6)),
        # A distinct object under a different label competes with no one.
        replace(view("object_mug", 0, vector(1.0, 0.0)), label="mug"),
    )

    thresholds = object_thresholds(
        views, summary_weight=0.5, floor=0.6, confusion_margin=0.04
    )

    assert thresholds["object_mine"] == pytest.approx(0.84)
    assert thresholds["object_other"] == pytest.approx(0.84)
    assert thresholds["object_mug"] == pytest.approx(0.6)


def test_defaults_impose_no_per_object_gate() -> None:
    # Callers that pass neither floor nor margin get threshold 0, so existing
    # single-cosine callers are unchanged until they opt in.
    result = score_gallery(
        (view("object_mine", 0, vector(1.0, 0.0)),),
        (query(vector(1.0, 0.0)),),
        label="keys",
        summary_weight=0.5,
    )

    assert result is not None
    assert result.threshold == pytest.approx(0.0)


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
