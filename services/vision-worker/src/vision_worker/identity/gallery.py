"""Versioned last-known-good gallery cache and pure-numpy matching."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_memory_contract.protocol import ObjectGallery, ObjectView

from vision_worker.identity.base import EmbeddingVectors

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GalleryView:
    object_id: str
    label: str
    view_id: str
    embedder_id: str
    pooling: str
    dim: int
    summary: NDArray[np.float32]
    pooled_spatial: NDArray[np.float32]
    crop_reference: str


@dataclass(frozen=True, slots=True)
class GalleryScore:
    object_id: str
    score: float
    margin: float | None
    runner_up_object_id: str | None
    crop_references: tuple[str, ...]
    stale_views: int = 0


@dataclass(slots=True)
class GalleryMetrics:
    refreshes: int = 0
    refresh_failures: int = 0
    stale_views: int = 0


class GalleryCache:
    """TTL refresh with a monotonic version and last-known-good fallback."""

    def __init__(self, client: MemoryClient, *, ttl_s: float = 30.0) -> None:
        self._client = client
        self._ttl_s = ttl_s
        self._version = 0
        self._views: tuple[GalleryView, ...] = ()
        self._refreshed_at: dt.datetime | None = None
        self._stale = False
        self.metrics = GalleryMetrics()

    @property
    def version(self) -> int:
        return self._version

    @property
    def has_snapshot(self) -> bool:
        return self._refreshed_at is not None

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(view.label for view in self._views)

    async def refresh(self, *, force: bool = False) -> bool:
        now = dt.datetime.now(dt.UTC)
        if (
            not force
            and self._refreshed_at is not None
            and (now - self._refreshed_at).total_seconds() < self._ttl_s
        ):
            return False
        try:
            gallery = await asyncio.to_thread(
                self._client.list_gallery,
                since_version=self._version if self.has_snapshot else None,
            )
        except MemoryError_:
            self.metrics.refresh_failures += 1
            self._stale = True
            logger.warning("identity gallery refresh failed; serving last-known-good")
            return False

        self.metrics.refreshes += 1
        self._refreshed_at = now
        self._stale = False
        if gallery.unchanged:
            self._version = gallery.registry_version
            return False
        self._apply(gallery)
        return True

    def match(
        self,
        queries: Sequence[EmbeddingVectors],
        *,
        label: str,
        summary_weight: float,
    ) -> GalleryScore | None:
        result = score_gallery(self._views, queries, label=label, summary_weight=summary_weight)
        if result is not None:
            self.metrics.stale_views += result.stale_views
        return result

    def status_payload(self) -> Mapping[str, object]:
        object_ids = {view.object_id for view in self._views}
        return {
            "registry_version": self._version,
            "gallery_objects": len(object_ids),
            "gallery_views": len(self._views),
            "gallery_stale": self._stale,
            "stale_views": self.metrics.stale_views,
            "refreshes": self.metrics.refreshes,
            "refresh_failures": self.metrics.refresh_failures,
        }

    def _apply(self, gallery: ObjectGallery) -> None:
        labels = {item.object_id: item.label for item in gallery.objects}
        self._views = tuple(
            _gallery_view(view, labels[view.object_id])
            for view in gallery.views
            if view.object_id in labels
        )
        self._version = gallery.registry_version


def score_gallery(
    views: Sequence[GalleryView],
    queries: Sequence[EmbeddingVectors],
    *,
    label: str,
    summary_weight: float,
) -> GalleryScore | None:
    """Mean over queries of max-over-reference-view weighted cosine."""
    if not queries or not 0.0 <= summary_weight <= 1.0:
        return None
    by_object: defaultdict[str, list[GalleryView]] = defaultdict(list)
    stale = 0
    first = queries[0]
    for view in views:
        if view.label != label:
            continue
        if (
            view.embedder_id != first.embedder_id
            or view.pooling != first.pooling
            or view.dim != first.dim
        ):
            stale += 1
            continue
        by_object[view.object_id].append(view)
    if not by_object:
        return None

    scored: list[tuple[str, float]] = []
    for object_id, object_views in by_object.items():
        per_query: list[float] = []
        for query in queries:
            if (
                query.embedder_id != first.embedder_id
                or query.pooling != first.pooling
                or query.dim != first.dim
            ):
                return None
            per_view = [
                summary_weight * float(np.dot(query.summary, view.summary))
                + (1.0 - summary_weight) * float(np.dot(query.pooled_spatial, view.pooled_spatial))
                for view in object_views
            ]
            per_query.append(max(per_view))
        scored.append((object_id, float(np.mean(per_query))))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score = scored[0]
    runner_id, runner_score = scored[1] if len(scored) > 1 else (None, None)
    margin = best_score - runner_score if runner_score is not None else None
    return GalleryScore(
        object_id=best_id,
        score=max(0.0, min(1.0, best_score)),
        margin=margin,
        runner_up_object_id=runner_id,
        crop_references=tuple(view.crop_reference for view in by_object[best_id]),
        stale_views=stale,
    )


def _gallery_view(view: ObjectView, label: str) -> GalleryView:
    return GalleryView(
        object_id=view.object_id,
        label=label,
        view_id=view.view_id,
        embedder_id=view.embedder_id,
        pooling=view.pooling,
        dim=view.dim,
        summary=_unit(np.asarray(view.summary, dtype=np.float32)),
        pooled_spatial=_unit(np.asarray(view.pooled_spatial, dtype=np.float32)),
        crop_reference=view.crop_reference,
    )


def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, np.finfo(np.float32).eps)


__all__ = [
    "GalleryCache",
    "GalleryMetrics",
    "GalleryScore",
    "GalleryView",
    "score_gallery",
]
