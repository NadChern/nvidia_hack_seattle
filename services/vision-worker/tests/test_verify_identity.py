"""Per-track identity resolution and VLM escalation gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.identity.base import IdentityFrame, MaskedCrop, SegmentedDetection
from vision_worker.identity.fixture import FixtureEmbedder
from vision_worker.identity.gallery import GalleryScore
from vision_worker.identity.resolver import IdentityResolver, IdentityResolverConfig

pytestmark = pytest.mark.anyio


class BoxSegmenter:
    async def segment(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[SegmentedDetection]:
        del labels
        mask = np.zeros(frame_rgb.shape[:2], dtype=np.bool_)
        mask[8:24, 8:24] = True
        return (SegmentedDetection(detection=a_detection(), mask=mask),)


class StaticGallery:
    labels = frozenset({"keys"})

    def __init__(self, score: GalleryScore) -> None:
        self._score = score

    async def refresh(self, *, force: bool = False) -> bool:
        del force
        return False

    def match(self, *_: object, **__: object) -> GalleryScore:
        return self._score

    def status_payload(self) -> Mapping[str, object]:
        return {"gallery_objects": 2, "gallery_views": 4}


class StaticEscalator:
    def __init__(self, verdict: bool | None) -> None:
        self.verdict = verdict
        self.calls = 0

    async def verify(
        self,
        *,
        object_id: str,
        label: str,
        query_crops: Sequence[MaskedCrop],
        crop_references: Sequence[str],
    ) -> bool | None:
        del object_id, label, query_crops, crop_references
        self.calls += 1
        return self.verdict


def a_detection() -> Detection:
    return Detection(
        label="keys",
        confidence=0.95,
        box=BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75),
        centroid=Point2D(x=0.5, y=0.5),
    )


def frames() -> tuple[IdentityFrame, ...]:
    image = np.full((32, 32, 3), (220, 20, 20), dtype=np.uint8)
    return tuple(IdentityFrame(frame_rgb=image.copy(), detection=a_detection()) for _ in range(3))


async def test_clear_embedding_match_resolves_without_escalation() -> None:
    escalator = StaticEscalator(True)
    resolver = IdentityResolver(
        segmenter=BoxSegmenter(),
        embedder=FixtureEmbedder(),
        gallery=StaticGallery(  # type: ignore[arg-type]
            GalleryScore(
                object_id="object_mine",
                score=0.93,
                margin=0.12,
                runner_up_object_id="object_other",
                crop_references=("/v1/objects/object_mine/views/view_1/crop",),
            )
        ),
        config=IdentityResolverConfig(min_cosine=0.85, min_margin=0.05, escalation_low=0.8),
        escalator=escalator,
    )
    await resolver.initialize()

    match = await resolver.resolve(frames())

    assert match.object_id == "object_mine"
    assert match.reason_code == "embedding_resolved"
    assert escalator.calls == 0


async def test_near_threshold_ambiguity_escalates_once_and_can_abstain() -> None:
    escalator = StaticEscalator(False)
    resolver = IdentityResolver(
        segmenter=BoxSegmenter(),
        embedder=FixtureEmbedder(),
        gallery=StaticGallery(  # type: ignore[arg-type]
            GalleryScore(
                object_id="object_mine",
                score=0.84,
                margin=0.01,
                runner_up_object_id="object_other",
                crop_references=("/v1/objects/object_mine/views/view_1/crop",),
            )
        ),
        config=IdentityResolverConfig(min_cosine=0.85, min_margin=0.05, escalation_low=0.8),
        escalator=escalator,
    )
    await resolver.initialize()

    match = await resolver.resolve(frames())

    assert match.object_id is None
    assert match.reason_code == "vlm_rejected"
    assert match.escalated is True
    assert escalator.calls == 1
