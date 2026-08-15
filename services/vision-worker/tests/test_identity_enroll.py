"""Registration extraction succeeds on good views and rejects weak footage."""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.evidence.ring import BufferedFrame
from vision_worker.identity.base import MaskedCrop, SegmentedDetection
from vision_worker.identity.crop import prepare_masked_crop
from vision_worker.identity.enroll import EnrollmentConfig, EnrollmentError, ObjectEnroller
from vision_worker.identity.fixture import FixtureEmbedder
from vision_worker.identity.selection import QualityConfig

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 8, 15, 12, 0, tzinfo=dt.UTC)
BOX = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)


class BoxSegmenter:
    async def segment(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[SegmentedDetection]:
        mask = np.zeros(frame_rgb.shape[:2], dtype=np.bool_)
        mask[16:48, 16:48] = True
        return (
            SegmentedDetection(
                detection=Detection(
                    label=labels[0],
                    confidence=0.95,
                    box=BOX,
                    centroid=Point2D(x=0.5, y=0.5),
                ),
                mask=mask,
            ),
        )


class SharedFixtureResolver:
    def __init__(self) -> None:
        self.embedder = FixtureEmbedder()
        self.refreshes = 0

    async def embed_crops(self, crops: Sequence[MaskedCrop]):  # type: ignore[no-untyped-def]
        return await self.embedder.embed(crops)

    async def refresh_gallery(self) -> None:
        self.refreshes += 1


class RecordingMemory:
    def __init__(self) -> None:
        self.uploads: list[object] = []
        self.deleted: list[str] = []

    def put_object_view(self, object_id: str, upload: object) -> object:
        self.uploads.append(upload)
        return upload

    def delete_object(self, object_id: str) -> None:
        self.deleted.append(object_id)


def frame(index: int, color: tuple[int, int, int], *, textured: bool = True) -> BufferedFrame:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    patch = np.full((32, 32, 3), color, dtype=np.int16)
    if textured:
        checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 30 - 15)[..., None]
        patch = np.clip(patch + checker, 0, 255)
    image[16:48, 16:48] = patch.astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=95)
    return BufferedFrame(
        captured_at=T0 + dt.timedelta(milliseconds=index * 100),
        payload=output.getvalue(),
        width=64,
        height=64,
    )


def config() -> EnrollmentConfig:
    return EnrollmentConfig(
        max_frames=12,
        target_views=4,
        min_views=2,
        dedup_threshold=0.9999,
        quality=QualityConfig(relative_sharpness_floor=0.5),
    )


async def test_good_capture_selects_and_persists_a_diverse_gallery() -> None:
    resolver = SharedFixtureResolver()
    memory = RecordingMemory()
    enroller = ObjectEnroller(
        segmenter=BoxSegmenter(),
        identity_resolver=resolver,  # type: ignore[arg-type]
        memory_client=memory,  # type: ignore[arg-type]
        config=config(),
    )
    frames = tuple(
        frame(index, color)
        for index, color in enumerate(
            ((220, 30, 30), (30, 220, 30), (30, 30, 220), (220, 180, 30), (200, 35, 35))
        )
    )

    result = await enroller.enroll(object_id="object_keys", label="keys", frames=frames)

    assert result.frames_total == 5
    assert result.quality_passed == 5
    assert 2 <= result.selected_views <= 4
    assert len(memory.uploads) == result.selected_views
    assert resolver.refreshes == 1


async def test_bad_capture_is_rejected_without_storing_a_weak_gallery() -> None:
    resolver = SharedFixtureResolver()
    memory = RecordingMemory()
    enroller = ObjectEnroller(
        segmenter=BoxSegmenter(),
        identity_resolver=resolver,  # type: ignore[arg-type]
        memory_client=memory,  # type: ignore[arg-type]
        config=config(),
    )
    frames = tuple(frame(index, (120, 120, 120), textured=False) for index in range(4))

    with pytest.raises(EnrollmentError, match="passed quality") as caught:
        await enroller.enroll(object_id="object_keys", label="keys", frames=frames)

    assert caught.value.reason_code == "too_few_quality_frames"
    assert memory.uploads == []


async def test_enrollment_and_matching_use_the_identical_crop_transform() -> None:
    source = np.full((64, 64, 3), 127, dtype=np.uint8)
    source[16:48, 16:48] = (220, 30, 30)
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[16:48, 16:48] = True
    enrollment_crop = prepare_masked_crop(source, mask, BOX)
    matching_crop = prepare_masked_crop(source, mask, BOX)
    embedder = FixtureEmbedder()

    enrollment_vector, matching_vector = await embedder.embed((enrollment_crop, matching_crop))
    cosine = float(np.dot(enrollment_vector.summary, matching_vector.summary))

    assert np.array_equal(enrollment_crop.image, matching_crop.image)
    assert np.array_equal(enrollment_crop.mask, matching_crop.mask)
    assert np.array_equal(enrollment_vector.summary, matching_vector.summary)
    assert cosine == pytest.approx(1.0, abs=2e-7)
