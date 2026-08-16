"""Vision-owned registration capture, selection, embedding, and persistence."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from visual_memory_media_contract.images import decode_video_payload
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_memory_contract.protocol import ObjectViewQuality, ObjectViewUpload

from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.identity.base import MaskedCrop, ObjectEmbedder
from vision_worker.identity.crop import box_to_mask, encode_jpeg, prepare_masked_crop
from vision_worker.identity.gallery import GalleryCache
from vision_worker.identity.selection import (
    EmbeddedCandidate,
    QualityConfig,
    QualityScore,
    apply_relative_sharpness,
    score_quality,
    select_diverse,
)
from vision_worker.reason.base import Localizer

logger = logging.getLogger(__name__)

EnrollmentState = Literal["capturing", "extracting", "succeeded", "failed"]

#: Cosmos reports no detection score, so enrollment uses a fixed confidence
#: above the quality filter's floor -- the object being deliberately held up
#: and rotated is, by construction, a confident presence.
_ENROLL_CONFIDENCE = 0.9


@dataclass(frozen=True, slots=True)
class EnrollmentConfig:
    capture_seconds: float = 6.0
    max_capture_seconds: float = 15.0
    max_frames: int = 48
    target_views: int = 4
    min_views: int = 2
    dedup_threshold: float = 0.95
    summary_weight: float = 0.5
    quality: QualityConfig = QualityConfig()


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    frames_total: int
    detections: int
    quality_passed: int
    selected_views: int


@dataclass(slots=True)
class EnrollmentProgress:
    object_id: str
    label: str
    state: EnrollmentState
    started_at: dt.datetime
    capture_ends_at: dt.datetime
    frames_total: int = 0
    detections: int = 0
    quality_passed: int = 0
    selected_views: int = 0
    reason_code: str | None = None
    message: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "state": self.state,
            "started_at": self.started_at.isoformat(),
            "capture_ends_at": self.capture_ends_at.isoformat(),
            "frames_total": self.frames_total,
            "detections": self.detections,
            "quality_passed": self.quality_passed,
            "selected_views": self.selected_views,
            "reason_code": self.reason_code,
            "message": self.message,
        }


class EnrollmentError(Exception):
    def __init__(self, reason_code: str, message: str, *, result: EnrollmentResult) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result = result


class ObjectEnroller:
    """Extract and persist a diverse reference gallery from one ring window."""

    def __init__(
        self,
        *,
        localizer: Localizer,
        embedder: ObjectEmbedder,
        gallery: GalleryCache,
        memory_client: MemoryClient,
        config: EnrollmentConfig,
        box_padding: float = 0.12,
    ) -> None:
        self._localizer = localizer
        self._embedder = embedder
        self._gallery = gallery
        self._memory = memory_client
        self._config = config
        self._box_padding = box_padding

    async def enroll(
        self,
        *,
        object_id: str,
        label: str,
        frames: Sequence[BufferedFrame],
    ) -> EnrollmentResult:
        sampled = _subsample(frames, self._config.max_frames)
        # Localize the object in every sampled frame at once. Cosmos is ~5s per
        # call, so localizing serially would make registration unbearable; the
        # frames are independent, so a single gather lets the model server batch
        # them. The reasoner's box is used for enrollment exactly as at query
        # time, so the enrolled crop is framed like the crops it will be matched
        # against -- the crop-parity the cosine depends on.
        boxes = await asyncio.gather(
            *(self._localizer.localize(buffered.payload, label) for buffered in sampled)
        )
        candidates: list[tuple[MaskedCrop, QualityScore]] = []
        detections = 0
        for buffered, box in zip(sampled, boxes, strict=True):
            if box is None:
                continue
            detections += 1
            rgb = decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            mask = box_to_mask(box, rgb.shape[0], rgb.shape[1], padding=self._box_padding)
            quality = score_quality(
                rgb,
                mask,
                box,
                _ENROLL_CONFIDENCE,
                angular_velocity=None,
                config=self._config.quality,
            )
            try:
                crop = prepare_masked_crop(rgb, mask, box)
            except ValueError:
                continue
            candidates.append((crop, quality))

        relative_scores = apply_relative_sharpness(
            [quality for _, quality in candidates], config=self._config.quality
        )
        physically_usable = [
            (crop, quality)
            for (crop, _), quality in zip(candidates, relative_scores, strict=True)
            if quality.accepted
        ]

        # A sharp, reasonably sized first box can still contain mostly a key
        # ring, fob, hand, or floor. Localize inside that crop, then build the
        # crop that will actually be embedded from the second, tighter box.
        # This is also the transform used for live identity below, preserving
        # enrollment/query crop parity instead of validating one box and storing
        # the broader, known-bad one.
        semantic_boxes = await asyncio.gather(
            *(
                self._localizer.localize(encode_jpeg(crop.image), label)
                for crop, _quality in physically_usable
            )
        )
        refined: list[tuple[MaskedCrop, QualityScore]] = []
        for (first_crop, _first_quality), semantic_box in zip(
            physically_usable, semantic_boxes, strict=True
        ):
            if semantic_box is None:
                continue
            height, width = first_crop.image.shape[:2]
            mask = box_to_mask(semantic_box, height, width, padding=self._box_padding)
            quality = score_quality(
                first_crop.image,
                mask,
                semantic_box,
                _ENROLL_CONFIDENCE,
                angular_velocity=None,
                config=self._config.quality,
            )
            try:
                crop = prepare_masked_crop(first_crop.image, mask, semantic_box)
            except ValueError:
                continue
            refined.append((crop, quality))

        refined_scores = apply_relative_sharpness(
            [quality for _, quality in refined], config=self._config.quality
        )
        semantically_localized = [
            (crop, quality)
            for (crop, _), quality in zip(refined, refined_scores, strict=True)
            if quality.accepted
        ]

        # Grounding answers where the model thinks an object might be. A
        # separate contrastive QC question is intentionally stricter: for keys,
        # for example, a ring/fob without a visible metal blade is REJECT. This
        # caught all three bad suggestions from the second physical retry even
        # though grounding had returned boxes for them.
        reference_results = await asyncio.gather(
            *(
                self._localizer.validate_reference(encode_jpeg(crop.image), label)
                for crop, _quality in semantically_localized
            )
        )
        passed = [
            candidate
            for candidate, valid in zip(semantically_localized, reference_results, strict=True)
            if valid
        ]
        logger.info(
            "registration semantic crop gate completed",
            extra={
                "object_id": object_id,
                "label": label,
                "localized": detections,
                "physical_quality": len(physically_usable),
                "semantic_localized": len(semantically_localized),
                "reference_valid": len(passed),
            },
        )
        preliminary = EnrollmentResult(len(sampled), detections, len(passed), 0)
        if len(passed) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_quality_frames",
                f"only {len(passed)} frames passed quality; need {self._config.min_views}",
                result=preliminary,
            )

        embeddings = await self._embedder.embed([crop for crop, _ in passed])
        embedded = tuple(
            EmbeddedCandidate(crop=crop, embedding=embedding, quality=quality)
            for (crop, quality), embedding in zip(passed, embeddings, strict=True)
        )
        selected_views = select_diverse(
            embedded,
            k=self._config.target_views,
            dedup_threshold=self._config.dedup_threshold,
            summary_weight=self._config.summary_weight,
        )
        result = EnrollmentResult(
            frames_total=len(sampled),
            detections=detections,
            quality_passed=len(passed),
            selected_views=len(selected_views),
        )
        if len(selected_views) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_diverse_views",
                f"only {len(selected_views)} diverse views; need {self._config.min_views}",
                result=result,
            )

        try:
            for view_index, candidate in enumerate(selected_views):
                crop_bytes = encode_jpeg(candidate.crop.image)
                digest = hashlib.sha256(crop_bytes).hexdigest()
                quality = candidate.quality
                upload = ObjectViewUpload(
                    view_index=view_index,
                    quality=ObjectViewQuality(
                        detection_confidence=quality.detection_confidence,
                        box_area_fraction=quality.box_area_fraction,
                        sharpness_score=quality.sharpness_score,
                        mask_box_ratio=quality.mask_box_ratio,
                        quality_score=quality.quality_score,
                        angular_velocity_rad_s=quality.angular_velocity_rad_s,
                    ),
                    embedder_id=candidate.embedding.embedder_id,
                    pooling=candidate.embedding.pooling,
                    dim=candidate.embedding.dim,
                    summary=tuple(float(value) for value in candidate.embedding.summary),
                    pooled_spatial=tuple(
                        float(value) for value in candidate.embedding.pooled_spatial
                    ),
                    crop_sha256=digest,
                    crop_media_type="image/jpeg",
                    crop_base64=base64.b64encode(crop_bytes).decode("ascii"),
                )
                await asyncio.to_thread(self._memory.put_object_view, object_id, upload)
        except MemoryError_:
            # A half-gallery fails silently forever. Remove the fresh object so
            # the caller gets an honest retry rather than apparent success.
            try:
                await asyncio.to_thread(self._memory.delete_object, object_id)
            except MemoryError_:
                logger.exception("failed to roll back partial registration")
            raise

        await self._gallery.refresh(force=True)
        return result

    async def _rollback_failed_object(self, object_id: str) -> None:
        """Remove an object whose capture produced no usable gallery."""
        try:
            await asyncio.to_thread(self._memory.delete_object, object_id)
        except MemoryError_:
            logger.exception(
                "failed to remove object after rejected registration",
                extra={"object_id": object_id},
            )


class EnrollmentManager:
    """Arms capture windows and owns their background tasks and status."""

    def __init__(
        self,
        *,
        evidence_ring: EvidenceRing,
        enroller: ObjectEnroller,
        config: EnrollmentConfig,
    ) -> None:
        self._ring = evidence_ring
        self._enroller = enroller
        self._config = config
        self._progress: dict[str, EnrollmentProgress] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self.attempts = 0
        self.succeeded = 0
        self.failed = 0

    def arm(
        self,
        *,
        object_id: str,
        label: str,
        capture_seconds: float | None = None,
    ) -> EnrollmentProgress:
        existing = self._tasks.get(object_id)
        if existing is not None and not existing.done():
            raise RuntimeError("registration capture is already active for this object")
        duration = capture_seconds or self._config.capture_seconds
        if duration <= 0.0 or duration > self._config.max_capture_seconds:
            raise ValueError(
                f"capture_seconds must be >0 and <= {self._config.max_capture_seconds}"
            )
        started = dt.datetime.now(dt.UTC)
        progress = EnrollmentProgress(
            object_id=object_id,
            label=label,
            state="capturing",
            started_at=started,
            capture_ends_at=started + dt.timedelta(seconds=duration),
        )
        self._progress[object_id] = progress
        self.attempts += 1
        self._tasks[object_id] = asyncio.create_task(
            self._run(progress, duration), name=f"enroll-{object_id}"
        )
        return progress

    def status(self, object_id: str) -> EnrollmentProgress | None:
        return self._progress.get(object_id)

    def status_payload(self) -> Mapping[str, object]:
        active = sum(1 for task in self._tasks.values() if not task.done())
        return {
            "attempts": self.attempts,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "active": active,
        }

    async def aclose(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, progress: EnrollmentProgress, duration: float) -> None:
        try:
            await asyncio.sleep(duration)
            progress.state = "extracting"
            frames = self._ring.window(
                started_at=progress.started_at,
                ended_at=progress.capture_ends_at,
            )
            result = await self._enroller.enroll(
                object_id=progress.object_id,
                label=progress.label,
                frames=frames,
            )
        except asyncio.CancelledError:
            raise
        except EnrollmentError as exc:
            progress.state = "failed"
            progress.reason_code = exc.reason_code
            progress.message = str(exc)
            _apply_result(progress, exc.result)
            self.failed += 1
        except Exception as exc:
            progress.state = "failed"
            progress.reason_code = "registration_unavailable"
            progress.message = str(exc)
            self.failed += 1
            logger.exception("registration extraction failed")
        else:
            progress.state = "succeeded"
            progress.reason_code = "enrollment_complete"
            progress.message = "reference gallery stored"
            _apply_result(progress, result)
            self.succeeded += 1


def _apply_result(progress: EnrollmentProgress, result: EnrollmentResult) -> None:
    progress.frames_total = result.frames_total
    progress.detections = result.detections
    progress.quality_passed = result.quality_passed
    progress.selected_views = result.selected_views


def _subsample(frames: Sequence[BufferedFrame], limit: int) -> tuple[BufferedFrame, ...]:
    if len(frames) <= limit:
        return tuple(frames)
    indexes = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return tuple(frames[int(index)] for index in dict.fromkeys(indexes.tolist()))


__all__ = [
    "EnrollmentConfig",
    "EnrollmentError",
    "EnrollmentManager",
    "EnrollmentProgress",
    "EnrollmentResult",
    "EnrollmentState",
    "ObjectEnroller",
]
