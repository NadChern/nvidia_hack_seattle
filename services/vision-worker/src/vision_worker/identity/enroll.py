"""Vision-owned registration capture, selection, embedding, and persistence."""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import io
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image
from visual_memory_media_contract.images import decode_video_payload
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_memory_contract.protocol import ObjectViewQuality, ObjectViewUpload

from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.identity.base import MaskedCrop, SegmentedDetection, SegmentingDetector
from vision_worker.identity.crop import prepare_masked_crop
from vision_worker.identity.resolver import IdentityResolver
from vision_worker.identity.selection import (
    EmbeddedCandidate,
    QualityConfig,
    QualityScore,
    apply_relative_sharpness,
    score_quality,
    select_diverse,
)

logger = logging.getLogger(__name__)

EnrollmentState = Literal["capturing", "extracting", "succeeded", "failed"]


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
        segmenter: SegmentingDetector,
        identity_resolver: IdentityResolver,
        memory_client: MemoryClient,
        config: EnrollmentConfig,
    ) -> None:
        self._segmenter = segmenter
        self._resolver = identity_resolver
        self._memory = memory_client
        self._config = config

    async def enroll(
        self,
        *,
        object_id: str,
        label: str,
        frames: Sequence[BufferedFrame],
    ) -> EnrollmentResult:
        sampled = _subsample(frames, self._config.max_frames)
        candidates: list[tuple[MaskedCrop, QualityScore]] = []
        detections = 0
        for buffered in sampled:
            rgb = decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            segments = await self._segmenter.segment(rgb, labels=(label,))
            selected = _best_segment(segments, label)
            if selected is None:
                continue
            detections += 1
            quality = score_quality(
                rgb,
                selected.mask,
                selected.detection.box,
                selected.detection.confidence,
                angular_velocity=None,
                config=self._config.quality,
            )
            try:
                crop = prepare_masked_crop(
                    rgb,
                    selected.mask,
                    selected.detection.box,
                )
            except ValueError:
                continue
            candidates.append((crop, quality))

        relative_scores = apply_relative_sharpness(
            [quality for _, quality in candidates], config=self._config.quality
        )
        passed = [
            (crop, quality)
            for (crop, _), quality in zip(candidates, relative_scores, strict=True)
            if quality.accepted
        ]
        preliminary = EnrollmentResult(len(sampled), detections, len(passed), 0)
        if len(passed) < self._config.min_views:
            raise EnrollmentError(
                "too_few_quality_frames",
                f"only {len(passed)} frames passed quality; need {self._config.min_views}",
                result=preliminary,
            )

        embeddings = await self._resolver.embed_crops([crop for crop, _ in passed])
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
            raise EnrollmentError(
                "too_few_diverse_views",
                f"only {len(selected_views)} diverse views; need {self._config.min_views}",
                result=result,
            )

        try:
            for view_index, candidate in enumerate(selected_views):
                crop_bytes = _jpeg(candidate.crop.image)
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

        await self._resolver.refresh_gallery()
        return result


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


def _best_segment(segments: Sequence[SegmentedDetection], label: str) -> SegmentedDetection | None:
    matching = [segment for segment in segments if segment.detection.label == label]
    return max(
        matching,
        key=lambda segment: (
            segment.detection.confidence
            * max(0.0, segment.detection.box.x_max - segment.detection.box.x_min)
            * max(0.0, segment.detection.box.y_max - segment.detection.box.y_min)
        ),
        default=None,
    )


def _jpeg(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=92)
    return output.getvalue()


__all__ = [
    "EnrollmentConfig",
    "EnrollmentError",
    "EnrollmentManager",
    "EnrollmentProgress",
    "EnrollmentResult",
    "EnrollmentState",
    "ObjectEnroller",
]
