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
from PIL import Image, ImageOps
from visual_memory_media_contract.images import decode_video_payload
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_memory_contract.protocol import ObjectViewQuality, ObjectViewUpload
from visual_memory_vision_contract.protocol import BoundingBox

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
from vision_worker.identity.track import DEFAULT_CENTRE_FRAC, Tracker, centre_box
from vision_worker.reason.base import Localizer

logger = logging.getLogger(__name__)

EnrollmentState = Literal["capturing", "extracting", "succeeded", "failed"]

#: How the reference boxes are obtained: ``grounded`` runs the VLM localizer
#: (voice registration); ``center-anchor`` propagates a fixed centre box with
#: the tracker (register button, no grounder, no speech).
EnrollmentMode = Literal["grounded", "center-anchor"]

#: Cosmos reports no detection score, so enrollment uses a fixed confidence
#: above the quality filter's floor -- the object being deliberately held up
#: and rotated is, by construction, a confident presence.
_ENROLL_CONFIDENCE = 0.9
_MANUAL_BOX = BoundingBox(x_min=0.01, y_min=0.01, x_max=0.99, y_max=0.99)
_MAX_MANUAL_CROP_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EnrollmentConfig:
    capture_seconds: float = 6.0
    max_capture_seconds: float = 15.0
    temporal_max_frames: int = 16
    temporal_batch_frames: int = 4
    candidate_interval_seconds: float = 0.75
    max_frames: int = 24
    target_views: int = 6
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
        tracker: Tracker | None = None,
        centre_frac: float = DEFAULT_CENTRE_FRAC,
    ) -> None:
        self._localizer = localizer
        self._embedder = embedder
        self._gallery = gallery
        self._memory = memory_client
        self._config = config
        self._box_padding = box_padding
        self._tracker = tracker
        self._centre_frac = centre_frac

    async def enroll(
        self,
        *,
        object_id: str,
        label: str,
        frames: Sequence[BufferedFrame],
    ) -> EnrollmentResult:
        frames_total = len(frames)
        coarse = _subsample_indexed(frames, self._config.temporal_max_frames)
        batches = tuple(
            coarse[index : index + self._config.temporal_batch_frames]
            for index in range(0, len(coarse), self._config.temporal_batch_frames)
        )
        temporal_results = await asyncio.gather(
            *(
                self._localizer.localize_sequence(
                    tuple(frame.payload for _source_index, frame in batch), label
                )
                for batch in batches
            )
        )
        hit_indices: set[int] = set()
        for batch, hits in zip(batches, temporal_results, strict=True):
            for hit in hits:
                if 0 <= hit.index < len(batch):
                    hit_indices.add(batch[hit.index][0])

        if not hit_indices:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_temporal_candidates",
                "the physical target was not recognizable across the capture",
                result=EnrollmentResult(frames_total, 0, 0, 0),
            )

        centers = tuple(frames[index].captured_at for index in sorted(hit_indices))
        interval_frames = tuple(
            frame
            for frame in frames
            if any(
                abs((frame.captured_at - center).total_seconds())
                <= self._config.candidate_interval_seconds
                for center in centers
            )
        )
        sampled = _subsample(interval_frames, self._config.max_frames)
        logger.info(
            "registration temporal search completed",
            extra={
                "object_id": object_id,
                "label": label,
                "source_frames": frames_total,
                "coarse_frames": len(coarse),
                "temporal_hits": len(hit_indices),
                "candidate_frames": len(sampled),
            },
        )

        # Ground the original relay frames only inside the temporal candidate
        # intervals. Calls remain independent so vLLM can dynamically batch
        # them, while the expensive image pass avoids irrelevant desk frames.
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

        preliminary = EnrollmentResult(frames_total, detections, len(physically_usable), 0)
        if len(physically_usable) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_quality_frames",
                f"only {len(physically_usable)} frames passed quality; "
                f"need {self._config.min_views}",
                result=preliminary,
            )

        # Refine when Cosmos can recognize the target inside its first crop,
        # but do not turn an abstention into an automatic zero-view failure.
        # Four live retries showed that recursive grounding is much less stable
        # on already masked 512px crops than on source frames: five to seven
        # useful first crops repeatedly became zero or one. The first strict
        # grounding remains the eligibility gate; this pass is an optional
        # tighter crop for the operator to review in the Console modal.
        semantic_boxes = await asyncio.gather(
            *(
                self._localizer.localize(encode_jpeg(crop.image), label)
                for crop, _quality in physically_usable
            )
        )
        review_candidates: list[tuple[MaskedCrop, QualityScore]] = []
        refined_count = 0
        for first_candidate, semantic_box in zip(physically_usable, semantic_boxes, strict=True):
            first_crop, _first_quality = first_candidate
            if semantic_box is None:
                review_candidates.append(first_candidate)
                continue
            height, width = first_crop.image.shape[:2]
            mask = box_to_mask(semantic_box, height, width, padding=self._box_padding)
            refined_quality = score_quality(
                first_crop.image,
                mask,
                semantic_box,
                _ENROLL_CONFIDENCE,
                angular_velocity=None,
                config=self._config.quality,
            )
            try:
                refined_crop = prepare_masked_crop(first_crop.image, mask, semantic_box)
            except ValueError:
                review_candidates.append(first_candidate)
                continue
            if not refined_quality.accepted:
                review_candidates.append(first_candidate)
                continue
            review_candidates.append((refined_crop, refined_quality))
            refined_count += 1

        # The contrastive judgment is a hard safety gate. Optional refinement
        # above prevents a second-pass abstention from erasing a usable first
        # crop, but a crop explicitly rejected as the wrong physical object
        # must never become a C-RADIO identity reference. In particular, the
        # keys prompt can otherwise confuse the prominent laptop keyboard with
        # the portable keys being enrolled.
        reference_results = await asyncio.gather(
            *(
                self._localizer.validate_reference(encode_jpeg(crop.image), label)
                for crop, _quality in review_candidates
            )
        )
        model_approved = [
            candidate
            for candidate, valid in zip(review_candidates, reference_results, strict=True)
            if valid
        ]
        passed = model_approved
        logger.info(
            "registration crop review completed",
            extra={
                "object_id": object_id,
                "label": label,
                "localized": detections,
                "physical_quality": len(physically_usable),
                "refined": refined_count,
                "reference_valid": len(model_approved),
            },
        )
        preliminary = EnrollmentResult(frames_total, detections, len(passed), 0)
        if len(passed) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_valid_references",
                f"only {len(passed)} frames clearly showed the physical target; "
                f"need {self._config.min_views}",
                result=preliminary,
            )

        return await self._persist_candidates(
            object_id=object_id,
            frames_total=frames_total,
            detections=detections,
            candidates=passed,
        )

    async def enroll_manual(
        self,
        *,
        object_id: str,
        crops: Sequence[bytes],
    ) -> EnrollmentResult:
        """Persist only operator-drawn crops submitted by explicit Confirm."""
        if not self._config.min_views <= len(crops) <= 8:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "invalid_manual_view_count",
                f"select between {self._config.min_views} and 8 views",
                result=EnrollmentResult(len(crops), 0, 0, 0),
            )

        candidates: list[tuple[MaskedCrop, QualityScore]] = []
        for payload in crops:
            try:
                rgb = _decode_manual_crop(payload)
            except ValueError:
                continue
            mask = np.ones(rgb.shape[:2], dtype=np.bool_)
            quality = score_quality(
                rgb,
                mask,
                _MANUAL_BOX,
                1.0,
                angular_velocity=None,
                config=self._config.quality,
            )
            try:
                crop = prepare_masked_crop(rgb, mask, _MANUAL_BOX)
            except ValueError:
                continue
            candidates.append((crop, quality))

        relative_scores = apply_relative_sharpness(
            [quality for _crop, quality in candidates], config=self._config.quality
        )
        passed = [
            (crop, quality)
            for (crop, _quality), quality in zip(candidates, relative_scores, strict=True)
            if quality.accepted
        ]
        if len(passed) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_manual_quality_views",
                f"only {len(passed)} operator crops passed image quality; "
                f"need {self._config.min_views}",
                result=EnrollmentResult(len(crops), len(candidates), len(passed), 0),
            )
        return await self._persist_candidates(
            object_id=object_id,
            frames_total=len(crops),
            detections=len(candidates),
            candidates=passed,
        )

    async def enroll_center_anchor(
        self,
        *,
        object_id: str,
        label: str,
        frames: Sequence[BufferedFrame],
    ) -> EnrollmentResult:
        """Register the object held centred, with no grounder (register button).

        Localisation is a fixed centre box propagated by SAM2 across the
        presentation -- the wearer said *which* object physically, so the VLM's
        "find the keys in the scene" is never asked. The crop/quality/persist
        tail is shared verbatim with the grounded path; only how the boxes are
        obtained differs.
        """
        if self._tracker is None:
            raise EnrollmentError(
                "tracker_unavailable",
                "center-anchor registration needs a tracker but none is configured",
                result=EnrollmentResult(len(frames), 0, 0, 0),
            )
        frames_total = len(frames)
        sampled = _subsample(frames, self._config.max_frames)
        rgb_frames = [
            decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            for buffered in sampled
        ]
        tracked = await self._tracker.track(rgb_frames, centre_box(self._centre_frac))
        logger.info(
            "center-anchor tracking completed",
            extra={
                "object_id": object_id,
                "label": label,
                "source_frames": frames_total,
                "tracked_frames": len(sampled),
                "tracked_hits": len(tracked),
            },
        )
        if not tracked:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_tracked_frames",
                "the centre anchor held no object across the presentation",
                result=EnrollmentResult(frames_total, 0, 0, 0),
            )

        candidates: list[tuple[MaskedCrop, QualityScore]] = []
        for frame_index, box in sorted(tracked.items()):
            if not 0 <= frame_index < len(rgb_frames):
                continue
            rgb = rgb_frames[frame_index]
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
        # A grab that is mostly hand or that never cohered fails the quality
        # gate here rather than being stored -- the "reject, don't register a
        # thumb" rule the register-button spec asks for.
        passed = [
            (crop, quality)
            for (crop, _), quality in zip(candidates, relative_scores, strict=True)
            if quality.accepted
        ]
        if len(passed) < self._config.min_views:
            await self._rollback_failed_object(object_id)
            raise EnrollmentError(
                "too_few_quality_frames",
                f"only {len(passed)} tracked frames passed quality; "
                f"need {self._config.min_views}",
                result=EnrollmentResult(frames_total, len(tracked), len(passed), 0),
            )
        return await self._persist_candidates(
            object_id=object_id,
            frames_total=frames_total,
            detections=len(tracked),
            candidates=passed,
        )

    async def _persist_candidates(
        self,
        *,
        object_id: str,
        frames_total: int,
        detections: int,
        candidates: Sequence[tuple[MaskedCrop, QualityScore]],
    ) -> EnrollmentResult:
        embeddings = await self._embedder.embed([crop for crop, _ in candidates])
        embedded = tuple(
            EmbeddedCandidate(crop=crop, embedding=embedding, quality=quality)
            for (crop, quality), embedding in zip(candidates, embeddings, strict=True)
        )
        selected_views = select_diverse(
            embedded,
            k=self._config.target_views,
            dedup_threshold=self._config.dedup_threshold,
            summary_weight=self._config.summary_weight,
        )
        result = EnrollmentResult(
            frames_total=frames_total,
            detections=detections,
            quality_passed=len(candidates),
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
        mode: EnrollmentMode = "grounded",
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
            self._run(progress, duration, mode), name=f"enroll-{object_id}"
        )
        return progress

    async def submit_manual(
        self,
        *,
        object_id: str,
        label: str,
        crops: Sequence[bytes],
    ) -> EnrollmentProgress:
        existing = self._tasks.get(object_id)
        if existing is not None and not existing.done():
            raise RuntimeError("registration capture is already active for this object")
        now = dt.datetime.now(dt.UTC)
        progress = EnrollmentProgress(
            object_id=object_id,
            label=label,
            state="extracting",
            started_at=now,
            capture_ends_at=now,
        )
        self._progress[object_id] = progress
        self.attempts += 1
        try:
            result = await self._enroller.enroll_manual(object_id=object_id, crops=crops)
        except EnrollmentError as exc:
            progress.state = "failed"
            progress.reason_code = exc.reason_code
            progress.message = str(exc)
            _apply_result(progress, exc.result)
            self.failed += 1
        else:
            progress.state = "succeeded"
            progress.reason_code = "manual_enrollment_complete"
            progress.message = "operator-confirmed reference gallery stored"
            _apply_result(progress, result)
            self.succeeded += 1
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

    async def _run(
        self, progress: EnrollmentProgress, duration: float, mode: EnrollmentMode = "grounded"
    ) -> None:
        try:
            await asyncio.sleep(duration)
            progress.state = "extracting"
            frames = self._ring.window(
                started_at=progress.started_at,
                ended_at=progress.capture_ends_at,
            )
            if mode == "center-anchor":
                result = await self._enroller.enroll_center_anchor(
                    object_id=progress.object_id,
                    label=progress.label,
                    frames=frames,
                )
            else:
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
            progress.message = "reference suggestions ready for review"
            _apply_result(progress, result)
            self.succeeded += 1


def _decode_manual_crop(payload: bytes) -> np.ndarray:
    if not payload or len(payload) > _MAX_MANUAL_CROP_BYTES:
        raise ValueError("manual crop has an invalid encoded size")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.width < 64 or image.height < 64:
                raise ValueError("manual crop is too small")
            if image.width > 4096 or image.height > 4096:
                raise ValueError("manual crop dimensions are too large")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            return np.asarray(image, dtype=np.uint8).copy()
    except (Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ValueError("manual crop is not a usable image") from exc


def _apply_result(progress: EnrollmentProgress, result: EnrollmentResult) -> None:
    progress.frames_total = result.frames_total
    progress.detections = result.detections
    progress.quality_passed = result.quality_passed
    progress.selected_views = result.selected_views


def _subsample_indexed(
    frames: Sequence[BufferedFrame], limit: int
) -> tuple[tuple[int, BufferedFrame], ...]:
    if len(frames) <= limit:
        return tuple(enumerate(frames))
    indexes = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return tuple((index, frames[index]) for index in dict.fromkeys(int(value) for value in indexes))


def _subsample(frames: Sequence[BufferedFrame], limit: int) -> tuple[BufferedFrame, ...]:
    return tuple(frame for _index, frame in _subsample_indexed(frames, limit))


__all__ = [
    "EnrollmentConfig",
    "EnrollmentError",
    "EnrollmentManager",
    "EnrollmentMode",
    "EnrollmentProgress",
    "EnrollmentResult",
    "EnrollmentState",
    "ObjectEnroller",
]
