"""Per-track identity resolver: masked embeddings, gates, optional VLM escalation."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from PIL import Image
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_vision_contract.protocol import IdentityMatch

from vision_worker.identity.base import (
    EmbeddingVectors,
    IdentityEscalator,
    IdentityFrame,
    MaskedCrop,
    ObjectEmbedder,
    SegmentedDetection,
    SegmentingDetector,
    box_iou,
)
from vision_worker.identity.crop import prepare_masked_crop
from vision_worker.identity.gallery import GalleryCache, GalleryScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IdentityResolverConfig:
    min_cosine: float = 0.8334
    min_margin: float = 0.0440
    escalation_low: float = 0.8216
    summary_weight: float = 0.5


@dataclass(slots=True)
class IdentityMetrics:
    resolved: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    escalated: int = 0
    unavailable: int = 0
    resolutions: int = 0
    average_latency_ms: float = 0.0


class IdentityResolver:
    """Resolves once when the pipeline supplies enough track frames."""

    def __init__(
        self,
        *,
        segmenter: SegmentingDetector,
        embedder: ObjectEmbedder,
        gallery: GalleryCache,
        config: IdentityResolverConfig | None = None,
        escalator: IdentityEscalator | None = None,
    ) -> None:
        self._segmenter = segmenter
        self._embedder = embedder
        self._gallery = gallery
        self._config = config or IdentityResolverConfig()
        self._escalator = escalator
        self.metrics = IdentityMetrics()

    async def initialize(self) -> None:
        await self._embedder.initialize()
        await self._gallery.refresh(force=True)

    def accepts_label(self, label: str) -> bool:
        return self._embedder.is_ready and label in self._gallery.labels

    async def resolve(self, frames: Sequence[IdentityFrame]) -> IdentityMatch:
        started = time.perf_counter()
        self.metrics.resolutions += 1
        try:
            return await self._resolve(frames)
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            count = self.metrics.resolutions
            self.metrics.average_latency_ms = (
                self.metrics.average_latency_ms * (count - 1) + elapsed
            ) / count

    async def _resolve(self, frames: Sequence[IdentityFrame]) -> IdentityMatch:
        if not frames or not self._embedder.is_ready:
            self.metrics.unavailable += 1
            return IdentityMatch(reason_code="identity_unavailable")
        label = frames[0].detection.label
        await self._gallery.refresh()
        if label not in self._gallery.labels:
            self.metrics.unavailable += 1
            return IdentityMatch(reason_code="gallery_unavailable")

        crops: list[MaskedCrop] = []
        try:
            for frame in frames:
                segments = await self._segmenter.segment(frame.frame_rgb, labels=(label,))
                selected = _select_segment(frame, segments)
                if selected is None:
                    continue
                crops.append(
                    prepare_masked_crop(
                        frame.frame_rgb,
                        selected.mask,
                        selected.detection.box,
                    )
                )
            if not crops:
                self.metrics.unavailable += 1
                return IdentityMatch(reason_code="segmentation_unavailable")
            embeddings = await self._embedder.embed(crops)
        except Exception:
            self.metrics.unavailable += 1
            logger.exception("identity extraction failed; leaving the track unresolved")
            return IdentityMatch(reason_code="identity_unavailable")

        score = self._gallery.match(
            embeddings,
            label=label,
            summary_weight=self._config.summary_weight,
        )
        if score is None:
            self.metrics.unavailable += 1
            return IdentityMatch(reason_code="gallery_incompatible")
        margin_ok = score.margin is None or score.margin >= self._config.min_margin
        if score.score >= self._config.min_cosine and margin_ok:
            self.metrics.resolved += 1
            return _match(score, reason_code="embedding_resolved")

        near = score.score >= self._config.escalation_low
        if near and self._escalator is not None:
            self.metrics.escalated += 1
            verified = await self._escalator.verify(
                object_id=score.object_id,
                label=label,
                query_crops=crops,
                crop_references=score.crop_references,
            )
            if verified is True:
                self.metrics.resolved += 1
                return _match(score, reason_code="vlm_resolved", escalated=True)
            reason = "vlm_rejected" if verified is False else "vlm_unavailable"
            self.metrics.ambiguous += 1
            return _match(score, resolved=False, reason_code=reason, escalated=True)

        if not margin_ok:
            self.metrics.ambiguous += 1
            return _match(score, resolved=False, reason_code="ambiguous")
        self.metrics.unmatched += 1
        return _match(score, resolved=False, reason_code="below_threshold")

    async def embed_crops(self, crops: Sequence[MaskedCrop]) -> Sequence[EmbeddingVectors]:
        """Enrollment uses the exact same loaded embedder as matching."""
        return await self._embedder.embed(crops)

    async def refresh_gallery(self) -> None:
        await self._gallery.refresh(force=True)

    async def aclose(self) -> None:
        await self._embedder.aclose()

    def status_payload(self) -> Mapping[str, object]:
        return {
            **self._gallery.status_payload(),
            "resolved": self.metrics.resolved,
            "ambiguous": self.metrics.ambiguous,
            "unmatched": self.metrics.unmatched,
            "escalated": self.metrics.escalated,
            "unavailable": self.metrics.unavailable,
            "resolutions": self.metrics.resolutions,
            "average_latency_ms": round(self.metrics.average_latency_ms, 1),
            "embedder": dict(self._embedder.readiness_payload()),
            "thresholds": {
                "min_cosine": self._config.min_cosine,
                "min_margin": self._config.min_margin,
                "escalation_low": self._config.escalation_low,
                "summary_weight": self._config.summary_weight,
            },
        }


class VlmIdentityEscalator:
    """Qwen-compatible same-instance verifier over stored and query crops."""

    def __init__(
        self,
        client: MemoryClient,
        *,
        base_url: str,
        model: str,
        timeout_s: float,
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._model = model
        self._timeout_s = timeout_s

    async def verify(
        self,
        *,
        object_id: str,
        label: str,
        query_crops: Sequence[MaskedCrop],
        crop_references: Sequence[str],
    ) -> bool | None:
        try:
            references = await asyncio.gather(
                *(asyncio.to_thread(self._fetch_reference, ref) for ref in crop_references)
            )
            query_bytes = tuple(_jpeg(crop.image) for crop in query_crops[:3])
            reply = await asyncio.to_thread(
                self._ask_blocking,
                label,
                tuple(references),
                query_bytes,
            )
        except (MemoryError_, urllib.error.URLError, TimeoutError, OSError, ValueError):
            logger.warning(
                "identity VLM escalation unavailable",
                extra={"object_id": object_id, "label": label},
            )
            return None
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        payload = cast("dict[str, Any]", parsed)
        if not bool(payload.get("certain", False)):
            return None
        value = payload.get("same_instance")
        return value if isinstance(value, bool) else None

    def _fetch_reference(self, reference: str) -> bytes:
        parts = reference.strip("/").split("/")
        if len(parts) != 6 or parts[:2] != ["v1", "objects"] or parts[3] != "views":
            raise ValueError(f"invalid crop reference: {reference}")
        return self._client.get_object_crop(parts[2], parts[4])

    def _ask_blocking(
        self, label: str, references: Sequence[bytes], queries: Sequence[bytes]
    ) -> str:
        prompt = (
            f"The first {len(references)} images are stored reference views of one personal "
            f"{label}. The remaining {len(queries)} images are a new sighting. Decide only "
            "whether they show the same physical instance, not merely the same class. "
            "Return JSON with boolean same_instance and boolean certain. If distinguishing "
            "details are not visible, set certain=false."
        )
        images = [base64.b64encode(value).decode("ascii") for value in (*references, *queries)]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "format": {
                "type": "object",
                "properties": {
                    "same_instance": {"type": "boolean"},
                    "certain": {"type": "boolean"},
                },
                "required": ["same_instance", "certain"],
            },
            "stream": False,
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self._base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            body = json.load(response)
        return str(body.get("message", {}).get("content", ""))


def _select_segment(
    frame: IdentityFrame, segments: Sequence[SegmentedDetection]
) -> SegmentedDetection | None:
    matching = [segment for segment in segments if segment.detection.label == frame.detection.label]
    return max(
        matching,
        key=lambda segment: box_iou(frame.detection.box, segment.detection.box),
        default=None,
    )


def _match(
    score: GalleryScore,
    *,
    resolved: bool = True,
    reason_code: str,
    escalated: bool = False,
) -> IdentityMatch:
    return IdentityMatch(
        object_id=score.object_id if resolved else None,
        best_score=score.score,
        margin=score.margin,
        runner_up_object_id=score.runner_up_object_id,
        reason_code=reason_code,
        escalated=escalated,
    )


def _jpeg(image: Any) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG", quality=90)
    return output.getvalue()


__all__ = [
    "IdentityMetrics",
    "IdentityResolver",
    "IdentityResolverConfig",
    "VlmIdentityEscalator",
]
