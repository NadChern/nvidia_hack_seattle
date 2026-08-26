"""Orchestrates one video stream: relay frames in, confirmed observations out.

This is the VLM-reasoner pipeline. It replaces the old per-frame chain
(detect -> track -> stability -> depth/pose -> verify) with one question asked
of a vision-language model over a short sliding window:

    ring push  ──every reason_interval_s──▶  reasoner.analyze(window, labels)
                                                       │  {label, box, action, location}
                        box ▶ crop ▶ C-RADIOv4 ▶ gallery match ──▶ IdentityMatch
                                                       │
                  promotion policy ─▶ identity gate ─▶ emit Observation
                  (placed-only by default)       no match ─▶ skip

Implements `consume.relay.FrameSink`. `video_frame` only pushes to the
`EvidenceRing` and, on a cadence, snapshots a window and hands it to the
reasoner **off the frame loop** via `verify/pending.PendingVerifications` -- a
reasoner call takes seconds, and running it inline would make the gateway
discard seconds of frames (see `verify/pending.py`). One window is analyzed at
a time (one GPU, one model server), so a new window is scheduled only when the
previous analysis has finished.

The pipeline is torch-free: the reasoner and the embedder are behind Protocols
(`reason/base.py`, `identity/base.py`), so `tests/test_domain_isolation.py`
covers this module too. Identity is a **write gate**: only objects that match a
registered gallery entry are written to memory; unregistered clutter is
skipped, and the reasoner is only ever asked about the labels currently in the
gallery.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from visual_memory_media_contract.images import decode_video_payload
from visual_memory_media_contract.protocol import VideoFrame
from visual_memory_vision_contract.ids import new_candidate_id
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    IdentityMatch,
    Point2D,
    VerifierResult,
)

from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.identity.base import MaskedCrop, ObjectEmbedder
from vision_worker.identity.crop import box_to_mask, encode_jpeg, prepare_masked_crop
from vision_worker.identity.gallery import GalleryCache, GalleryScore
from vision_worker.reason.base import ReasonerLocalizer, WindowEvent

logger = logging.getLogger(__name__)

#: Called only for a matched, memory-worthy event, with the window's frames --
#: `emit.memory.MemoryEmitter.emit` is the intended implementation.
OnConfirmed = Callable[[CandidateEvent, VerifierResult, Sequence[BufferedFrame]], Awaitable[None]]

#: A synthetic "tracker" ref -- this pipeline has no tracker, but the candidate
#: contract carries one for provenance. Named so a recorded observation says
#: plainly that no per-frame tracker produced it.
_NO_TRACKER = DetectorRef(name="none", checkpoint="reasoner-window", revision="n/a")

_STATE_MACHINE_VERSION = "reasoner-v1"

_RECENT_EVENTS_MAXLEN = 200


@dataclass(slots=True)
class PipelineMetrics:
    """Counters reported at `/v1/status`."""

    frames_processed: int = 0
    windows_analyzed: int = 0
    #: Memory-worthy events the reasoner reported (placed/picked_up/carried).
    events_detected: int = 0
    #: Events whose crop matched a registered gallery object.
    identity_matched: int = 0
    #: Events with no gallery match -- deliberately not written (the write gate).
    identity_skipped: int = 0
    #: Motion events withheld by the placed-only promotion policy.
    motion_events_suppressed: int = 0
    #: Matched events suppressed as a repeat of a recent (object, action).
    events_deduped: int = 0
    #: Observations actually written to memory.
    observations_written: int = 0


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One reasoner event's outcome, for a human watching the pipeline work."""

    at: dt.datetime
    label: str
    action: str
    object_id: str | None
    outcome: str
    score: float | None


class Pipeline:
    """Drives the reasoner window loop for one relay video stream."""

    def __init__(
        self,
        *,
        reasoner: ReasonerLocalizer,
        embedder: ObjectEmbedder,
        gallery: GalleryCache,
        evidence_ring: EvidenceRing,
        on_confirmed: OnConfirmed,
        pipeline_version: str,
        reason_window_s: float = 6.0,
        reason_interval_s: float = 7.0,
        reason_max_frames: int = 4,
        identity_min_cosine: float = 0.75,
        identity_min_margin: float = 0.0,
        identity_summary_weight: float = 0.5,
        identity_pool_frames: int = 1,
        box_padding: float = 0.12,
        event_cooldown_s: float = 20.0,
        promote_motion_events: bool = False,
        work_queue_depth: int = 4,
    ) -> None:
        self._reasoner = reasoner
        self._embedder = embedder
        self._gallery = gallery
        self._evidence_ring = evidence_ring
        self._on_confirmed = on_confirmed
        self._pipeline_version = pipeline_version
        self._reason_window_s = reason_window_s
        self._reason_interval_s = reason_interval_s
        self._reason_max_frames = reason_max_frames
        self._identity_min_cosine = identity_min_cosine
        self._identity_min_margin = identity_min_margin
        self._identity_summary_weight = identity_summary_weight
        self._identity_pool_frames = max(1, identity_pool_frames)
        self._box_padding = box_padding
        self._event_cooldown_s = event_cooldown_s
        self._promote_motion_events = promote_motion_events

        self._current_epoch_id: str | None = None
        self._current_session_id: str | None = None
        self._current_device_id: str | None = None
        self._last_window_at: dt.datetime | None = None
        #: Last time an (object_id, action) was written, for cooldown dedup.
        self._last_emitted: dict[tuple[str, str], dt.datetime] = {}

        # One window at a time: a single model server behind the reasoner, so a
        # larger pool would only move the queue somewhere it cannot be counted.
        from vision_worker.verify.pending import PendingVerifications

        self._pending = PendingVerifications(depth=work_queue_depth, concurrency=1)
        self.metrics = PipelineMetrics()
        self._events: deque[PipelineEvent] = deque(maxlen=_RECENT_EVENTS_MAXLEN)

    # ------ Status surface -------------------------------------------------

    @property
    def evidence_ring(self) -> EvidenceRing:
        return self._evidence_ring

    @property
    def gallery(self) -> GalleryCache:
        return self._gallery

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @property
    def current_device_id(self) -> str | None:
        return self._current_device_id

    @property
    def pending_analyses(self) -> int:
        return self._pending.pending

    @property
    def analyses_dropped(self) -> int:
        return self._pending.dropped

    @property
    def analyses_failed(self) -> int:
        return self._pending.failed

    @property
    def recent_events(self) -> Sequence[PipelineEvent]:
        return tuple(self._events)

    # ------ FrameSink ------------------------------------------------------

    async def epoch_started(self, *, session_id: str, device_id: str, epoch_id: str) -> None:
        self._evidence_ring.reset()
        self._last_window_at = None
        self._last_emitted.clear()
        self._current_epoch_id = epoch_id
        self._current_session_id = session_id
        self._current_device_id = device_id
        logger.info("pipeline epoch reset", extra={"session_id": session_id, "epoch_id": epoch_id})

    async def epoch_ended(self, *, session_id: str, epoch_id: str) -> None:
        del session_id
        if epoch_id == self._current_epoch_id:
            self._current_session_id = None
            self._current_device_id = None

    async def video_frame(
        self, *, session_id: str, device_id: str, epoch_id: str, frame: VideoFrame
    ) -> None:
        if epoch_id != self._current_epoch_id:
            return
        self.metrics.frames_processed += 1
        self._evidence_ring.push(
            BufferedFrame(
                captured_at=frame.captured_at,
                payload=frame.payload,
                width=frame.width,
                height=frame.height,
            )
        )
        self._maybe_schedule_window(
            session_id=session_id, device_id=device_id, epoch_id=epoch_id, now=frame.captured_at
        )

    def _maybe_schedule_window(
        self, *, session_id: str, device_id: str, epoch_id: str, now: dt.datetime
    ) -> None:
        """Snapshot a window and hand it to the reasoner, on a cadence.

        Cadence gate and single-flight gate: the interval must have elapsed and
        the previous analysis must have finished (one model, one at a time).

        There is deliberately **no** "gallery is empty, skip" gate here. The
        gallery cache is refreshed inside `_analyze_window`, so gating window
        scheduling on a non-empty gallery would be a deadlock after a restart --
        the cache starts empty, so no window would schedule, so the cache would
        never refresh, so an object already registered in memory would stay
        invisible forever. Scheduling always and refreshing-then-checking inside
        the worker is what lets a prior registration reappear once frames flow;
        the refresh is TTL-guarded and cheap, and no Cosmos call is made while
        the gallery is empty.
        """
        if self._pending.pending > 0:
            return
        if (
            self._last_window_at is not None
            and (now - self._last_window_at).total_seconds() < self._reason_interval_s
        ):
            return
        window = self._evidence_ring.window(
            started_at=now - dt.timedelta(seconds=self._reason_window_s), ended_at=now
        )
        if not window:
            return
        self._last_window_at = now
        self._pending.submit(
            functools.partial(
                self._analyze_window,
                window=window,
                session_id=session_id,
                device_id=device_id,
                epoch_id=epoch_id,
            )
        )

    # ------ Window analysis (worker task) ----------------------------------

    async def _analyze_window(
        self,
        *,
        window: Sequence[BufferedFrame],
        session_id: str,
        device_id: str,
        epoch_id: str,
    ) -> None:
        await self._gallery.refresh()
        labels = tuple(self._gallery.labels)
        if not labels or not window:
            return
        self.metrics.windows_analyzed += 1
        events = await self._reasoner.analyze([f.payload for f in window], labels=labels)
        memory_events = [event for event in events if event.is_memory_event]
        self.metrics.events_detected += len(memory_events)

        promoted_events: list[WindowEvent] = []
        for event in memory_events:
            if event.action != "placed" and not self._promote_motion_events:
                self.metrics.motion_events_suppressed += 1
                self._record(event, None, "suppressed_by_policy", None)
                continue
            promoted_events.append(event)
        if not promoted_events:
            return

        for event in promoted_events:
            await self._handle_event(
                event=event,
                window=window,
                session_id=session_id,
                device_id=device_id,
                epoch_id=epoch_id,
            )

    async def _handle_event(
        self,
        *,
        event: WindowEvent,
        window: Sequence[BufferedFrame],
        session_id: str,
        device_id: str,
        epoch_id: str,
    ) -> None:
        occurred_at = window[-1].captured_at
        score = await self._resolve_identity(window, event)
        # Per-object gate: the winner carries its own bar (floor for a lone
        # object, raised by any same-label sibling it could be confused with).
        if score is None or score.score < score.threshold:
            self.metrics.identity_skipped += 1
            self._record(event, None, "skipped_no_identity", score)
            return
        self.metrics.identity_matched += 1

        key = (score.object_id, event.action)
        previous = self._last_emitted.get(key)
        within_cooldown = (
            previous is not None
            and (occurred_at - previous).total_seconds() < self._event_cooldown_s
        )
        if within_cooldown:
            self.metrics.events_deduped += 1
            self._record(event, score.object_id, "deduped", score)
            return

        candidate, result = self._build(event, score, session_id, device_id, epoch_id, window)
        await self._on_confirmed(candidate, result, window)
        self._last_emitted[key] = occurred_at
        self.metrics.observations_written += 1
        self._record(event, score.object_id, "written", score)

    async def _resolve_identity(
        self, window: Sequence[BufferedFrame], event: WindowEvent
    ) -> GalleryScore | None:
        """Refine the event box, embed the sighting's settled tail, match the gallery.

        The verdict is pooled over the last ``identity_pool_frames`` frames of the
        window rather than decided on one frame: per frame the accept/reject
        headroom is thin, and the gallery's median over a settled placement holds
        against a few mis-boxed or blurred frames (Spike 9b). Grounding stays
        per-sighting -- one ``localize`` on the representative (last) frame, whose
        box is reused across the tail -- because grounding is the expensive step
        and the object is at rest across these frames.

        Event grounding and enrollment both start with one strict box and try a
        tighter second localization. Cosmos may abstain on an already-masked crop
        even when the first crop is usable, so both fall back to the first crop
        rather than turning the identity gate into a permanent miss.
        """
        if not self._embedder.is_ready or not window:
            return None
        tail = window[-self._identity_pool_frames :]
        frames = [
            decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            for buffered in tail
        ]
        representative = frames[-1]
        rep_height, rep_width = representative.shape[:2]
        rep_mask = box_to_mask(event.box, rep_height, rep_width, padding=self._box_padding)
        if not rep_mask.any():
            return None
        try:
            rep_crop = prepare_masked_crop(representative, rep_mask, event.box)
            semantic_box = await self._reasoner.localize(encode_jpeg(rep_crop.image), event.label)
            crops = [self._sighting_crop(rgb, event.box, semantic_box) for rgb in frames]
            vectors = await self._embedder.embed([crop for crop in crops if crop is not None])
        except Exception:
            logger.exception("identity embedding failed; event left unmatched")
            return None
        if not vectors:
            return None
        return self._gallery.match(
            vectors,
            label=event.label,
            summary_weight=self._identity_summary_weight,
            floor=self._identity_min_cosine,
            confusion_margin=self._identity_min_margin,
        )

    def _sighting_crop(
        self, rgb: NDArray[np.uint8], box: BoundingBox, semantic_box: BoundingBox | None
    ) -> MaskedCrop | None:
        """One tail frame's crop, using the per-sighting boxes (no re-grounding)."""
        height, width = rgb.shape[:2]
        first_mask = box_to_mask(box, height, width, padding=self._box_padding)
        if not first_mask.any():
            return None
        first_crop = prepare_masked_crop(rgb, first_mask, box)
        if semantic_box is None:
            return first_crop
        crop_height, crop_width = first_crop.image.shape[:2]
        semantic_mask = box_to_mask(
            semantic_box, crop_height, crop_width, padding=self._box_padding
        )
        return prepare_masked_crop(first_crop.image, semantic_mask, semantic_box)

    def _build(
        self,
        event: WindowEvent,
        score: GalleryScore,
        session_id: str,
        device_id: str,
        epoch_id: str,
        window: Sequence[BufferedFrame],
    ) -> tuple[CandidateEvent, VerifierResult]:
        occurred_at = window[-1].captured_at
        box = event.box
        detection = Detection(
            label=event.label,
            confidence=event.confidence,
            box=box,
            centroid=Point2D(x=(box.x_min + box.x_max) / 2.0, y=(box.y_min + box.y_max) / 2.0),
        )
        identity = IdentityMatch(
            object_id=score.object_id,
            best_score=score.score,
            margin=score.margin,
            runner_up_object_id=score.runner_up_object_id,
            reason_code="resolved",
        )
        candidate_id = new_candidate_id()
        candidate = CandidateEvent(
            candidate_id=candidate_id,
            session_id=session_id,
            device_id=device_id,
            media_epoch_id=epoch_id,
            track_id=f"obj-{score.object_id}",
            label=event.label,
            action=event.action,  # type: ignore[arg-type]
            window=EvidenceWindow(
                window_started_at=window[0].captured_at,
                window_ended_at=occurred_at,
                frame_count=len(window),
            ),
            object_candidate=detection,
            detector=self._reasoner.ref,
            tracker=_NO_TRACKER,
            state_machine_version=_STATE_MACHINE_VERSION,
            pipeline_version=self._pipeline_version,
            identity=identity,
        )
        result = VerifierResult(
            candidate_id=candidate_id,
            outcome="confirmed",
            reason_code="reasoner_confirmed",
            latency_ms=0.0,
            verifier=self._reasoner.ref,
            prompt_version=self._reasoner.ref.revision,
            occurred_at=occurred_at,
            description=event.location_description,
        )
        return candidate, result

    def _record(
        self, event: WindowEvent, object_id: str | None, outcome: str, score: GalleryScore | None
    ) -> None:
        self._events.append(
            PipelineEvent(
                at=dt.datetime.now(dt.UTC),
                label=event.label,
                action=event.action,
                object_id=object_id,
                outcome=outcome,
                score=score.score if score is not None else None,
            )
        )

    # ------ Lifecycle ------------------------------------------------------

    async def drain(self) -> None:
        await self._pending.drain()

    async def aclose(self) -> None:
        await self.drain()
        await self._pending.aclose()


__all__ = [
    "OnConfirmed",
    "Pipeline",
    "PipelineEvent",
    "PipelineMetrics",
]
