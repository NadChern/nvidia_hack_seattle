"""Orchestrates one video stream: relay frames in, confirmed candidates out.

Implements `consume.relay.FrameSink`. Wires together every stage built so
far -- detect, track, pose, the stability machine, the evidence ring, and
the verifier -- into the pipeline the plan's critical-path claim depends on:
stop after this and the demo works end to end on recorded clips, with no
model required.

Every stage is swapped in through an interface (`Detector`, `Tracker`,
`PoseSource`, `Verifier`, `DepthEstimator`), so this module never imports
torch, ultralytics, or any model runtime -- `tests/test_domain_isolation.py`
covers this file too, not just `domain/`.

`depth_estimator` is optional and, when configured, runs at low cadence: once
per candidate about to be proposed, not once per frame -- see
`depth/moge.py`'s module docstring for why. It annotates `Detection.depth_m`
but never `TrackSample.world_point`, which needs a capture pose
(`domain/geometry.py`) that nothing in this service produces until task
#46's `DevicePose` lands; a `depth_estimator=None` pipeline is the same
honest image-space-only shape this module has always had.

Not every action the state machine emits becomes an observation. A first
sighting (`observed`) is logged and dropped here rather than verified and
uploaded -- see `_NON_PROMOTING_ACTIONS` for why, and for the plan criterion
that requires it.

**Proposing a candidate does not verify it.** `video_frame` hands the candidate
and its evidence to `verify/pending.PendingVerifications` and returns; the
verifier runs on a worker task. This is not an optimisation -- a verifier that
takes seconds, which the VLM one does, would otherwise make the gateway discard
seconds of frames and leave the stability machine reading a gap it cannot see.
`verify/pending.py` documents that failure in full.

Two consequences worth knowing. `on_confirmed` fires after `video_frame` has
already returned, so anything asserting on what was confirmed must `await
drain()` first. And candidates are answered in proposal order only while
`verification_concurrency` is 1.

One `Tracker`, one `PoseSource`, one `TrackRegistry`, and one `EvidenceRing`
for the whole pipeline, not one per epoch -- reset together whenever any
`epoch_started` fires. This does not support two sessions publishing
simultaneously; a demo runs one glasses stream at a time, and a later
multi-session deployment would key these per `epoch_id` instead.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray
from visual_memory_media_contract.images import decode_video_payload
from visual_memory_media_contract.protocol import VideoFrame
from visual_memory_vision_contract.ids import new_candidate_id
from visual_memory_vision_contract.protocol import (
    CandidateAction,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    IdentityMatch,
    OverlayFrame,
    OverlayTrack,
    TrackSample,
    VerifierOutcome,
    VerifierResult,
)

from vision_worker.depth.base import DepthEstimator
from vision_worker.detect.base import Detector
from vision_worker.domain.stability import TrackRegistry
from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.identity.base import IdentityFrame, IdentityResolverProtocol
from vision_worker.pose.base import PoseSource
from vision_worker.track.base import Tracker
from vision_worker.verify.base import Verifier
from vision_worker.verify.pending import PendingVerifications

logger = logging.getLogger(__name__)

#: Called only for a `confirmed` VerifierResult, with the window's buffered
#: frames -- see `emit.memory.MemoryEmitter.emit`, the intended implementation.
OnConfirmed = Callable[[CandidateEvent, VerifierResult, Sequence[BufferedFrame]], Awaitable[None]]

#: A registered track retired without a confirmed placement. This bounded weak
#: write bypasses the ordinary first-sighting drop and never runs a verifier.
OnObserved = Callable[[CandidateEvent, Sequence[BufferedFrame]], Awaitable[None]]

#: Called once per processed frame with what a viewer needs to draw it.
#:
#: Synchronous and non-awaitable **by type**, so it is impossible to pass
#: something that could block the frame loop -- see `overlay/hub.py` for why
#: that matters more here than it looks.
OverlaySink = Callable[[OverlayFrame], None]


class _LazyDecodedFrames(Sequence["NDArray[np.uint8]"]):
    """A window's frames as arrays, decoded on first access and cached.

    `len()` answers from the buffer without decoding anything, so a verifier
    that only checks whether evidence exists -- which the default one does --
    costs nothing, while one that reads pixels pays exactly for the frames it
    actually touches.
    """

    def __init__(self, frames: Sequence[BufferedFrame]) -> None:
        self._frames = frames
        self._decoded: dict[int, NDArray[np.uint8]] = {}

    def __len__(self) -> int:
        return len(self._frames)

    @overload
    def __getitem__(self, index: int) -> NDArray[np.uint8]: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[NDArray[np.uint8]]: ...

    def __getitem__(self, index: int | slice) -> NDArray[np.uint8] | Sequence[NDArray[np.uint8]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self._frames)))]
        if index < 0:
            index += len(self._frames)
        cached = self._decoded.get(index)
        if cached is None:
            buffered = self._frames[index]
            cached = decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            self._decoded[index] = cached
        return cached


@dataclass(slots=True)
class PipelineMetrics:
    """Counters reported at `/v1/status` (task #48)."""

    frames_processed: int = 0
    candidates_proposed: int = 0
    candidates_confirmed: int = 0
    candidates_rejected: int = 0
    candidates_unverified: int = 0
    candidates_skipped_empty_window: int = 0
    #: First sightings that were logged and then dropped -- see
    #: `_NON_PROMOTING_ACTIONS`. Counted rather than silent so the ratio of
    #: sightings to placements stays visible.
    sightings_not_promoted: int = 0
    #: Resting objects that stopped being detected and had to be asked about.
    vanishings_questioned: int = 0
    identity_tracks_started: int = 0
    identity_tracks_resolved: int = 0
    registered_last_seen_emitted: int = 0


#: Actions that never become a trusted observation, however confident.
#:
#: A first sighting is worth watching (`/v1/events`) but is not evidence that
#: anything happened: `application_memory.domain.reducer` explicitly does not
#: create a placement from an `observed`, so uploading a still and an encoded
#: clip and writing an observation for every object that enters frame spends
#: a clip encode and two HTTP round-trips to move a pointer. Prompt-free
#: detection makes that every object in the room.
#:
#: The plan is explicit about the acceptance criterion: clips 4 and 5
#: ("object visible, never touched"; "walking past an object") must produce
#: zero observations. `domain/stability.py` still emits `observed` -- the
#: state machine's job is to describe what it saw, not to decide what is
#: worth storing. This is where that decision belongs.
_NON_PROMOTING_ACTIONS: frozenset[CandidateAction] = frozenset({"observed"})

#: How many frame intervals the observed-rate estimate averages over, and the
#: minimum before it will report anything. ~5s of a 24fps stream, ~30s of a
#: 2fps one -- long enough that one slow frame does not swing it.
_RATE_WINDOW_FRAMES = 120

#: Per-track sample history retained for a verifier's window. 480 frames is
#: a minute at 8fps -- longer than any evidence ring this service would be
#: configured with, so the ring, not this, is what bounds a window.
_SAMPLE_HISTORY_MAXLEN = 480

#: How far the measured rate may drift from the configured `source_fps`
#: before it is worth a warning. Every stability threshold is a frame count
#: derived from that setting, so a rate that is wrong by more than this
#: scales all of them by the same factor.
_RATE_TOLERANCE = 0.25


#: Bounded so a long-running dev session's memory footprint stays flat --
#: this is a live-viewing aid (`/v1/events`, the publisher page's debug
#: panel), not a durable audit trail. `application_memory`'s `audit` table
#: is the actual record; this is allowed to lose old entries.
_RECENT_EVENTS_MAXLEN = 200


#: `not_promoted` is not a verifier outcome and never reaches a verifier --
#: it is this pipeline saying it declined to propose a candidate at all. Kept
#: distinct from `unverified` so the activity log does not imply a verifier
#: looked and could not decide.
PipelineOutcome: TypeAlias = VerifierOutcome | Literal["not_promoted"]

#: The reason code paired with `not_promoted`.
OBSERVED_NOT_PROMOTED = "observed_not_promoted"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One candidate's outcome, for a human watching the pipeline work."""

    at: dt.datetime
    track_id: str
    label: str
    action: CandidateAction
    outcome: PipelineOutcome
    reason_code: str
    confidence: float


class Pipeline:
    """Drives every stage from one relay video stream."""

    def __init__(
        self,
        *,
        detector: Detector,
        detector_ref: DetectorRef,
        tracker: Tracker,
        tracker_ref: DetectorRef,
        pose_source: PoseSource,
        track_registry: TrackRegistry,
        evidence_ring: EvidenceRing,
        verifier: Verifier,
        detection_labels: Sequence[str],
        state_machine_version: str,
        pipeline_version: str,
        on_confirmed: OnConfirmed,
        source_fps: float,
        vanish_lookback_s: float = 3.0,
        depth_estimator: DepthEstimator | None = None,
        depth_model_ref: DetectorRef | None = None,
        verification_queue_depth: int = 8,
        verification_concurrency: int = 1,
        overlay_sink: OverlaySink | None = None,
        overlay_depth_interval_s: float = 1.0,
        max_detections_per_frame: int = 20,
        identity_resolver: IdentityResolverProtocol | None = None,
        identity_track_frames: int = 3,
        identity_min_detection_confidence: float = 0.5,
        identity_min_scale: float = 0.01,
        on_observed: OnObserved | None = None,
    ) -> None:
        self._detector = detector
        self._detector_ref = detector_ref
        self._tracker = tracker
        self._tracker_ref = tracker_ref
        self._pose_source = pose_source
        self._track_registry = track_registry
        self._evidence_ring = evidence_ring
        self._verifier = verifier
        self._detection_labels = tuple(detection_labels)
        self._max_detections_per_frame = max_detections_per_frame
        self._state_machine_version = state_machine_version
        self._pipeline_version = pipeline_version
        self._on_confirmed = on_confirmed
        self._on_observed = on_observed
        self._identity_resolver = identity_resolver
        self._identity_track_frames = identity_track_frames
        self._identity_min_detection_confidence = identity_min_detection_confidence
        self._identity_min_scale = identity_min_scale
        #: `None` means nothing is watching, and costs nothing: no overlay is
        #: assembled at all, which is the normal state of a deployed service.
        self._overlay_sink = overlay_sink
        #: `None` means "no depth adapter configured" -- every candidate then
        #: carries `depth_model=None`, `object_candidate.depth_m=None`, the
        #: same honest image-space-only shape the pipeline has always had.
        self._depth_estimator = depth_estimator
        self._depth_model_ref = depth_model_ref

        self._source_fps = source_fps
        #: How far before its last sighting a vanish window reaches. The
        #: interesting moment is the approach -- a hand arriving -- not the
        #: empty table afterwards, so the window looks backwards.
        self._vanish_lookback_s = vanish_lookback_s
        #: Per-track sample history, so a verifier can be handed where the
        #: object was across its window rather than only where it ended up.
        #: Bounded by the same retention the evidence ring uses -- a sample
        #: whose frame has already been evicted can never be part of a window.
        self._samples: dict[str, deque[TrackSample]] = {}
        #: Identity is resolved once per track from a small bounded frame set.
        #: The tasks run independently of verification and never block the
        #: detector loop; their result is cached until retirement or epoch reset.
        self._identity_frames: dict[str, list[IdentityFrame]] = {}
        self._identity_attempted: set[str] = set()
        self._identity_by_track: dict[str, IdentityMatch] = {}
        self._identity_tasks: dict[str, asyncio.Task[None]] = {}
        self._last_seen_tasks: set[asyncio.Task[None]] = set()
        #: Last depth reading per track, and when it was taken. Depth is
        #: sampled at a cadence rather than per frame, so a viewer sees the
        #: most recent measurement plus its age -- never a stale number
        #: presented as if it were live.
        self._depth_by_track: dict[str, tuple[float, dt.datetime]] = {}
        self._depth_sampled_at: dt.datetime | None = None
        self._overlay_depth_interval_s = overlay_depth_interval_s
        self._current_epoch_id: str | None = None
        self._current_session_id: str | None = None
        self._current_device_id: str | None = None
        #: Verification runs here, never on the frame loop. One worker by
        #: default: there is one GPU and one model server behind `verify/vlm`,
        #: so a larger pool would only move the queue somewhere it cannot be
        #: seen or counted.
        self._pending = PendingVerifications(
            depth=verification_queue_depth, concurrency=verification_concurrency
        )
        self.metrics = PipelineMetrics()
        self._events: deque[PipelineEvent] = deque(maxlen=_RECENT_EVENTS_MAXLEN)
        self._frame_intervals: deque[float] = deque(maxlen=_RATE_WINDOW_FRAMES)
        self._last_frame_at: dt.datetime | None = None
        self._warned_about_rate = False

    @property
    def detector(self) -> Detector:
        """Registration reuses the initialized segmentation-capable detector."""
        return self._detector

    @property
    def evidence_ring(self) -> EvidenceRing:
        return self._evidence_ring

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @property
    def current_device_id(self) -> str | None:
        return self._current_device_id

    @property
    def observed_fps(self) -> float | None:
        """The rate frames are actually arriving at, from their own
        `captured_at` stamps -- `None` until enough have arrived to mean
        anything.

        Reported at `/v1/status`. Every stability threshold is a frame count
        derived from the configured `source_fps`, and nothing else in the
        system fails loudly when that setting disagrees with the gateway's
        `VMA_SAMPLE_FPS`: the thresholds just quietly come out scaled. This is
        the measurement that catches it.
        """
        if len(self._frame_intervals) < _RATE_WINDOW_FRAMES // 4:
            return None
        mean_interval = sum(self._frame_intervals) / len(self._frame_intervals)
        return 1.0 / mean_interval if mean_interval > 0 else None

    @property
    def pending_verifications(self) -> int:
        """Candidates proposed and not yet answered, including the one being
        verified right now. Reported at `/v1/status`: a number that climbs and
        does not come back down is verification falling behind the stream."""
        return self._pending.pending

    @property
    def verifications_dropped(self) -> int:
        """Candidates discarded because verification could not keep up.

        Must be zero. Every one of these is a real event -- a pickup, a
        placement -- that was seen, proposed, and then never recorded, so this
        is the counter that says the service is quietly forgetting things.
        """
        return self._pending.dropped

    @property
    def verifications_failed(self) -> int:
        """Candidates whose verifier raised. Distinct from `unverified`, which
        is a verifier that looked and declined to decide."""
        return self._pending.failed

    @property
    def recent_events(self) -> Sequence[PipelineEvent]:
        """Oldest first, for `/v1/events` -- a live-viewing aid, not the
        canonical record (that is `application_memory`'s `audit` table,
        reached only for `confirmed` outcomes via `on_confirmed`)."""
        return tuple(self._events)

    @property
    def track_registry(self) -> TrackRegistry:
        """Read-only access to the thresholds actually in effect, for
        `/v1/status` -- the plan requires configuration to be reported, not
        just held, matching how the Memory Service reports its
        `PromotionPolicy`."""
        return self._track_registry

    async def epoch_started(self, *, session_id: str, device_id: str, epoch_id: str) -> None:
        """Reset every piece of per-epoch state together. `track_id` is only
        ever meaningful within one epoch; carrying any of this across a
        reconnect is the exact trap docs/06 warns about."""
        self._tracker.reset()
        self._pose_source.reset()
        self._track_registry.reset()
        self._evidence_ring.reset()
        self._samples.clear()
        for task in self._identity_tasks.values():
            task.cancel()
        self._identity_frames.clear()
        self._identity_attempted.clear()
        self._identity_by_track.clear()
        self._identity_tasks.clear()
        self._depth_by_track.clear()
        self._depth_sampled_at = None
        self._frame_intervals.clear()
        self._last_frame_at = None
        self._current_epoch_id = epoch_id
        self._current_session_id = session_id
        self._current_device_id = device_id
        logger.info("pipeline epoch reset", extra={"session_id": session_id, "epoch_id": epoch_id})

    async def epoch_ended(self, *, session_id: str, epoch_id: str) -> None:
        # State stays until the next epoch_started resets it -- any
        # verification still in flight for this epoch's last frames should
        # complete against valid state rather than an emptied one.
        del session_id
        if epoch_id == self._current_epoch_id:
            self._current_session_id = None
            self._current_device_id = None

    async def video_frame(
        self, *, session_id: str, device_id: str, epoch_id: str, frame: VideoFrame
    ) -> None:
        if epoch_id != self._current_epoch_id:
            # A frame for an epoch this pipeline never got epoch_started for
            # (e.g. arriving right after a reset) -- process nothing against
            # state that belongs to a different epoch.
            return

        self.metrics.frames_processed += 1
        self._record_frame_interval(frame.captured_at)
        rgb = frame.rgb
        self._evidence_ring.push(
            BufferedFrame(
                captured_at=frame.captured_at,
                payload=frame.payload,
                width=frame.width,
                height=frame.height,
            )
        )
        background_motion = self._pose_source.observe(rgb)

        detections = await self._detector.detect(rgb, labels=self._detection_labels)
        if len(detections) > self._max_detections_per_frame:
            detections = tuple(
                sorted(detections, key=lambda detection: detection.confidence, reverse=True)[
                    : self._max_detections_per_frame
                ]
            )
        matches = self._tracker.update(detections)
        matched_ids = {track_id for track_id, _ in matches}

        # The registry is the single owner of which ids are still live: it
        # drops a track once it has been absent past `reacquire_within_frames`
        # (`StabilityStep.retired`), and this sweep shrinks with it. Keeping a
        # second set here instead would never shrink, so both the memory and
        # the per-frame work would grow with every id the tracker ever minted.
        for lost_track_id in self._track_registry.active_track_ids - matched_ids:
            lost = self._track_registry.observe(lost_track_id, None)
            if lost.action is not None:
                # A confirmed placement disappearing asks the stronger
                # `vanished` question. It must never also emit weak last-seen.
                await self._propose_vanished(
                    session_id=session_id,
                    device_id=device_id,
                    epoch_id=epoch_id,
                    track_id=lost_track_id,
                    action=lost.action,
                    now=frame.captured_at,
                )
            elif lost.retired:
                self._schedule_last_seen(
                    session_id=session_id,
                    device_id=device_id,
                    epoch_id=epoch_id,
                    track_id=lost_track_id,
                )

        # Retired tracks take their sample history with them, for the same
        # reason their state goes: nothing can ever ask about them again.
        for gone in self._samples.keys() - self._track_registry.active_track_ids - matched_ids:
            del self._samples[gone]
            self._depth_by_track.pop(gone, None)
            self._identity_frames.pop(gone, None)
            self._identity_attempted.discard(gone)
            self._identity_by_track.pop(gone, None)
            task = self._identity_tasks.pop(gone, None)
            if task is not None and not task.done():
                task.cancel()

        # Built only when something is watching. `None` skips the work
        # entirely rather than assembling a list nobody reads.
        overlay_tracks: list[OverlayTrack] | None = [] if self._overlay_sink is not None else None

        if overlay_tracks is not None and self._should_sample_depth(frame.captured_at):
            await self._sample_depth(rgb, matches, now=frame.captured_at)

        for track_id, detection in matches:
            sample = TrackSample(
                track_id=track_id,
                frame_index=frame.sequence,
                captured_at=frame.captured_at,
                detection=detection,
                # `None` until a capture pose exists to back-project through
                # (`domain/geometry.py`) -- gated on task #46's `DevicePose`,
                # not on the depth adapter below, which only supplies the
                # `depth_m` half of that computation.
                world_point=None,
                background_motion=background_motion,
                identity=self._identity_by_track.get(track_id),
            )
            self._samples.setdefault(track_id, deque(maxlen=_SAMPLE_HISTORY_MAXLEN)).append(sample)
            self._maybe_schedule_identity(track_id, detection, rgb, epoch_id=epoch_id)
            result = self._track_registry.observe(track_id, sample)

            # Collected here, above every `continue` below, because a viewer
            # wants to see *every* tracked object -- including the ones this
            # pipeline deliberately declines to promote. A console that only
            # drew objects worth remembering would show an empty frame for the
            # commonest case and look broken.
            if overlay_tracks is not None:
                overlay_tracks.append(
                    OverlayTrack(
                        track_id=track_id,
                        label=detection.label,
                        confidence=detection.confidence,
                        box=detection.box,
                        motion_state=result.state.motion_state,
                        **self._depth_for(track_id, now=frame.captured_at),
                    )
                )

            if result.action is None:
                continue

            if result.action in _NON_PROMOTING_ACTIONS:
                # Logged for the activity panel, then dropped -- no depth
                # inference, no evidence window, no verifier call, no upload.
                # See `_NON_PROMOTING_ACTIONS`.
                self.metrics.sightings_not_promoted += 1
                self._events.append(
                    PipelineEvent(
                        at=frame.captured_at,
                        track_id=track_id,
                        label=detection.label,
                        action=result.action,
                        outcome="not_promoted",
                        reason_code=OBSERVED_NOT_PROMOTED,
                        confidence=detection.confidence,
                    )
                )
                continue

            candidate_detection = detection
            if self._depth_estimator is not None:
                # Low cadence, on settling: depth runs only for the frame
                # that is about to become a candidate's evidence, not on
                # every frame -- see depth/moge.py's module docstring.
                try:
                    [candidate_detection] = await self._depth_estimator.estimate(rgb, (detection,))
                except Exception:
                    # The same rule `depth/moge.py` applies to loading applies
                    # to running: a pipeline that annotates nothing is degraded,
                    # not broken. The candidate still goes forward with
                    # `depth_m=None`, which is a shape the emitter and the
                    # memory contract already treat as ordinary.
                    logger.exception("candidate depth estimation failed; proposing without depth")
                    candidate_detection = detection

            await self._propose_candidate(
                session_id=session_id,
                device_id=device_id,
                epoch_id=epoch_id,
                track_id=track_id,
                action=result.action,
                detection=candidate_detection,
                started_at=result.state.state_started_at or frame.captured_at,
                ended_at=frame.captured_at,
            )

        if overlay_tracks is not None:
            self._publish_overlay(session_id=session_id, frame=frame, tracks=overlay_tracks)

    def _maybe_schedule_identity(
        self,
        track_id: str,
        detection: Detection,
        rgb: NDArray[np.uint8],
        *,
        epoch_id: str,
    ) -> None:
        resolver = self._identity_resolver
        if (
            resolver is None
            or track_id in self._identity_attempted
            or not resolver.accepts_label(detection.label)
        ):
            return
        box = detection.box
        area = max(0.0, box.x_max - box.x_min) * max(0.0, box.y_max - box.y_min)
        if (
            detection.confidence < self._identity_min_detection_confidence
            or area < self._identity_min_scale
            or box.x_min <= 0.0
            or box.y_min <= 0.0
            or box.x_max >= 1.0
            or box.y_max >= 1.0
        ):
            return
        frames = self._identity_frames.setdefault(track_id, [])
        frames.append(IdentityFrame(frame_rgb=rgb.copy(), detection=detection))
        if len(frames) < self._identity_track_frames:
            return
        self._identity_attempted.add(track_id)
        selected = tuple(frames[: self._identity_track_frames])
        self._identity_frames.pop(track_id, None)
        self.metrics.identity_tracks_started += 1
        task = asyncio.create_task(
            self._resolve_track_identity(track_id, selected, epoch_id=epoch_id),
            name=f"identity-{track_id}",
        )
        self._identity_tasks[track_id] = task

    async def _resolve_track_identity(
        self,
        track_id: str,
        frames: Sequence[IdentityFrame],
        *,
        epoch_id: str,
    ) -> None:
        resolver = self._identity_resolver
        if resolver is None:
            return
        try:
            match = await resolver.resolve(frames)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("per-track identity resolution failed; track remains unregistered")
            return
        finally:
            current = asyncio.current_task()
            if self._identity_tasks.get(track_id) is current:
                self._identity_tasks.pop(track_id, None)
        if (
            epoch_id != self._current_epoch_id
            or track_id not in self._track_registry.active_track_ids
        ):
            return
        self._identity_by_track[track_id] = match
        if match.object_id is not None:
            self.metrics.identity_tracks_resolved += 1

    def _schedule_last_seen(
        self,
        *,
        session_id: str,
        device_id: str,
        epoch_id: str,
        track_id: str,
    ) -> None:
        identity = self._identity_by_track.get(track_id)
        history = self._samples.get(track_id)
        if (
            self._on_observed is None
            or identity is None
            or identity.object_id is None
            or not history
        ):
            return
        last = history[-1]
        started_at = last.captured_at - dt.timedelta(seconds=1.0 / self._source_fps)
        frames = self._evidence_ring.window(
            started_at=started_at,
            ended_at=last.captured_at,
        )
        if not frames:
            return
        candidate = CandidateEvent(
            candidate_id=new_candidate_id(),
            session_id=session_id,
            device_id=device_id,
            media_epoch_id=epoch_id,
            track_id=track_id,
            label=last.detection.label,
            action="observed",
            window=EvidenceWindow(
                window_started_at=started_at,
                window_ended_at=last.captured_at,
                frame_count=len(frames),
            ),
            object_candidate=last.detection,
            detector=self._detector_ref,
            tracker=self._tracker_ref,
            state_machine_version=self._state_machine_version,
            pipeline_version=self._pipeline_version,
            identity=identity,
        )
        task = asyncio.create_task(
            self._write_last_seen(candidate, frames),
            name=f"last-seen-{track_id}",
        )
        self._last_seen_tasks.add(task)
        task.add_done_callback(self._last_seen_finished)
        self.metrics.registered_last_seen_emitted += 1

    async def _write_last_seen(
        self, candidate: CandidateEvent, frames: Sequence[BufferedFrame]
    ) -> None:
        callback = self._on_observed
        if callback is not None:
            await callback(candidate, frames)

    def _last_seen_finished(self, task: asyncio.Task[None]) -> None:
        self._last_seen_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("registered track-end last-seen write failed")

    def _should_sample_depth(self, now: dt.datetime) -> bool:
        """Whether to spend a depth pass on this frame.

        Three conditions, all necessary. There must be an adapter; something
        must be watching, since this exists only to put a number on a box; and
        the cadence must have elapsed. A deployment with no viewer attached --
        which is most of them, most of the time -- never runs depth here at
        all, and the candidate path's own low-cadence call is unaffected.
        """
        if self._depth_estimator is None:
            return False
        if self._depth_sampled_at is None:
            return True
        return (now - self._depth_sampled_at).total_seconds() >= self._overlay_depth_interval_s

    async def _sample_depth(
        self,
        rgb: NDArray[np.uint8],
        matches: Sequence[tuple[str, Detection]],
        *,
        now: dt.datetime,
    ) -> None:
        """Measure depth for everything currently tracked, and remember it.

        Runs on the frame loop, deliberately. A depth pass is slow enough to
        make this frame late, but at one pass per second that is a handful of
        dropped frames -- and `consume/relay.py` now drops rather than
        accumulating lag, so the cost is bounded and visible instead of
        compounding. Moving it to a worker would mean holding the frame and
        reconciling a result that arrives after the track it describes has
        moved, which is more machinery than a once-a-second number deserves.
        """
        if self._depth_estimator is None or not matches:
            return
        self._depth_sampled_at = now
        try:
            annotated = await self._depth_estimator.estimate(
                rgb, [detection for _, detection in matches]
            )
        except Exception:
            # Depth is an enhancement; `depth/moge.py` is explicit that a
            # pipeline annotating nothing is degraded rather than broken.
            logger.exception("overlay depth sampling failed; keeping previous readings")
            return

        for (track_id, _), detection in zip(matches, annotated, strict=False):
            if detection.depth_m is not None:
                self._depth_by_track[track_id] = (detection.depth_m, now)

    def _depth_for(self, track_id: str, *, now: dt.datetime) -> dict[str, float | None]:
        """This track's last depth reading and how old it is.

        The age travels with the value on purpose: a number shown as if it were
        live when it is seconds stale is a worse failure than showing none.
        """
        reading = self._depth_by_track.get(track_id)
        if reading is None:
            return {"depth_m": None, "depth_age_s": None}
        depth_m, measured_at = reading
        return {
            "depth_m": depth_m,
            "depth_age_s": round(max(0.0, (now - measured_at).total_seconds()), 2),
        }

    def _publish_overlay(
        self, *, session_id: str, frame: VideoFrame, tracks: Sequence[OverlayTrack]
    ) -> None:
        """Hand this frame's detections to whatever is watching.

        Wrapped in a catch-all on purpose. A viewer is a debugging and demo
        surface; a bug in it, or in serializing an overlay, must never be able
        to stop the pipeline from processing video. The service's job continues
        whether or not anyone is looking at it.
        """
        assert self._overlay_sink is not None
        emitted_at = dt.datetime.now(dt.UTC)
        try:
            self._overlay_sink(
                OverlayFrame(
                    session_id=session_id,
                    media_epoch_id=frame.epoch_id,
                    sequence=frame.sequence,
                    captured_at=frame.captured_at,
                    relayed_at=frame.relayed_at,
                    emitted_at=emitted_at,
                    width=frame.width,
                    height=frame.height,
                    tracks=tuple(tracks),
                    # Both stamps come from this process, so this measures the
                    # pipeline rather than the gap between two machines' clocks.
                    # Clamped at zero: a relay stamp fractionally ahead of this
                    # one is a clock adjustment, not negative work.
                    pipeline_latency_ms=max(
                        0.0, (emitted_at - frame.relayed_at).total_seconds() * 1000.0
                    ),
                )
            )
        except Exception:
            logger.exception("overlay publication failed; continuing")

    def _record_frame_interval(self, captured_at: dt.datetime) -> None:
        """Track the real inter-frame interval, and say something once if it
        disagrees with what the thresholds were built from."""
        previous = self._last_frame_at
        self._last_frame_at = captured_at
        if previous is None:
            return

        interval = (captured_at - previous).total_seconds()
        if interval <= 0:
            # Duplicate or out-of-order stamps say nothing about the rate.
            return
        self._frame_intervals.append(interval)

        measured = self.observed_fps
        if measured is None or self._warned_about_rate:
            return
        if abs(measured - self._source_fps) / self._source_fps > _RATE_TOLERANCE:
            self._warned_about_rate = True
            logger.warning(
                "relay frame rate disagrees with VMA_SOURCE_FPS -- every stability "
                "threshold is a frame count derived from it, so they are all "
                "scaled by this ratio",
                extra={"configured_fps": self._source_fps, "observed_fps": round(measured, 2)},
            )

    async def _propose_vanished(
        self,
        *,
        session_id: str,
        device_id: str,
        epoch_id: str,
        track_id: str,
        action: CandidateAction,
        now: dt.datetime,
    ) -> None:
        """Propose a candidate for a track that is no longer in frame.

        Unlike every other candidate, there is no current detection to attach
        -- that is the whole point. The last sample this track produced stands
        in as `object_candidate`: it is where the object was when it was last
        genuinely seen, which is exactly what a verifier needs to reason about
        its disappearance.

        The window reaches **backwards** from that last sighting rather than
        around a current frame. What matters happened just before the object
        stopped being detected -- a hand entering, a lid closing -- and those
        frames are still in the ring.
        """
        history = self._samples.get(track_id)
        if not history:
            # Retired without ever producing a sample this epoch. Nothing to
            # describe and nothing to reason about.
            return

        self.metrics.vanishings_questioned += 1
        last = history[-1]
        started_at = last.captured_at - dt.timedelta(seconds=self._vanish_lookback_s)
        await self._propose_candidate(
            session_id=session_id,
            device_id=device_id,
            epoch_id=epoch_id,
            track_id=track_id,
            action=action,
            detection=last.detection,
            started_at=started_at,
            ended_at=now,
            samples=tuple(s for s in history if s.captured_at >= started_at),
        )

    async def _propose_candidate(
        self,
        *,
        session_id: str,
        device_id: str,
        epoch_id: str,
        track_id: str,
        action: CandidateAction,
        detection: Detection,
        started_at: dt.datetime,
        ended_at: dt.datetime,
        samples: Sequence[TrackSample] | None = None,
    ) -> None:
        window_frames = self._evidence_ring.window(started_at=started_at, ended_at=ended_at)
        if not window_frames:
            # EvidenceWindow.frame_count requires at least 1 -- an empty
            # window (e.g. an action firing on the first frame after a reset,
            # before the ring has accumulated anything in range) cannot even
            # be represented as a candidate, let alone verified. Skip rather
            # than raise: this is expected at an epoch's very start, not a bug.
            self.metrics.candidates_skipped_empty_window += 1
            logger.warning(
                "stability action with no buffered evidence in its window -- skipped",
                extra={"track_id": track_id, "action": action},
            )
            return

        candidate = CandidateEvent(
            candidate_id=new_candidate_id(),
            session_id=session_id,
            device_id=device_id,
            media_epoch_id=epoch_id,
            track_id=track_id,
            label=detection.label,
            action=action,
            window=EvidenceWindow(
                window_started_at=started_at,
                window_ended_at=ended_at,
                frame_count=len(window_frames),
            ),
            object_candidate=detection,
            detector=self._detector_ref,
            tracker=self._tracker_ref,
            depth_model=self._depth_model_ref if detection.depth_m is not None else None,
            state_machine_version=self._state_machine_version,
            pipeline_version=self._pipeline_version,
            identity=self._identity_by_track.get(track_id),
        )
        self.metrics.candidates_proposed += 1

        window_samples = (
            tuple(samples)
            if samples is not None
            else tuple(
                sample
                for sample in self._samples.get(track_id, ())
                if started_at <= sample.captured_at <= ended_at
            )
        )

        # Hand off and return. Everything the verifier will need is captured
        # here, while it is still true: the evidence ring keeps only a few
        # seconds, and a VLM answering twenty seconds from now would otherwise
        # find its window already evicted. Holding the frames is what bounds
        # the queue depth -- see `verify/pending.py`.
        self._pending.submit(
            functools.partial(
                self._verify_and_emit,
                candidate=candidate,
                detection=detection,
                action=action,
                track_id=track_id,
                window_frames=window_frames,
                window_samples=window_samples,
            )
        )

    async def _verify_and_emit(
        self,
        *,
        candidate: CandidateEvent,
        detection: Detection,
        action: CandidateAction,
        track_id: str,
        window_frames: Sequence[BufferedFrame],
        window_samples: Sequence[TrackSample],
    ) -> None:
        """Verify one candidate and record the outcome. Runs on a worker task,
        never on the frame loop -- see `verify/pending.py` for why that
        distinction is load-bearing rather than an optimisation."""
        result = await self._verifier.verify(
            candidate,
            frames=tuple(f.payload for f in window_frames),
            samples=tuple(window_samples),
            # Lazy: a window is tens of JPEGs, and the default rule-based
            # verifier never looks at a pixel. Decoding eagerly would charge
            # every candidate for a capability most verifiers do not use.
            decoded=_LazyDecodedFrames(window_frames),
        )
        self._events.append(
            PipelineEvent(
                at=result.occurred_at,
                track_id=track_id,
                label=detection.label,
                action=action,
                outcome=result.outcome,
                reason_code=result.reason_code,
                confidence=detection.confidence,
            )
        )

        if result.outcome == "confirmed":
            self.metrics.candidates_confirmed += 1
            await self._on_confirmed(candidate, result, window_frames)
        elif result.outcome == "rejected":
            self.metrics.candidates_rejected += 1
        else:
            self.metrics.candidates_unverified += 1

    async def drain(self) -> None:
        """Wait for every proposed candidate to be verified and emitted.

        The sync point for anything that needs the pipeline to have finished
        thinking: a replay reaching the end of a clip, a test asserting on what
        was confirmed, a service shutting down.
        """
        await self._pending.drain()
        identity_tasks = tuple(self._identity_tasks.values())
        if identity_tasks:
            await asyncio.gather(*identity_tasks, return_exceptions=True)
        last_seen_tasks = tuple(self._last_seen_tasks)
        if last_seen_tasks:
            await asyncio.gather(*last_seen_tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Finish outstanding verification and identity work, then stop workers."""
        await self.drain()
        await self._pending.aclose()


__all__ = [
    "OBSERVED_NOT_PROMOTED",
    "OnConfirmed",
    "OnObserved",
    "Pipeline",
    "PipelineEvent",
    "PipelineMetrics",
    "PipelineOutcome",
]
