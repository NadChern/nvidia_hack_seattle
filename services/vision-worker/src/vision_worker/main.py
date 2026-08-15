"""Application wiring: settings, logging, lifespan, and routers."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from visual_memory_media_contract.client import MediaClient
from visual_memory_memory_contract.client import MemoryClient
from visual_memory_vision_contract.protocol import DetectorRef

from vision_worker import __version__
from vision_worker.api import events, health, overlay, status
from vision_worker.config import Settings, get_settings
from vision_worker.consume.relay import RelayConsumer
from vision_worker.depth.base import DepthEstimator, MetricDepthReference
from vision_worker.depth.fixture import FixtureDepthEstimator
from vision_worker.depth.moge import MogeDepthEstimator
from vision_worker.depth.yolo import YoloDepthEstimator
from vision_worker.detect.base import Detector
from vision_worker.detect.fixture import FixtureDetector
from vision_worker.detect.yoloe import YoloeDetector
from vision_worker.domain.stability import StabilityConfig, TrackRegistry
from vision_worker.emit.memory import MemoryEmitter
from vision_worker.errors import VisionError
from vision_worker.evidence.ring import EvidenceRing
from vision_worker.identity.base import IdentityResolverProtocol, SegmentingDetector
from vision_worker.identity.fixture import FixtureEmbedder
from vision_worker.identity.gallery import GalleryCache
from vision_worker.identity.radio import RadioEmbedder
from vision_worker.identity.resolver import (
    IdentityResolver,
    IdentityResolverConfig,
    VlmIdentityEscalator,
)
from vision_worker.logging import configure_logging
from vision_worker.overlay.hub import OverlayHub
from vision_worker.pipeline import Pipeline
from vision_worker.pose.da3 import Da3WindowPose
from vision_worker.pose.image_motion import ImageMotionPose
from vision_worker.readiness import Readiness
from vision_worker.track.greedy_iou import GreedyIoUTracker
from vision_worker.verify.base import Verifier
from vision_worker.verify.rules import RuleBasedVerifier, RuleBasedVerifierConfig
from vision_worker.verify.vlm import VlmVerifier, VlmVerifierConfig
from vision_worker.verify.world_motion import WorldMotionConfig, WorldMotionVerifier

logger = logging.getLogger(__name__)

#: Bumped whenever the state machine's rules change in a way that would
#: affect a reproduced evaluation run -- carried in every CandidateEvent's
#: provenance, per docs/06.
STATE_MACHINE_VERSION = "vision-stability-v1"
PIPELINE_VERSION = "vision-pipeline-v1"


def build_detector(settings: Settings) -> tuple[Detector, DetectorRef]:
    """Construct the configured detector.

    `fixture` needs no GPU and no model -- an empty, looping script finds
    nothing, but proves every other piece of plumbing works, which is an
    honest state to ship rather than a placeholder lie. `yoloe` is the real
    detector; `YoloeDetector` is imported unconditionally at module level
    (see its own docstring) but only actually imports torch and ultralytics
    once `initialize()` runs, so selecting `fixture` never pays that cost.
    """
    if settings.detector_kind == "fixture":
        return FixtureDetector([[]], loop=True), DetectorRef(
            name="fixture", checkpoint="n/a", revision="v1"
        )
    if settings.detector_kind == "yoloe":
        detector = YoloeDetector(
            text_model=settings.yoloe_text_model,
            prompt_free_model=settings.yoloe_prompt_free_model,
            score_threshold=settings.yoloe_score_threshold,
            device=settings.yoloe_device,
        )
        return detector, DetectorRef(
            name="yoloe", checkpoint=settings.yoloe_text_model, revision="v1"
        )
    raise ValueError(f"unknown detector_kind {settings.detector_kind!r}")  # pragma: no cover


def build_depth_estimator(settings: Settings) -> tuple[DepthEstimator | None, DetectorRef | None]:
    """Construct the configured depth adapter, or none at all.

    `none` (the default) is not a placeholder -- it is the pipeline's
    original, honest shape: image-space stability, `depth_m=None` on every
    candidate, `world_point` always `None`. Depth is an enhancement layered
    on top once a checkpoint and a GPU are actually available, per
    `pipeline.py`'s own docstring.
    """
    if settings.depth_kind == "none":
        return None, None
    if settings.depth_kind == "fixture":
        return FixtureDepthEstimator(range_m=settings.fixture_depth_range_m), DetectorRef(
            name="fixture", checkpoint="n/a", revision="v1"
        )
    if settings.depth_kind == "moge":
        estimator = MogeDepthEstimator(
            model_id=settings.moge_model_id, emit_box3d=settings.moge_emit_box3d
        )
        return estimator, DetectorRef(name="moge", checkpoint=settings.moge_model_id, revision="v1")
    if settings.depth_kind == "yolo":
        estimator = YoloDepthEstimator(
            model=settings.yolo_depth_model, device=settings.yolo_depth_device
        )
        return estimator, DetectorRef(
            name="yolo-depth", checkpoint=settings.yolo_depth_model, revision="v1"
        )
    raise ValueError(f"unknown depth_kind {settings.depth_kind!r}")  # pragma: no cover


async def build_verifier(
    settings: Settings, *, metric_reference: MetricDepthReference | None = None
) -> tuple[Verifier, RuleBasedVerifierConfig]:
    """Construct the configured verifier, and the rule thresholds it reports.

    `world_motion` layers the DA3-backed world-trajectory check over the rule
    verifier rather than replacing it -- a reconstruction says whether an
    object moved, never whether a detection was confident enough to trust, so
    both questions still get asked. If the checkpoint will not load, the
    wrapper is dropped entirely and the rules stand alone: a pose adapter that
    is not there should cost accuracy, not availability.
    """
    rule_config = RuleBasedVerifierConfig(
        min_confidence=settings.rule_verifier_min_confidence,
        min_frame_count=settings.rule_verifier_min_frame_count,
    )
    rules = RuleBasedVerifier(rule_config)
    if settings.verifier_kind == "rules":
        return rules, rule_config

    if settings.verifier_kind == "vlm":
        # Replaces the rule check rather than wrapping it: a model that can
        # read the frames has strictly more to go on than a confidence
        # threshold, and running both would let a thin-but-correct detection
        # be rejected by the weaker of the two.
        return (
            VlmVerifier(
                VlmVerifierConfig(
                    base_url=settings.vlm_base_url,
                    model=settings.vlm_model,
                    max_frames=settings.vlm_max_frames,
                    num_ctx=settings.vlm_num_ctx,
                    timeout_s=settings.vlm_timeout_s,
                )
            ),
            rule_config,
        )

    # MoGe, when configured, doubles as the metric anchor that turns DA3's
    # arbitrary reconstruction units into metres -- so the world-motion
    # thresholds can be stated in centimetres rather than as fractions of an
    # unfixed scene scale. Without it the verifier still works, just in
    # ratios; see WorldMotionConfig.
    pose_source = Da3WindowPose(
        model_id=settings.da3_model_id,
        max_views=settings.da3_max_views,
        metric_reference=metric_reference,
    )
    await pose_source.initialize()
    if not pose_source.is_ready:
        logger.warning(
            "world-motion verifier requested but DA3 did not load; using rules alone",
            extra={"da3_model_id": settings.da3_model_id},
        )
        return rules, rule_config

    verifier = WorldMotionVerifier(
        rules,
        pose_source,
        config=WorldMotionConfig(
            still_m=settings.world_motion_still_m,
            settled_m=settings.world_motion_settled_m,
            still_ratio=settings.world_motion_still_ratio,
            settled_ratio=settings.world_motion_settled_ratio,
        ),
    )
    return verifier, rule_config


async def build_identity_resolver(
    settings: Settings,
    *,
    detector: Detector,
    memory_client: MemoryClient,
) -> IdentityResolverProtocol | None:
    if settings.identity_kind == "none":
        return None
    if not hasattr(detector, "segment"):
        logger.warning("identity requested but detector cannot segment; identity disabled")
        return None
    embedder = (
        FixtureEmbedder()
        if settings.identity_kind == "fixture"
        else RadioEmbedder(device=settings.identity_device)
    )
    escalator = (
        VlmIdentityEscalator(
            memory_client,
            base_url=settings.vlm_base_url,
            model=settings.vlm_model,
            timeout_s=settings.vlm_timeout_s,
        )
        if settings.identity_vlm_escalation
        else None
    )
    resolver = IdentityResolver(
        segmenter=cast(SegmentingDetector, detector),
        embedder=embedder,
        gallery=GalleryCache(memory_client, ttl_s=settings.identity_gallery_ttl_s),
        config=IdentityResolverConfig(
            min_cosine=settings.identity_min_cosine,
            min_margin=settings.identity_min_margin,
            escalation_low=settings.identity_escalation_low,
            summary_weight=settings.identity_summary_weight,
        ),
        escalator=escalator,
    )
    await resolver.initialize()
    return resolver


def build_stability_config(settings: Settings) -> StabilityConfig:
    """Convert the configured durations into frame counts at `source_fps`.

    The conversion itself is pure and lives in `domain/stability.py`. What
    happens here is the part that needs a logger: reporting both halves, and
    warning when a duration was too short to survive the rounding.

    A `dwell_frames` of 1 means a single stable frame confirms a placement --
    which is not a dwell at all, and any source rate at or below 1/`dwell_
    seconds` produces exactly that. It is a legitimate configuration to run
    with (the passive path still guards the genuinely ambiguous case), but it
    is never one to arrive at by accident, so it says so.
    """
    config = StabilityConfig.from_durations(
        source_fps=settings.source_fps,
        dwell_seconds=settings.dwell_seconds,
        passive_confirmation_seconds=settings.passive_confirmation_seconds,
        reacquire_within_seconds=settings.reacquire_within_seconds,
        carried_emit_interval_seconds=settings.carried_emit_interval_seconds,
        world_motion_threshold_m=settings.world_motion_threshold_m,
        image_residual_threshold=settings.image_residual_threshold,
    )
    logger.info(
        "stability thresholds resolved",
        extra={
            "source_fps": settings.source_fps,
            "dwell_frames": config.dwell_frames,
            "passive_confirmation_frames": config.passive_confirmation_frames,
            "reacquire_within_frames": config.reacquire_within_frames,
            "carried_emit_interval_frames": config.carried_emit_interval_frames,
        },
    )
    if config.dwell_frames < 2:
        logger.warning(
            "dwell_seconds rounds to a single frame at this source rate -- one "
            "stable frame will confirm a placement; raise VMA_DWELL_SECONDS or "
            "the gateway's VMA_SAMPLE_FPS",
            extra={"source_fps": settings.source_fps, "dwell_seconds": settings.dwell_seconds},
        )
    return config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Wire every stage together and drive the relay in a background task."""
    settings: Settings = app.state.settings
    readiness: Readiness = app.state.readiness

    detector, detector_ref = build_detector(settings)
    await detector.initialize()
    tracker_ref = DetectorRef(name="greedy-iou", checkpoint="n/a", revision="v1")

    depth_estimator, depth_model_ref = build_depth_estimator(settings)
    if depth_estimator is not None:
        # Unlike the primary detector, a depth-load failure must not crash
        # startup -- `MogeDepthEstimator.initialize()` already catches its
        # own exceptions and degrades to `is_ready=False`; `fixture`'s never
        # raises. Nothing here needs a try/except of its own.
        await depth_estimator.initialize()

    memory_client = MemoryClient(
        settings.memory_base_url,
        token=settings.memory_token,
        timeout=settings.memory_request_timeout_s,
    )
    emitter = MemoryEmitter(
        memory_client,
        clip_fps=settings.resolved_clip_fps,
        min_identity_cosine=settings.identity_min_cosine,
        memory_min_identity_confidence=settings.memory_min_identity_confidence,
    )
    identity_resolver = await build_identity_resolver(
        settings,
        detector=detector,
        memory_client=memory_client,
    )

    # Only a genuinely metric adapter may anchor DA3's scale -- the fixture
    # one scripts a constant and would fit a confidently wrong factor.
    metric_reference = depth_estimator if settings.depth_kind == "moge" else None
    verifier, verifier_config = await build_verifier(
        settings,
        metric_reference=metric_reference,  # pyright: ignore[reportArgumentType]
    )
    stability_config = build_stability_config(settings)
    # Off by default in `deploy`: the overlay stream exists for a console
    # watching the pipeline, and a production run with nobody watching should
    # not carry a surface it never uses.
    overlay_hub = (
        OverlayHub(max_subscribers=settings.overlay_max_viewers)
        if settings.overlay_enabled
        else None
    )
    pipeline = Pipeline(
        detector=detector,
        detector_ref=detector_ref,
        # Kept at least as large as the stability machine's own occlusion
        # tolerance: if the tracker forgot an id first, identity would already
        # be lost a layer below the machine that is willing to wait for it.
        tracker=GreedyIoUTracker(max_age_frames=stability_config.reacquire_within_frames),
        tracker_ref=tracker_ref,
        pose_source=ImageMotionPose(),
        track_registry=TrackRegistry(stability_config),
        source_fps=settings.source_fps,
        evidence_ring=EvidenceRing(dt.timedelta(seconds=settings.evidence_ring_seconds)),
        verifier=verifier,
        detection_labels=settings.detection_labels,
        state_machine_version=STATE_MACHINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        on_confirmed=emitter.emit,
        depth_estimator=depth_estimator,
        depth_model_ref=depth_model_ref,
        verification_queue_depth=settings.verification_queue_depth,
        verification_concurrency=settings.verification_concurrency,
        # `None` when disabled, so the pipeline assembles no overlay at all
        # rather than building one for a hub nobody can reach.
        overlay_sink=overlay_hub.publish if overlay_hub is not None else None,
        overlay_depth_interval_s=settings.overlay_depth_interval_s,
        max_detections_per_frame=settings.max_detections_per_frame,
        identity_resolver=identity_resolver,
        identity_track_frames=settings.identity_track_frames,
        identity_min_detection_confidence=settings.identity_min_detection_confidence,
        identity_min_scale=settings.identity_min_scale,
        on_observed=emitter.emit_last_seen,
    )

    app.state.overlay_hub = overlay_hub
    app.state.detector = detector
    app.state.depth_estimator = depth_estimator
    app.state.identity_resolver = identity_resolver

    media_client = MediaClient(settings.gateway_video_url, token=settings.memory_token)
    consumer = RelayConsumer(media_client, pipeline)
    app.state.relay_consumer = consumer
    relay_task = asyncio.create_task(consumer.run(), name="relay-consumer")

    def relay_check() -> str | None:
        # MediaClient reconnects on its own with backoff and never gives up
        # on a transient failure, so a still-running task is "ready" even
        # mid-reconnect. Only an unexpected exit -- a bug, not connectivity
        # -- should fail readiness.
        if relay_task.done() and not relay_task.cancelled():
            exc = relay_task.exception()
            return f"relay consumer task exited: {exc}" if exc else "relay consumer task exited"
        return None

    readiness.register("relay", relay_check)

    app.state.pipeline = pipeline
    app.state.verifier = verifier
    app.state.verifier_config = verifier_config
    app.state.started_at = dt.datetime.now(dt.UTC)

    logger.info(
        "vision worker starting",
        extra={
            "environment": settings.environment,
            "detector_kind": settings.detector_kind,
            "depth_kind": settings.depth_kind,
            "gateway_video_url": settings.gateway_video_url,
            "memory_base_url": settings.memory_base_url,
        },
    )
    try:
        yield
    finally:
        readiness.begin_shutdown()
        if overlay_hub is not None:
            # Before the relay stops: a viewer's send loop awaits the hub, and
            # closing it is what lets those tasks finish rather than hang until
            # the server's own timeout.
            overlay_hub.close()
        await media_client.aclose()
        relay_task.cancel()
        try:
            await asyncio.wait_for(relay_task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        # Stop reading frames first, then let outstanding verification finish
        # before the Memory client it needs is closed underneath it. A
        # candidate already proposed has evidence held in memory and nowhere
        # else; dropping it here would lose an event the service had committed
        # to answering. Bounded, because the queue is.
        try:
            await asyncio.wait_for(pipeline.aclose(), timeout=settings.shutdown_drain_timeout_s)
        except TimeoutError:
            logger.warning(
                "verification still in flight at shutdown; abandoning it",
                extra={"pending": pipeline.pending_verifications},
            )
        if identity_resolver is not None:
            await identity_resolver.aclose()
        await asyncio.to_thread(memory_client.close)
        await detector.aclose()
        if depth_estimator is not None:
            await depth_estimator.aclose()
        logger.info("vision worker stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Takes settings explicitly so tests can construct an app without mutating
    the process environment or clearing the settings cache.
    """
    resolved = settings or get_settings()
    configure_logging(
        level=resolved.log_level,
        service=resolved.service_name,
        version=__version__,
    )

    # No interactive docs or schema in deploy, matching every other service
    # here -- the port is published to the trusted LAN.
    hide_schema = resolved.environment == "deploy"
    app = FastAPI(
        title=resolved.service_name,
        version=__version__,
        lifespan=lifespan,
        openapi_url=None if hide_schema else "/openapi.json",
        docs_url=None if hide_schema else "/docs",
        redoc_url=None if hide_schema else "/redoc",
    )
    app.state.settings = resolved
    app.state.readiness = Readiness()

    if resolved.environment != "deploy":
        # The publisher dev page's debug panel (task #55) runs in the
        # browser at the gateway's origin (a different port) and fetches
        # this service's /v1/status and /v1/events directly -- a real
        # cross-origin read, which the browser blocks without this. Never
        # enabled in deploy: nothing there needs to fetch this service from
        # a browser, and an open CORS policy on a LAN-published port is
        # exactly the kind of default docs/07 asks services to not carry.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET"],
            allow_headers=["authorization"],
        )

    async def handle_vision_error(_: Request, exc: VisionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.add_exception_handler(VisionError, handle_vision_error)  # pyright: ignore[reportArgumentType]

    app.include_router(health.router)
    app.include_router(status.router)
    app.include_router(events.router)
    app.include_router(overlay.router)
    return app


app = create_app()
