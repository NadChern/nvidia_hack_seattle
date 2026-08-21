"""Application wiring: settings, logging, lifespan, and routers."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from visual_memory_media_contract.client import MediaClient
from visual_memory_memory_contract.client import MemoryClient

from vision_worker import __version__
from vision_worker.api import events, health, objects, status
from vision_worker.config import Settings, get_settings
from vision_worker.consume.relay import RelayConsumer
from vision_worker.emit.memory import MemoryEmitter
from vision_worker.errors import VisionError
from vision_worker.evidence.ring import EvidenceRing
from vision_worker.identity.base import ObjectEmbedder
from vision_worker.identity.enroll import EnrollmentConfig, EnrollmentManager, ObjectEnroller
from vision_worker.identity.fixture import FixtureEmbedder
from vision_worker.identity.gallery import GalleryCache
from vision_worker.identity.radio import RadioEmbedder
from vision_worker.identity.selection import QualityConfig
from vision_worker.identity.track import Sam2Tracker, Tracker
from vision_worker.logging import configure_logging
from vision_worker.pipeline import Pipeline
from vision_worker.readiness import Readiness
from vision_worker.reason.base import ReasonerLocalizer
from vision_worker.reason.cosmos import CosmosReasoner, CosmosReasonerConfig
from vision_worker.reason.fixture import FixtureReasoner

logger = logging.getLogger(__name__)

#: Carried in every CandidateEvent's provenance, per docs/06 -- bumped whenever
#: the perception pipeline changes in a way that would affect a reproduced run.
PIPELINE_VERSION = "vision-reasoner-v1"


def build_reasoner(settings: Settings) -> ReasonerLocalizer:
    """Construct the configured window reasoner.

    `fixture` needs no GPU and no Cosmos -- it returns a scripted empty result,
    the honest ci/no-GPU shape that exercises the whole window loop while
    finding nothing. `cosmos` is the real reasoner, talking to a vLLM endpoint;
    it holds no weights in-process, which is what keeps this service deployable
    on a machine that cannot host the model.
    """
    if settings.reason_kind == "fixture":
        return FixtureReasoner()
    return CosmosReasoner(
        CosmosReasonerConfig(
            base_url=settings.reason_base_url,
            model=settings.reason_model,
            max_frames=settings.reason_max_frames,
            event_confidence=settings.reason_event_confidence,
            max_tokens=settings.reason_max_tokens,
            timeout_s=settings.reason_timeout_s,
        )
    )


def build_embedder(settings: Settings) -> ObjectEmbedder:
    """The identity embedder behind the C-RADIOv4 gate.

    `fixture` is deterministic CPU CI; `radio` loads the pinned C-RADIOv4
    adapter from the models extra. Anything other than `radio` uses the fixture
    so the pipeline still builds and the gallery still resolves in a no-GPU run.
    """
    if settings.identity_kind == "radio":
        return RadioEmbedder(device=settings.identity_device)
    return FixtureEmbedder()


def build_tracker(settings: Settings) -> Tracker | None:
    """The register button's grounder-free localiser.

    Only the real identity path gets SAM2; the fixture/CI path returns ``None``,
    so a ``center-anchor`` capture there fails cleanly with ``tracker_unavailable``
    rather than trying to load a model a no-GPU run cannot serve. The tracker is
    lazy: constructing it here is free until the first register-button capture.
    """
    if settings.identity_kind == "radio":
        return Sam2Tracker(device=settings.identity_device)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Wire every stage together and drive the relay in a background task."""
    settings: Settings = app.state.settings
    readiness: Readiness = app.state.readiness

    reasoner = build_reasoner(settings)
    embedder = build_embedder(settings)
    tracker = build_tracker(settings)
    await embedder.initialize()

    memory_client = MemoryClient(
        settings.memory_base_url,
        token=settings.memory_token,
        timeout=settings.memory_request_timeout_s,
    )
    gallery = GalleryCache(memory_client, ttl_s=settings.identity_gallery_ttl_s)
    emitter = MemoryEmitter(
        memory_client,
        clip_fps=settings.resolved_clip_fps,
        min_identity_cosine=settings.identity_min_cosine,
        memory_min_identity_confidence=settings.memory_min_identity_confidence,
    )
    evidence_ring = EvidenceRing(dt.timedelta(seconds=settings.evidence_ring_seconds))

    pipeline = Pipeline(
        reasoner=reasoner,
        embedder=embedder,
        gallery=gallery,
        evidence_ring=evidence_ring,
        on_confirmed=emitter.emit,
        pipeline_version=PIPELINE_VERSION,
        reason_window_s=settings.reason_window_seconds,
        reason_interval_s=settings.reason_interval_seconds,
        reason_max_frames=settings.reason_max_frames,
        identity_min_cosine=settings.identity_min_cosine,
        identity_summary_weight=settings.identity_summary_weight,
        box_padding=settings.identity_box_padding,
        event_cooldown_s=settings.event_cooldown_seconds,
        promote_motion_events=settings.promote_motion_events,
        work_queue_depth=settings.verification_queue_depth,
    )

    enrollment_config = EnrollmentConfig(
        capture_seconds=settings.registration_capture_seconds,
        max_capture_seconds=settings.registration_max_capture_seconds,
        temporal_max_frames=settings.registration_temporal_max_frames,
        temporal_batch_frames=settings.registration_temporal_batch_frames,
        candidate_interval_seconds=settings.registration_candidate_interval_seconds,
        max_frames=settings.registration_max_frames,
        target_views=settings.registration_target_views,
        min_views=settings.registration_min_views,
        dedup_threshold=settings.registration_dedup_threshold,
        summary_weight=settings.identity_summary_weight,
        quality=QualityConfig(
            min_detection_confidence=settings.identity_min_detection_confidence,
            min_scale=settings.identity_min_scale,
            min_mask_box_ratio=settings.registration_min_mask_box_ratio,
            max_mask_box_ratio=settings.registration_max_mask_box_ratio,
            relative_sharpness_floor=settings.registration_relative_sharpness_floor,
            max_angular_velocity_rad_s=settings.registration_max_angular_velocity_rad_s,
        ),
    )
    enrollment_manager = EnrollmentManager(
        evidence_ring=evidence_ring,
        enroller=ObjectEnroller(
            localizer=reasoner,
            embedder=embedder,
            gallery=gallery,
            memory_client=memory_client,
            config=enrollment_config,
            box_padding=settings.identity_box_padding,
            tracker=tracker,
            centre_frac=settings.registration_centre_frac,
        ),
        config=enrollment_config,
    )

    app.state.pipeline = pipeline
    app.state.reasoner = reasoner
    app.state.embedder = embedder
    app.state.gallery = gallery
    app.state.enrollment_manager = enrollment_manager
    app.state.memory_client = memory_client
    app.state.started_at = dt.datetime.now(dt.UTC)

    media_client = MediaClient(settings.gateway_video_url, token=settings.memory_token)
    consumer = RelayConsumer(media_client, pipeline)
    app.state.relay_consumer = consumer
    relay_task = asyncio.create_task(consumer.run(), name="relay-consumer")

    def relay_check() -> str | None:
        # MediaClient reconnects on its own with backoff, so a still-running
        # task is "ready" even mid-reconnect. Only an unexpected exit fails it.
        if relay_task.done() and not relay_task.cancelled():
            exc = relay_task.exception()
            return f"relay consumer task exited: {exc}" if exc else "relay consumer task exited"
        return None

    readiness.register("relay", relay_check)

    logger.info(
        "vision worker starting",
        extra={
            "environment": settings.environment,
            "reason_kind": settings.reason_kind,
            "identity_kind": settings.identity_kind,
            "gateway_video_url": settings.gateway_video_url,
            "memory_base_url": settings.memory_base_url,
        },
    )
    try:
        yield
    finally:
        readiness.begin_shutdown()
        await media_client.aclose()
        relay_task.cancel()
        try:
            await asyncio.wait_for(relay_task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        # Stop reading frames first, then let outstanding window analysis finish
        # before the Memory client it needs is closed underneath it.
        await enrollment_manager.aclose()
        try:
            await asyncio.wait_for(pipeline.aclose(), timeout=settings.shutdown_drain_timeout_s)
        except TimeoutError:
            logger.warning(
                "window analysis still in flight at shutdown; abandoning it",
                extra={"pending": pipeline.pending_analyses},
            )
        await embedder.aclose()
        await asyncio.to_thread(memory_client.close)
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
    app.include_router(objects.router)
    return app


app = create_app()
