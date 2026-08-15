"""Application wiring: settings, logging, lifespan, and routers."""

from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import logging
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from media_gateway import __version__
from media_gateway.api import device_events, health, pairing, return_audio, sessions, status, stream
from media_gateway.config import Settings, get_settings
from media_gateway.domain.device_events import DeviceEventHub
from media_gateway.domain.epoch import EpochRegistry
from media_gateway.domain.manual_trigger import ManualTriggerRegistry
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.domain.pairing import DeviceCredentialSigner, PairingRegistry
from media_gateway.domain.ratelimit import FixedWindowLimiter
from media_gateway.domain.session import SessionRegistry
from media_gateway.errors import GatewayError
from media_gateway.logging import configure_logging
from media_gateway.pipeline import MediaPipeline
from media_gateway.readiness import Readiness
from media_gateway.relay.hub import RelayHub
from media_gateway.transport.memory_sink import MemorySink
from media_gateway.transport.scripted import ScriptedMediaSource, ScriptedPlan
from media_gateway.transport.source import MediaSource
from media_gateway.transport.supervisor import SessionSupervisor

logger = logging.getLogger(__name__)


def build_media_source(settings: Settings) -> MediaSource | None:
    """Return the configured ingress, or None when one is wired externally."""
    if settings.media_source == "scripted":
        return ScriptedMediaSource(
            ScriptedPlan(
                width=settings.expected_video_width,
                height=settings.expected_video_height,
                # Paced and looping, so a gateway left running by hand keeps
                # producing rather than falling silent after one pass.
                frame_interval_s=settings.scripted_frame_interval_s,
                loop=True,
            )
        )
    # The LiveKit source is driven by the session supervisor, not the lifespan.
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Validate configuration, start ingress, and drain on shutdown."""
    settings: Settings = app.state.settings
    readiness: Readiness = app.state.readiness

    if settings.media_source == "livekit":
        # Fail the process rather than starting a gateway that cannot connect.
        settings.require_livekit_credentials()

    hub = RelayHub(
        max_subscribers=settings.ws_max_subscribers,
        audio_queue_chunks=settings.audio_queue_chunks,
    )
    metrics = MetricsRegistry()
    epochs = EpochRegistry(settings)
    # Off unless VMA_LIFECYCLE_SINK_URL is set: the gateway is fully useful
    # with no Memory Service, and an unconfigured sink must not produce noise.
    memory_sink = MemorySink(settings)
    await memory_sink.start()
    app.state.memory_sink = memory_sink

    pipeline = MediaPipeline(
        settings=settings,
        hub=hub,
        epochs=epochs,
        metrics=metrics,
        lifecycle_sink=memory_sink,
    )
    app.state.hub = hub
    app.state.metrics = metrics
    app.state.epochs = epochs
    app.state.started_at = dt.datetime.now(dt.UTC)
    app.state.pipeline = pipeline
    app.state.sessions = SessionRegistry(settings)
    app.state.session_limiter = FixedWindowLimiter(limit=settings.sessions_rate_limit_per_minute)
    internal_secret = (
        settings.internal_api_token.get_secret_value().encode()
        if settings.internal_api_token is not None
        else secrets.token_bytes(32)
    )
    credential_secret = hmac.digest(
        internal_secret, b"visual-memory/device-credential/v1", "sha256"
    )
    app.state.device_credentials = DeviceCredentialSigner(
        secret=credential_secret,
        ttl_s=settings.device_credential_ttl_s,
    )
    app.state.pairing = PairingRegistry(
        ttl_s=settings.pairing_code_ttl_s,
        max_pending=settings.pairing_max_pending,
        signer=app.state.device_credentials,
    )
    app.state.pairing_claim_limiter = FixedWindowLimiter(
        limit=settings.pairing_claims_rate_limit_per_minute
    )
    app.state.device_events = DeviceEventHub(
        queue_size=settings.device_event_queue_size,
        max_subscribers=settings.device_event_max_subscribers,
    )
    app.state.manual_triggers = ManualTriggerRegistry(ttl_s=settings.manual_trigger_ttl_s)

    supervisor = SessionSupervisor(settings=settings, sink=pipeline)
    await supervisor.start()
    app.state.supervisor = supervisor
    # Readiness reflects whether LiveKit is reachable, never whether a device
    # happens to be connected.
    readiness.register("livekit", supervisor.readiness)

    source = build_media_source(settings)
    source_task: asyncio.Task[None] | None = None
    if source is not None:
        source_task = asyncio.create_task(source.run(pipeline), name="media-source")

    logger.info(
        "media gateway starting",
        extra={
            "environment": settings.environment,
            "media_source": settings.media_source,
            "livekit_url": settings.livekit_url,
            "sample_fps": settings.sample_fps,
            "video_encoding": settings.video_encoding,
        },
    )
    try:
        yield
    finally:
        # Order matters: stop producing, let the pipeline emit its closing
        # messages, then release subscribers so they receive that tail.
        readiness.begin_shutdown()
        await supervisor.stop()
        if source is not None:
            await source.aclose()
        if source_task is not None:
            try:
                await asyncio.wait_for(source_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                source_task.cancel()
        await pipeline.stop()
        # After pipeline.stop(), because shutdown is when the last signals are
        # produced -- closing the sink first would discard exactly those.
        await memory_sink.stop()
        hub.close_all("gateway_shutdown")
        app.state.device_events.close_all("gateway_shutdown")
        logger.info("media gateway stopped")


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

    # No interactive docs or schema in deploy. The port is published to the
    # trusted LAN, and every other route there refuses an unauthenticated
    # caller; serving the full API surface -- the relay, return audio, session
    # minting, and their field shapes -- to anyone who can reach it undoes that
    # for no operational benefit.
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

    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.add_exception_handler(GatewayError, handle_gateway_error)  # pyright: ignore[reportArgumentType]

    app.include_router(health.router)
    app.include_router(pairing.router)
    app.include_router(sessions.router)
    app.include_router(device_events.router)
    app.include_router(stream.router)
    app.include_router(return_audio.router)
    app.include_router(status.router)
    return app


app = create_app()
