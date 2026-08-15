"""Application assembly and lifespan.

Startup opens the database, ensures the schema, and starts the retention
sweeper. Everything the endpoints need hangs off `app.state`, so a test can
build an app with different settings and get a different database without
touching a global.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from application_memory import __version__
from application_memory.activity import ActivityLog
from application_memory.api import (
    events,
    evidence,
    health,
    lifecycle,
    objects,
    observations,
    query,
    sessions,
    status,
)
from application_memory.config import Settings, get_settings
from application_memory.domain.reducer import PromotionPolicy
from application_memory.errors import MemoryServiceError
from application_memory.evidence.registration import RegistrationCropStore
from application_memory.evidence.store import EvidenceStore
from application_memory.logging import configure_logging
from application_memory.store import repository
from application_memory.store.engine import create_all, create_db_engine, create_session_factory

logger = logging.getLogger(__name__)


async def _retention_sweeper(app: FastAPI) -> None:
    """Delete sessions past the retention window.

    docs/07 makes evidence and event metadata session-scoped with a 24 hour
    default. Retention that only ran at startup would keep data for as long as
    the process happened to stay up, which is not a retention policy.
    """
    settings: Settings = app.state.settings
    while True:
        await asyncio.sleep(settings.retention_sweep_interval_s)
        try:
            cutoff = repository.utcnow() - dt.timedelta(hours=settings.retention_hours)
            stale: list[str] = []
            with app.state.sessions() as db:
                stale = repository.sessions_older_than(db, cutoff)
                policy = PromotionPolicy(
                    min_event_confidence=settings.promote_min_event_confidence,
                    min_identity_confidence=settings.promote_min_identity_confidence,
                    require_evidence_for_placement=settings.require_evidence_for_placement,
                )
                for session_id in stale:
                    repository.delete_session(db, session_id, policy=policy)
                db.commit()
            for session_id in stale:
                app.state.evidence.delete_session(session_id)
            if stale:
                logger.info("retention removed sessions", extra={"sessions": len(stale)})
        except asyncio.CancelledError:
            raise
        except Exception:
            # A sweep failure must not take the service down: answering
            # queries matters more than deleting on schedule, and the next
            # tick retries.
            logger.exception("retention sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, service=settings.service_name, version=__version__)

    engine = create_db_engine(settings)
    # create_all rather than running a migration: Alembic owns schema
    # evolution, but a service that will not start because nobody ran a
    # migration is a poor first experience. The two agree -- `alembic check`
    # reports no pending operations.
    create_all(engine)

    app.state.engine = engine
    app.state.sessions = create_session_factory(engine)
    app.state.evidence = EvidenceStore(settings.evidence_dir)
    app.state.registration_crops = RegistrationCropStore(settings.registration_crop_dir)
    app.state.started_at = dt.datetime.now(dt.UTC)

    logger.info(
        "memory service starting",
        extra={
            "environment": settings.environment,
            "database": "sqlite" if settings.is_sqlite else "external",
            "retention_hours": settings.retention_hours,
        },
    )

    sweeper = asyncio.create_task(_retention_sweeper(app), name="retention-sweeper")
    try:
        yield
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper
        engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    # No interactive docs or schema in deploy: every other route there refuses
    # an unauthenticated caller, and serving the full API surface to anyone who
    # can reach the port undoes that for no operational benefit.
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
    app.state.activity = ActivityLog()

    if resolved.environment != "deploy":
        # The publisher dev page's debug panel (vision-worker task #55)
        # runs in the browser at the gateway's origin and fetches this
        # service's /v1/status and /v1/events directly -- a real
        # cross-origin read the browser blocks without this. See
        # vision_worker.main's identical block for the deploy-mode reasoning.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET"],
            allow_headers=["authorization"],
        )

    async def handle(_: Request, exc: MemoryServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    app.add_exception_handler(MemoryServiceError, handle)  # type: ignore[arg-type]

    for module in (
        health,
        sessions,
        observations,
        lifecycle,
        query,
        evidence,
        objects,
        status,
        events,
    ):
        app.include_router(module.router)

    return app


app = create_app()
