"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from agent.deps import settings_of
from agent.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", service=settings_of(request).service_name)


@router.get("/health/ready", response_model=HealthResponse)
def readiness(request: Request, response: Response) -> HealthResponse:
    # Backend construction happens once in create_app. Downstream outages are
    # reported per request, but a crashed hands-free supervisor cannot recover
    # and should make orchestration restart this process.
    settings = settings_of(request)
    backend_ready = getattr(request.app.state, "backend", None) is not None
    listener_task = getattr(request.app.state, "listener_task", None)
    listener_ready = not settings.hands_free_enabled or (
        listener_task is not None and not listener_task.done()
    )
    ready = backend_ready and listener_ready
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if ready else "not_ready",
        service=settings.service_name,
    )
