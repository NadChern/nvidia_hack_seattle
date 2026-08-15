"""Liveness and readiness endpoints. Matches `media_gateway.api.health`."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    reason: str | None = None


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    """Report that the process and its event loop are running.

    Deliberately touches no dependency: a relay or Memory outage must not
    cause the orchestrator to restart an otherwise healthy process.
    """
    return HealthResponse(status="ok", service=request.app.state.settings.service_name)


@router.get("/health/ready", response_model=HealthResponse)
def readiness(request: Request, response: Response) -> HealthResponse:
    """Report whether this process can serve traffic right now."""
    service = request.app.state.settings.service_name
    reason = request.app.state.readiness.evaluate()
    if reason is not None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="not_ready", service=service, reason=reason)
    return HealthResponse(status="ready", service=service)
