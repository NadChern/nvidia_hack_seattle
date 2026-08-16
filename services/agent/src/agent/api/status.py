"""Report the selected backend without exposing credentials."""

from __future__ import annotations

from fastapi import APIRouter, Request

from agent.deps import authorize_request, metrics_of, settings_of
from agent.models import AgentMetricsResponse, AgentStatusResponse, StatusBackend

router = APIRouter(tags=["status"])


@router.get("/v1/status", response_model=AgentStatusResponse)
def status(request: Request) -> AgentStatusResponse:
    authorize_request(request)
    settings = settings_of(request)
    backend: StatusBackend
    if settings.agent_backend == "stub":
        backend = "stub"
    else:
        backend = settings.endpoint_scope
    return AgentStatusResponse(
        backend=backend,
        model=settings.llm_model,
        endpoint_host=settings.endpoint_host,
        vision_endpoint_host=settings.vision_endpoint_host,
        registration_capture_seconds=settings.registration_capture_seconds,
        registration_timeout_s=settings.registration_timeout_s,
        metrics=AgentMetricsResponse.model_validate(metrics_of(request).snapshot()),
    )
