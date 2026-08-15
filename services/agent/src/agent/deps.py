"""Request-scoped helpers shared by routers."""

from __future__ import annotations

import hmac

from fastapi import Request

from agent.config import Settings
from agent.errors import UnauthorizedError
from agent.metrics import AgentMetrics
from agent.stub import QueryBackend


def settings_of(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def backend_of(request: Request) -> QueryBackend:
    backend: QueryBackend = request.app.state.backend
    return backend


def metrics_of(request: Request) -> AgentMetrics:
    metrics: AgentMetrics = request.app.state.metrics
    return metrics


def authorize_request(request: Request) -> None:
    """Require the configured bearer token using constant-time comparison."""
    settings = settings_of(request)
    if settings.internal_api_token is None:
        return

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise UnauthorizedError("a bearer token is required")
    expected = settings.internal_api_token.get_secret_value()
    if not hmac.compare_digest(presented, expected):
        raise UnauthorizedError("the bearer token is not valid")


__all__ = ["authorize_request", "backend_of", "metrics_of", "settings_of"]
