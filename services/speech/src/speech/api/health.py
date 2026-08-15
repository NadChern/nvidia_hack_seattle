"""Liveness and readiness.

Mirrors `application_memory/api/health.py`'s shape (`app.state.settings`, a
plain dict body, no `response_model`). Readiness has nothing external to
check yet -- no database, no gateway connection required at startup -- so
today it reports exactly what liveness does; that will change once this
service depends on something that can be genuinely unreachable.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health/live")
def liveness(request: Request) -> dict[str, str]:
    return {"status": "ok", "service": request.app.state.settings.service_name}


@router.get("/health/ready")
def readiness(request: Request) -> dict[str, str]:
    return {"status": "ready", "service": request.app.state.settings.service_name}
