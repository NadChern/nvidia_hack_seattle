"""Liveness and readiness.

Readiness reports whether the database can be reached, never whether any
observation has arrived. An idle memory service with no data is healthy; making
readiness depend on traffic would deadlock a `depends_on: service_healthy`.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from application_memory.deps import session_factory_of

router = APIRouter(tags=["health"])


@router.get("/health/live")
def liveness(request: Request) -> dict[str, str]:
    return {"status": "ok", "service": request.app.state.settings.service_name}


@router.get("/health/ready")
def readiness(request: Request, response: Response) -> dict[str, str]:
    factory = session_factory_of(request)
    try:
        with factory() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "database unreachable"}
    return {"status": "ready", "service": request.app.state.settings.service_name}
