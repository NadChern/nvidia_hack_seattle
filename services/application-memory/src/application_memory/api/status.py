"""Observable state for dashboards and release reports.

docs/04 requires the promotion threshold set used for an evaluation run to be
recorded, so the thresholds are reported here rather than left implicit in a
config file nobody reads back.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from application_memory import __version__
from application_memory.deps import authorize_request, session_factory_of, settings_of
from application_memory.store import models

router = APIRouter(tags=["status"])


@router.get("/v1/status")
def status(request: Request) -> dict[str, Any]:
    authorize_request(request)
    settings = settings_of(request)
    factory = session_factory_of(request)
    started_at: dt.datetime = request.app.state.started_at

    with factory() as db:

        def count(model: type[models.Base]) -> int:
            return int(db.scalar(select(func.count()).select_from(model)) or 0)

        by_status = {
            str(row[0]): int(row[1])
            for row in db.execute(
                select(models.ObjectStateRow.current_status, func.count()).group_by(
                    models.ObjectStateRow.current_status
                )
            ).all()
        }
        promoted = int(
            db.scalar(
                select(func.count())
                .select_from(models.Observation)
                .where(models.Observation.promoted.is_(True))
            )
            or 0
        )
        totals = {
            "sessions": count(models.Session),
            "observations": count(models.Observation),
            "observations_promoted": promoted,
            "lifecycle_signals": count(models.LifecycleSignal),
            "objects": count(models.ObjectStateRow),
            "evidence": count(models.EvidenceRow),
        }

    return {
        "service": settings.service_name,
        "version": __version__,
        "environment": settings.environment,
        "uptime_s": round((dt.datetime.now(dt.UTC) - started_at).total_seconds(), 1),
        "config": {
            # The threshold set an evaluation run must cite.
            "promote_min_event_confidence": settings.promote_min_event_confidence,
            "promote_min_identity_confidence": settings.promote_min_identity_confidence,
            "require_evidence_for_placement": settings.require_evidence_for_placement,
            "retention_hours": settings.retention_hours,
            "database": "sqlite" if settings.is_sqlite else "external",
        },
        "totals": totals,
        # The field that explains a silent pipeline: objects exist but nothing
        # is confirmed means promotion is rejecting everything.
        "objects_by_status": by_status,
    }
