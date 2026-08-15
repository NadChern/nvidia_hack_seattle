"""Recent ingestion activity, for a human watching the pipeline work.

`ActivityLog.recent` is a bounded, in-process ring -- not the canonical
record (the `audit` table is) and not durable across a restart. Backs the
publisher dev page's live log, matching `vision_worker`'s identical
`/v1/events`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from application_memory.deps import authorize_request

router = APIRouter(tags=["events"])


@router.get("/v1/events")
def events(request: Request) -> dict[str, Any]:
    authorize_request(request)
    return {
        "events": [
            {
                "at": event.at,
                "label": event.label,
                "action": event.action,
                "object_id": event.object_id,
                "promoted": event.promoted,
                "current_status": event.current_status,
            }
            for event in request.app.state.activity.recent
        ]
    }
