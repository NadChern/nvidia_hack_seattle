"""Recent pipeline activity, for a human watching the pipeline work.

`Pipeline.recent_events` is a bounded, in-process ring -- not the canonical
record (that is `application_memory`'s `audit` table, reached only for
`confirmed` outcomes) and not durable across a restart. This exists for the
publisher dev page's live log, matching how `/v1/status` exists for its
status panel.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from vision_worker.deps import authorize_request
from vision_worker.pipeline import Pipeline

router = APIRouter(tags=["events"])


@router.get("/v1/events")
def events(request: Request) -> dict[str, Any]:
    authorize_request(request)
    pipeline: Pipeline = request.app.state.pipeline
    return {
        "events": [
            {
                "at": event.at,
                "track_id": event.track_id,
                "label": event.label,
                "action": event.action,
                "outcome": event.outcome,
                "reason_code": event.reason_code,
                "confidence": event.confidence,
            }
            for event in pipeline.recent_events
        ]
    }
