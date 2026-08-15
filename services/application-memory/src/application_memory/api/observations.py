"""Ingestion.

Validate, resolve identity, store immutably, recompute. The endpoint does those
four things and maps exceptions; every rule lives in the domain or the
repository so it can be tested without an ASGI app.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status
from visual_memory_memory_contract.protocol import Observation

from application_memory.activity import MemoryEvent
from application_memory.deps import authorize_device, authorize_request, session_factory_of
from application_memory.domain.reducer import PromotionPolicy
from application_memory.errors import ConflictError
from application_memory.store import repository

router = APIRouter(tags=["observations"])


def policy_of(request: Request) -> PromotionPolicy:
    settings = request.app.state.settings
    return PromotionPolicy(
        min_event_confidence=settings.promote_min_event_confidence,
        min_identity_confidence=settings.promote_min_identity_confidence,
        require_evidence_for_placement=settings.require_evidence_for_placement,
    )


@router.post("/v1/observations", status_code=status.HTTP_201_CREATED)
def record(observation: Observation, request: Request, response: Response) -> dict[str, Any]:
    """Accept one observation.

    A `state` of null is a normal outcome, not an error: the observation was
    stored as history but did not clear the promotion thresholds, so trusted
    state is untouched.
    """
    authorize_request(request)
    authorize_device(request, observation.device_id)

    factory = session_factory_of(request)
    with factory() as db:
        try:
            result = repository.record_observation(db, observation, policy=policy_of(request))
        except repository.ConflictingObservation as exc:
            raise ConflictError(str(exc), observation_id=observation.observation_id) from exc
        db.commit()

    if result.duplicate:
        # Already applied. Report the original outcome rather than pretending a
        # second write happened.
        response.status_code = status.HTTP_200_OK
    else:
        # Duplicates are re-deliveries of something already logged the first
        # time -- recording them again would double-count in the live log.
        request.app.state.activity.record(
            MemoryEvent(
                at=repository.utcnow(),
                label=observation.object.label,
                action=observation.event.action,
                object_id=result.object_id,
                promoted=result.promoted,
                current_status=result.state.current_status if result.state else None,
            )
        )

    return {
        "observation_id": result.observation_id,
        "object_id": result.object_id,
        "promoted": result.promoted,
        "duplicate": result.duplicate,
        "state": result.state.model_dump(mode="json") if result.state else None,
    }
