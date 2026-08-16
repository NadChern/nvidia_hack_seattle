"""Session registration and deletion.

The Media Gateway mints `session_id` because it is the only component present
when a session starts. Memory owns persistence and deletion. This endpoint is
the handover: the gateway calls it when VMA_SESSION_REGISTRY_URL is set and
adopts whatever comes back.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from application_memory.api.observations import policy_of
from application_memory.deps import (
    authorize_device,
    authorize_request,
    session_factory_of,
    settings_of,
)
from application_memory.evidence.store import EvidenceStore
from application_memory.store import repository

router = APIRouter(tags=["sessions"])


class RegisterSession(BaseModel):
    session_id: str
    device_id: str


@router.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
def register(body: RegisterSession, request: Request) -> dict[str, Any]:
    """Adopt a gateway-minted session id.

    Memory returns the identifier it will use. Today that is the one it was
    given; keeping the round trip means Memory can start minting its own later
    without the gateway changing.
    """
    authorize_request(request)
    authorize_device(request, body.device_id)

    factory = session_factory_of(request)
    with factory() as db:
        row = repository.ensure_session(db, session_id=body.session_id, device_id=body.device_id)
        session_id = row.session_id
        db.commit()

    return {"session_id": session_id, "device_id": body.device_id}


@router.delete("/v1/sessions/{session_id}", status_code=status.HTTP_200_OK)
def forget(session_id: str, request: Request) -> dict[str, Any]:
    """Delete a session's observations, derived state, and evidence.

    docs/07 requires deletion to remove database records and evidence files
    together. State is derived, so removing the observations removes the claim;
    there is no second place a deleted memory survives.
    """
    authorize_request(request)
    settings = settings_of(request)
    factory = session_factory_of(request)

    with factory() as db:
        counts = repository.delete_session(db, session_id, policy=policy_of(request))
        db.commit()

    counts["evidence_files"] = EvidenceStore(settings.evidence_dir).delete_session(session_id)
    return {"session_id": session_id, "deleted": counts}
