"""Lifecycle signals from the Media Gateway.

A separate endpoint rather than an observation with null fields. An observation
carrying no object would fail the promotion rules, and widening those rules to
admit it would weaken them for real observations -- which is the reasoning
recorded in docs/06 and signed off by the Memory owner.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from visual_memory_memory_contract.protocol import LifecycleEnvelope

from application_memory.api.observations import policy_of
from application_memory.deps import authorize_request, session_factory_of
from application_memory.store import repository

router = APIRouter(tags=["lifecycle"])


@router.post("/v1/lifecycle")
def signal(envelope: LifecycleEnvelope, request: Request) -> dict[str, Any]:
    """Apply a signal to every object it scopes.

    The gateway names an epoch because it has never seen an object. Memory
    turns that into a transition per object whose identity began in that epoch.
    """
    authorize_request(request)

    factory = session_factory_of(request)
    with factory() as db:
        states = repository.record_lifecycle(db, envelope, policy=policy_of(request))
        db.commit()

    return {
        "signal_id": envelope.signal_id,
        "affected": len(states),
        "states": [state.model_dump(mode="json") for state in states],
    }
