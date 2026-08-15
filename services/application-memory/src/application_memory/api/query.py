"""Answering "where is my ...?".

The endpoint resolves the object, checks that any cited evidence can actually
be loaded, and hands both to the pure answer layer. Deciding what may be
claimed happens in domain/answers.py; deciding what is retrievable requires
I/O, so it happens here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy.orm import Session as DbSession
from visual_memory_memory_contract.protocol import ObjectState, QueryRequest

from application_memory.deps import authorize_request, session_factory_of, settings_of
from application_memory.domain.answers import EvidenceRef, answer_for
from application_memory.evidence.store import EvidenceStore
from application_memory.store import models, repository

router = APIRouter(tags=["query"])


def _evidence_ref(
    db: DbSession, store: EvidenceStore, state: ObjectState | None, base_url: str | None
) -> EvidenceRef | None:
    """Resolve the evidence behind a claim into something a client can fetch.

    A reference is not evidence. docs/04 counts a confirmed answer whose
    evidence cannot be retrieved as an unsupported confident answer, so this
    checks the *file* rather than the row -- retention deletes files, and a row
    pointing at a deleted file looks exactly like a valid one.

    Returning None is the "not loadable" signal. It is deliberately impossible
    to hand back a URL without having confirmed the bytes exist, so an answer
    can never carry a link that 404s.
    """
    if state is None or state.last_confirmed_placement is None:
        return None
    evidence_id = state.last_confirmed_placement.evidence_id
    if evidence_id is None:
        return None
    row = db.get(models.EvidenceRow, evidence_id)
    if row is None or not store.exists(row.relative_path):
        return None

    path = f"/v1/evidence/{evidence_id}"
    return EvidenceRef(
        evidence_id=evidence_id,
        url=f"{base_url.rstrip('/')}{path}" if base_url else path,
        # Whatever was uploaded: image/jpeg for a frame, video/mp4 for a clip.
        # The client picks <img> or <video> from this rather than sniffing.
        media_type=row.media_type,
    )


@router.post("/v1/query")
def ask(query: QueryRequest, request: Request) -> dict[str, Any]:
    authorize_request(request)
    settings = settings_of(request)
    store = EvidenceStore(settings.evidence_dir)
    factory = session_factory_of(request)

    with factory() as db:
        label = query.label or ""
        candidates: tuple[str, ...] = ()
        object_id = query.object_id

        if object_id is None:
            matches = repository.find_objects_by_label(db, label, session_id=query.session_id)
            if len(matches) > 1:
                # Two things share a name. Picking one would be a coin flip
                # presented as a memory.
                candidates = tuple(matches)
            elif matches:
                object_id = matches[0]

        state = repository.state_of(db, object_id) if not candidates else None
        if state is not None and not label:
            row = db.get(models.ObjectStateRow, state.object_id)
            label = row.label if row else "object"

        answer = answer_for(
            state,
            label=label or "object",
            evidence=_evidence_ref(db, store, state, settings.public_base_url),
            candidates=candidates,
        )

    return answer.model_dump(mode="json")
