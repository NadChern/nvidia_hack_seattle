"""Evidence upload and retrieval.

Bytes in the body, digest in the query string, path assigned by the server.
docs/06 forbids clients submitting local file paths, and the digest is what
makes a stored frame evidence rather than merely an image.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response, status
from visual_memory_memory_contract.ids import new_evidence_id

from application_memory.deps import authorize_request, session_factory_of, settings_of
from application_memory.errors import InvalidRequestError, NotFoundError
from application_memory.evidence.store import EvidenceStore
from application_memory.store import models, repository

router = APIRouter(tags=["evidence"])


@router.post("/v1/evidence", status_code=status.HTTP_201_CREATED)
async def upload(
    request: Request,
    session_id: Annotated[str, Query()],
    sha256: Annotated[str, Query(min_length=64, max_length=64)],
) -> dict[str, Any]:
    authorize_request(request)
    settings = settings_of(request)
    limit = settings.max_evidence_bytes

    # Refuse before reading when the sender declares an oversized body, so the
    # bytes are never allocated. Chunked uploads declare nothing, hence the
    # second check below.
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise InvalidRequestError(
            "evidence is larger than the configured limit",
            limit_bytes=limit,
            declared_bytes=int(declared),
        )

    data = await request.body()
    if len(data) > limit:
        raise InvalidRequestError(
            "evidence is larger than the configured limit",
            limit_bytes=limit,
            received_bytes=len(data),
        )

    evidence_id = new_evidence_id()
    store = EvidenceStore(settings.evidence_dir)
    # Raises before anything is written if the bytes are not what was declared.
    relative = store.put(
        data, session_id=session_id, evidence_id=evidence_id, declared_sha256=sha256
    )

    factory = session_factory_of(request)
    with factory() as db:
        db.add(
            models.EvidenceRow(
                evidence_id=evidence_id,
                session_id=session_id,
                sha256=sha256.lower(),
                media_type=request.headers.get("content-type", "image/jpeg"),
                relative_path=str(relative),
                size_bytes=len(data),
                created_at=repository.utcnow(),
            )
        )
        db.commit()

    return {"evidence_id": evidence_id, "size_bytes": len(data), "sha256": sha256.lower()}


@router.get("/v1/evidence/{evidence_id}")
def download(evidence_id: str, request: Request) -> Response:
    authorize_request(request)
    settings = settings_of(request)
    factory = session_factory_of(request)

    with factory() as db:
        row = db.get(models.EvidenceRow, evidence_id)
        if row is None:
            raise NotFoundError("no such evidence", evidence_id=evidence_id)
        media_type = row.media_type
        relative = row.relative_path

    data = EvidenceStore(settings.evidence_dir).get(relative)
    if data is None:
        # The row outlived the file -- retention, or a manual deletion. Say so
        # rather than serving nothing with a 200.
        raise NotFoundError("evidence is no longer stored", evidence_id=evidence_id)
    return Response(content=data, media_type=media_type)
