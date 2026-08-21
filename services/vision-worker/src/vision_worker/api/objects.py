"""Registration control API; extraction stays in `identity/enroll.py`."""

from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field
from visual_memory_memory_contract.client import MemoryClient, MemoryError_

from vision_worker.deps import authorize_request
from vision_worker.errors import (
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    UnavailableError,
)
from vision_worker.identity.enroll import EnrollmentManager, EnrollmentMode
from vision_worker.pipeline import Pipeline

router = APIRouter(prefix="/v1/objects", tags=["objects"])


class CreateObjectRequest(BaseModel):
    label: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=512)


class CaptureRequest(BaseModel):
    capture_seconds: float | None = Field(default=None, gt=0)
    #: ``grounded`` runs the VLM localizer (voice registration). ``center-anchor``
    #: propagates a fixed centre box with the tracker -- the register button's
    #: grounder-free, speech-free path.
    mode: EnrollmentMode = "grounded"


class ManualEnrollmentRequest(BaseModel):
    views_base64: list[str] = Field(min_length=2, max_length=8)


def _memory(request: Request) -> MemoryClient:
    client: MemoryClient = request.app.state.memory_client
    return client


def _manager(request: Request) -> EnrollmentManager:
    manager: EnrollmentManager | None = getattr(request.app.state, "enrollment_manager", None)
    if manager is None:
        raise UnavailableError("registration requires an enabled identity embedder")
    return manager


def _validate_label(request: Request, label: str) -> str:
    resolved = label.strip()
    if not resolved:
        raise InvalidRequestError("object label cannot be blank")
    configured = request.app.state.settings.detection_labels
    if not configured:
        return resolved

    # A spoken label ("key", "these keys") rarely reproduces a detection prompt
    # verbatim, and STT drift makes it worse, so match loosely and register
    # under the CANONICAL detection label: normalized exact first, then
    # substring containment either way. A genuine non-tracked word ("case")
    # still fails loudly with the trackable set so the agent can tell the
    # wearer what it can actually track.
    def _norm(value: str) -> str:
        return " ".join(value.casefold().split())

    wanted = _norm(resolved)
    canonical = {_norm(candidate): candidate for candidate in configured}
    if wanted in canonical:
        return canonical[wanted]
    for norm_label, original in canonical.items():
        if wanted in norm_label or norm_label in wanted:
            return original
    raise InvalidRequestError(
        "object label is not configured for detection",
        label=resolved,
        detection_labels=list(configured),
    )


def _translate_memory_error(exc: MemoryError_) -> Exception:
    if exc.status_code == 404:
        return NotFoundError(str(exc))
    if exc.status_code == 409:
        return ConflictError(str(exc))
    return UnavailableError("memory object registry is unavailable")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_object(body: CreateObjectRequest, request: Request) -> dict[str, Any]:
    authorize_request(request)
    _manager(request)
    label = _validate_label(request, body.label)
    try:
        enrolled = await asyncio.to_thread(
            _memory(request).create_object,
            label=label,
            idempotency_key=body.idempotency_key,
        )
    except MemoryError_ as exc:
        raise _translate_memory_error(exc) from exc
    return enrolled.model_dump(mode="json")


@router.post("/{object_id}/capture", status_code=status.HTTP_202_ACCEPTED)
async def capture(
    object_id: str,
    body: CaptureRequest,
    request: Request,
) -> dict[str, object]:
    authorize_request(request)
    manager = _manager(request)
    pipeline: Pipeline = request.app.state.pipeline
    if pipeline.current_session_id is None:
        raise UnavailableError("registration requires an active video session")
    try:
        enrolled = await asyncio.to_thread(_memory(request).get_object, object_id)
    except MemoryError_ as exc:
        raise _translate_memory_error(exc) from exc
    label = _validate_label(request, enrolled.label)
    try:
        progress = manager.arm(
            object_id=object_id,
            label=label,
            capture_seconds=body.capture_seconds,
            mode=body.mode,
        )
    except RuntimeError as exc:
        raise ConflictError(str(exc), object_id=object_id) from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc), object_id=object_id) from exc
    return progress.payload()


@router.post("/{object_id}/manual")
async def manual_enrollment(
    object_id: str,
    body: ManualEnrollmentRequest,
    request: Request,
) -> dict[str, object]:
    """Persist operator-drawn crops only after explicit Console confirmation."""
    authorize_request(request)
    manager = _manager(request)
    try:
        enrolled = await asyncio.to_thread(_memory(request).get_object, object_id)
    except MemoryError_ as exc:
        raise _translate_memory_error(exc) from exc
    label = _validate_label(request, enrolled.label)
    crops: list[bytes] = []
    for encoded in body.views_base64:
        if len(encoded) > 3 * 1024 * 1024:
            raise InvalidRequestError("manual reference image is too large")
        try:
            crops.append(base64.b64decode(encoded, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequestError("manual reference image is not valid base64") from exc
    try:
        progress = await manager.submit_manual(
            object_id=object_id,
            label=label,
            crops=crops,
        )
    except RuntimeError as exc:
        raise ConflictError(str(exc), object_id=object_id) from exc
    return progress.payload()


@router.get("")
async def list_objects(request: Request) -> dict[str, Any]:
    authorize_request(request)
    try:
        gallery = await asyncio.to_thread(_memory(request).list_gallery)
    except MemoryError_ as exc:
        raise _translate_memory_error(exc) from exc
    return gallery.model_dump(mode="json")


@router.delete("/{object_id}")
async def delete_object(object_id: str, request: Request) -> dict[str, str]:
    """Delete registry evidence and immediately invalidate Vision's gallery."""
    authorize_request(request)
    try:
        await asyncio.to_thread(_memory(request).delete_object, object_id)
    except MemoryError_ as exc:
        raise _translate_memory_error(exc) from exc
    await request.app.state.gallery.refresh(force=True)
    return {"object_id": object_id}


@router.get("/{object_id}/status")
def registration_status(object_id: str, request: Request) -> dict[str, object]:
    authorize_request(request)
    progress = _manager(request).status(object_id)
    if progress is None:
        raise NotFoundError("registration attempt does not exist", object_id=object_id)
    return progress.payload()
