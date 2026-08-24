"""Stable personal-object registry and durable reference crops."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from visual_memory_memory_contract.ids import new_view_id
from visual_memory_memory_contract.protocol import ObjectViewUpload

from application_memory.deps import authorize_request, session_factory_of, settings_of
from application_memory.errors import ConflictError, InvalidRequestError, NotFoundError
from application_memory.evidence.registration import RegistrationCropStore
from application_memory.store import models, repository

router = APIRouter(prefix="/v1/objects", tags=["objects"])


class CreateObjectRequest(BaseModel):
    label: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=512)


class RenameObjectRequest(BaseModel):
    label: str = Field(min_length=1, max_length=128)


def _crop_store(request: Request) -> RegistrationCropStore:
    store: RegistrationCropStore = request.app.state.registration_crops
    return store


@router.post("", status_code=status.HTTP_201_CREATED)
def create_object(
    body: CreateObjectRequest, request: Request, response: Response
) -> dict[str, Any]:
    authorize_request(request)
    label = body.label.strip()
    if not label:
        raise InvalidRequestError("object label cannot be blank")

    factory = session_factory_of(request)
    with factory() as db:
        try:
            enrolled, created = repository.create_enrolled_object(
                db, label=label, idempotency_key=body.idempotency_key
            )
        except repository.ConflictingObservation as exc:
            raise ConflictError(str(exc), idempotency_key=body.idempotency_key) from exc
        db.commit()
    if not created:
        response.status_code = status.HTTP_200_OK
    return enrolled.model_dump(mode="json")


@router.post("/{object_id}/views", status_code=status.HTTP_201_CREATED)
def put_view(
    object_id: str,
    upload: ObjectViewUpload,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    authorize_request(request)
    settings = settings_of(request)
    if len(upload.crop_base64) > settings.max_registration_crop_bytes * 2:
        raise InvalidRequestError("encoded registration crop is larger than the configured limit")
    try:
        crop = base64.b64decode(upload.crop_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError("registration crop is not valid base64") from exc
    if not crop:
        raise InvalidRequestError("registration crop cannot be empty")
    if len(crop) > settings.max_registration_crop_bytes:
        raise InvalidRequestError(
            "registration crop is larger than the configured limit",
            limit=settings.max_registration_crop_bytes,
            received=len(crop),
        )

    factory = session_factory_of(request)
    store = _crop_store(request)
    relative_path: str | None = None
    with factory() as db:
        owner = repository.enrolled_object(db, object_id)
        if owner is None:
            raise NotFoundError("registered object does not exist", object_id=object_id)
        duplicate = repository.find_object_view_by_content(
            db,
            object_id=object_id,
            view_index=upload.view_index,
            crop_sha256=upload.crop_sha256,
        )
        if duplicate is not None:
            response.status_code = status.HTTP_200_OK
            return duplicate.model_dump(mode="json")

        current_views = int(
            db.scalar(
                select(func.count())
                .select_from(models.ObjectViewRow)
                .where(models.ObjectViewRow.object_id == object_id)
            )
            or 0
        )
        if current_views >= settings.registry_max_views_per_object:
            raise ConflictError(
                "registered object has reached its reference-view limit",
                object_id=object_id,
                limit=settings.registry_max_views_per_object,
            )
        if upload.dim > settings.registry_max_embedding_dim:
            raise InvalidRequestError(
                "embedding dimension exceeds the configured limit",
                dim=upload.dim,
                limit=settings.registry_max_embedding_dim,
            )

        view_id = new_view_id()
        relative = store.put(
            crop,
            object_id=object_id,
            view_id=view_id,
            declared_sha256=upload.crop_sha256,
        )
        relative_path = str(relative)
        try:
            view = repository.put_object_view(
                db,
                object_id=object_id,
                view_id=view_id,
                upload=upload,
                crop_relative_path=relative_path,
                max_views=settings.registry_max_views_per_object,
                max_dim=settings.registry_max_embedding_dim,
            )
            db.commit()
        except (LookupError, OverflowError, ValueError) as exc:
            db.rollback()
            store.delete(relative_path)
            raise InvalidRequestError(str(exc), object_id=object_id) from exc
    return view.model_dump(mode="json")


@router.get("")
def gallery(request: Request, since_version: int | None = None) -> dict[str, Any]:
    authorize_request(request)
    if since_version is not None and since_version < 0:
        raise InvalidRequestError("since_version cannot be negative")
    factory = session_factory_of(request)
    with factory() as db:
        result = repository.list_gallery(db, since_version=since_version)
    return result.model_dump(mode="json")


@router.patch("/{object_id}")
def rename_object(
    object_id: str, body: RenameObjectRequest, request: Request
) -> dict[str, Any]:
    """Rename a registered object -- the operator path for the placeholder
    ``item {uuid}`` labels the register button mints when it has no name.

    Unlike the vision worker's create endpoint, no detection-label constraint
    applies here: a display label is a human name ("Alex's car keys"), not a
    grounding prompt.
    """
    authorize_request(request)
    label = body.label.strip()
    if not label:
        raise InvalidRequestError("object label cannot be blank")
    factory = session_factory_of(request)
    with factory() as db:
        enrolled = repository.rename_enrolled_object(db, object_id, label=label)
        if enrolled is None:
            raise NotFoundError("registered object does not exist", object_id=object_id)
        db.commit()
    return enrolled.model_dump(mode="json")


@router.get("/{object_id}")
def get_object(object_id: str, request: Request) -> dict[str, Any]:
    authorize_request(request)
    factory = session_factory_of(request)
    with factory() as db:
        enrolled = repository.enrolled_object(db, object_id)
    if enrolled is None:
        raise NotFoundError("registered object does not exist", object_id=object_id)
    return enrolled.model_dump(mode="json")


@router.get("/{object_id}/views/{view_id}/crop")
def get_crop(object_id: str, view_id: str, request: Request) -> FastAPIResponse:
    authorize_request(request)
    factory = session_factory_of(request)
    with factory() as db:
        row = repository.object_view_row(db, object_id, view_id)
    if row is None:
        raise NotFoundError("registered object view does not exist", view_id=view_id)
    crop = _crop_store(request).get(row.crop_relative_path)
    if crop is None:
        raise NotFoundError("registered object crop is not retrievable", view_id=view_id)
    return FastAPIResponse(content=crop, media_type=row.crop_media_type)


@router.delete("/{object_id}")
def delete_object(object_id: str, request: Request) -> dict[str, Any]:
    authorize_request(request)
    factory = session_factory_of(request)
    with factory() as db:
        deleted = repository.delete_enrolled_object(db, object_id)
        if deleted is None:
            raise NotFoundError("registered object does not exist", object_id=object_id)
        db.commit()
    files = _crop_store(request).delete_object(object_id)
    return {
        "object_id": object_id,
        "registry_version": deleted.registry_version,
        "deleted_views": len(deleted.crop_relative_paths),
        "deleted_crop_files": files,
    }
