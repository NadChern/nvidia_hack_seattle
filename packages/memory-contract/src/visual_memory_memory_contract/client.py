"""A small client for posting observations and asking questions.

Vision uses this to write; the Agent uses it to read. It exists so neither has
to hand-assemble JSON against `docs/06` and drift from it -- the models are the
contract, and this is the thinnest possible wrapper over them.

Requires the `client` extra:

    uv add "visual-memory-memory-contract[client]"
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from types import TracebackType
from typing import Self

from visual_memory_memory_contract.protocol import (
    EnrolledObject,
    LifecycleEnvelope,
    ObjectGallery,
    ObjectState,
    ObjectView,
    ObjectViewUpload,
    Observation,
    QueryRequest,
    QueryResponse,
)

try:
    import httpx
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the extra
    raise ModuleNotFoundError(
        "MemoryClient needs httpx; install this package with the 'client' extra"
    ) from exc


class MemoryError_(Exception):
    """Memory refused a request.

    Named with a trailing underscore because `MemoryError` is a builtin, and
    shadowing it inside a library that other services import would be rude.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MemoryClient:
    """Synchronous client. Memory is a request/response service, not a stream."""

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        *,
        token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"authorization": f"Bearer {token}"} if token else {}
        # `transport` is the standard httpx testing seam
        # (https://www.python-httpx.org/advanced/transports/#mock-transports):
        # a producer's test suite can hand this an `httpx.MockTransport` and
        # exercise real request construction and response parsing with no
        # network and no Memory Service process, rather than mocking this
        # class's own methods and never noticing a real wire-format drift.
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: object) -> dict[str, object]:
        try:
            response = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"{path} refused the request ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        body: dict[str, object] = response.json()
        return body

    def _get(
        self, path: str, *, params: Mapping[str, str | int] | None = None
    ) -> dict[str, object]:
        """GET JSON with the same bounded error translation as `_post`."""
        try:
            response = self._http.get(path, params=params)
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"{path} refused the request ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        body: dict[str, object] = response.json()
        return body

    def record(self, observation: Observation) -> ObjectState | None:
        """Submit one observation, returning the resulting trusted state.

        `None` means the observation was accepted and stored but did not
        promote -- low confidence, or unresolved identity. That is a normal
        outcome, not an error: history is kept, trusted state is not touched.
        """
        body = self._post("/v1/observations", observation.model_dump(mode="json"))
        state = body.get("state")
        return ObjectState.model_validate(state) if state else None

    def signal(self, envelope: LifecycleEnvelope) -> None:
        """Report that a track or session went away."""
        self._post("/v1/lifecycle", envelope.model_dump(mode="json"))

    def ask(
        self,
        *,
        object_id: str | None = None,
        label: str | None = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        """Ask where something is.

        The answer carries an `answer_status` that must be preserved by any
        layer that rewords it. Shortening "I last confirmed them there, but
        they were picked up afterward" into its first half turns a truthful
        answer into a false one.
        """
        request = QueryRequest(object_id=object_id, label=label, session_id=session_id)
        return QueryResponse.model_validate(
            self._post("/v1/query", request.model_dump(mode="json"))
        )

    def put_evidence(
        self, data: bytes, *, session_id: str, sha256: str, media_type: str = "image/jpeg"
    ) -> str:
        """Upload one piece of evidence, returning its id.

        The server verifies the digest before storing and refuses a mismatch,
        so a corrupted upload fails here rather than becoming an unloadable
        reference behind a confident answer.

        `media_type` defaults to `image/jpeg` for the common single-frame
        case, but the server already stores whatever content-type it is
        given (`application_memory/api/evidence.py`), so a producer that
        confirms a candidate over several seconds can upload a short
        `video/mp4` clip instead -- see `AnsweredPlacement.evidence_media_type`,
        which exists precisely so a client can choose `<video>` over `<img>`.
        """
        try:
            response = self._http.post(
                "/v1/evidence",
                params={"session_id": session_id, "sha256": sha256},
                content=data,
                headers={"content-type": media_type},
            )
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"evidence upload refused ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        evidence_id: str = response.json()["evidence_id"]
        return evidence_id

    def get_evidence(self, evidence_id: str) -> bytes:
        response = self._http.get(f"/v1/evidence/{evidence_id}")
        if response.status_code >= 400:
            raise MemoryError_(
                f"evidence {evidence_id} is not retrievable ({response.status_code})",
                status_code=response.status_code,
            )
        return response.content

    def delete_session(self, session_id: str) -> None:
        """Remove a session's observations, derived state, and evidence."""
        try:
            response = self._http.delete(f"/v1/sessions/{session_id}")
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"session deletion refused ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    def create_object(self, *, label: str, idempotency_key: str) -> EnrolledObject:
        """Mint or idempotently return one stable personal-object identity."""
        return EnrolledObject.model_validate(
            self._post(
                "/v1/objects",
                {"label": label, "idempotency_key": idempotency_key},
            )
        )

    def put_object_view(self, object_id: str, upload: ObjectViewUpload) -> ObjectView:
        """Persist one bounded reference crop and its pooled embeddings."""
        return ObjectView.model_validate(
            self._post(
                f"/v1/objects/{object_id}/views",
                upload.model_dump(mode="json"),
            )
        )

    def list_gallery(self, *, since_version: int | None = None) -> ObjectGallery:
        params = {"since_version": since_version} if since_version is not None else None
        return ObjectGallery.model_validate(self._get("/v1/objects", params=params))

    def get_object(self, object_id: str) -> EnrolledObject:
        return EnrolledObject.model_validate(self._get(f"/v1/objects/{object_id}"))

    def get_object_crop(self, object_id: str, view_id: str) -> bytes:
        path = f"/v1/objects/{object_id}/views/{view_id}/crop"
        try:
            response = self._http.get(path)
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"{path} refused the request ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )
        return response.content

    def delete_object(self, object_id: str) -> None:
        try:
            response = self._http.delete(f"/v1/objects/{object_id}")
        except httpx.HTTPError as exc:
            raise MemoryError_(f"could not reach the memory service: {exc}") from exc
        if response.status_code >= 400:
            raise MemoryError_(
                f"object deletion refused ({response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    def register_session(self, *, session_id: str, device_id: str) -> str:
        """Register a session and adopt the identifier Memory returns.

        The Media Gateway mints `session_id` because it is the only component
        present when a session starts, but Memory owns persistence. This is the
        handover: the gateway calls it when `VMA_SESSION_REGISTRY_URL` is set,
        and adopts whatever comes back.
        """
        body = self._post(
            "/v1/sessions",
            {"session_id": session_id, "device_id": device_id},
        )
        registered: str = str(body["session_id"])
        return registered


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


__all__ = ["MemoryClient", "MemoryError_", "utcnow"]
