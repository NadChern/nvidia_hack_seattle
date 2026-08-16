"""Vision registration control API and label/session gates."""

from __future__ import annotations

import base64
import datetime as dt
import json

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from visual_memory_memory_contract.client import MemoryClient

from vision_worker.config import Settings
from vision_worker.identity.enroll import EnrollmentProgress
from vision_worker.main import create_app

pytestmark = pytest.mark.anyio


class ActivePipeline:
    current_session_id: str | None = "sess_1"


class RecordingManager:
    def __init__(self) -> None:
        self.progress: EnrollmentProgress | None = None
        self.manual_crops: tuple[bytes, ...] = ()

    def arm(
        self, *, object_id: str, label: str, capture_seconds: float | None = None
    ) -> EnrollmentProgress:
        now = dt.datetime.now(dt.UTC)
        self.progress = EnrollmentProgress(
            object_id=object_id,
            label=label,
            state="capturing",
            started_at=now,
            capture_ends_at=now + dt.timedelta(seconds=capture_seconds or 6.0),
        )
        return self.progress

    async def submit_manual(
        self, *, object_id: str, label: str, crops: list[bytes]
    ) -> EnrollmentProgress:
        now = dt.datetime.now(dt.UTC)
        self.manual_crops = tuple(crops)
        self.progress = EnrollmentProgress(
            object_id=object_id,
            label=label,
            state="succeeded",
            started_at=now,
            capture_ends_at=now,
            frames_total=len(crops),
            quality_passed=len(crops),
            selected_views=len(crops),
        )
        return self.progress

    def status(self, object_id: str) -> EnrollmentProgress | None:
        if self.progress is not None and self.progress.object_id == object_id:
            return self.progress
        return None


def memory_handler(request: httpx.Request) -> httpx.Response:
    now = "2026-08-15T12:00:00.000Z"
    if request.method == "POST" and request.url.path == "/v1/objects":
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "schema_version": "1.0",
                "object_id": "object_keys",
                "label": body["label"],
                "idempotency_key": body["idempotency_key"],
                "created_at": now,
                "updated_at": now,
                "registry_version": 1,
            },
        )
    if request.method == "GET" and request.url.path == "/v1/objects/object_keys":
        return httpx.Response(
            200,
            json={
                "schema_version": "1.0",
                "object_id": "object_keys",
                "label": "keys",
                "idempotency_key": "register/keys",
                "created_at": now,
                "updated_at": now,
                "registry_version": 1,
            },
        )
    if request.method == "GET" and request.url.path == "/v1/objects":
        return httpx.Response(
            200,
            json={"schema_version": "1.0", "registry_version": 1, "objects": [], "views": []},
        )
    return httpx.Response(404)


def app_for() -> tuple[FastAPI, RecordingManager, ActivePipeline]:
    app = create_app(
        Settings(
            environment="ci",
            detection_labels=("keys",),
            identity_kind="fixture",
        )
    )
    manager = RecordingManager()
    pipeline = ActivePipeline()
    app.state.enrollment_manager = manager
    app.state.pipeline = pipeline
    app.state.memory_client = MemoryClient(transport=httpx.MockTransport(memory_handler))
    return app, manager, pipeline


async def test_create_capture_and_poll_registration() -> None:
    app, manager, _ = app_for()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post(
            "/v1/objects", json={"label": "keys", "idempotency_key": "register/keys"}
        )
        capture = await client.post(
            "/v1/objects/object_keys/capture", json={"capture_seconds": 4.0}
        )
        progress = await client.get("/v1/objects/object_keys/status")

    assert created.status_code == 201
    assert capture.status_code == 202
    assert capture.json()["state"] == "capturing"
    assert progress.json()["object_id"] == "object_keys"
    assert manager.progress is not None


async def test_manual_enrollment_decodes_operator_confirmed_crops() -> None:
    app, manager, _ = app_for()
    encoded = base64.b64encode(b"jpeg-bytes").decode()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/objects/object_keys/manual",
            json={"views_base64": [encoded, encoded]},
        )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"
    assert manager.manual_crops == (b"jpeg-bytes", b"jpeg-bytes")


async def test_manual_enrollment_rejects_invalid_base64() -> None:
    app, _, _ = app_for()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/objects/object_keys/manual",
            json={"views_base64": ["not base64!", "also invalid!@"]},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_untrackable_label_is_rejected_before_promising_registration() -> None:
    app, _, _ = app_for()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/objects", json={"label": "wallet", "idempotency_key": "register/wallet"}
        )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    assert response.json()["context"]["detection_labels"] == ["keys"]


async def test_capture_requires_an_active_video_session() -> None:
    app, _, pipeline = app_for()
    pipeline.current_session_id = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/objects/object_keys/capture", json={})

    assert response.status_code == 503
    assert response.json()["code"] == "unavailable"
