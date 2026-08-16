"""The registry client constructs and parses the shared wire contract."""

from __future__ import annotations

import datetime as dt
import json

import httpx

from visual_memory_memory_contract.client import MemoryClient
from visual_memory_memory_contract.protocol import ObjectViewQuality, ObjectViewUpload

NOW = dt.datetime(2026, 8, 15, 12, 0, tzinfo=dt.UTC)


def test_registry_client_round_trip_and_since_version() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/objects":
            body = json.loads(request.content)
            return httpx.Response(
                201,
                json={
                    "schema_version": "1.0",
                    "object_id": "object_1",
                    "label": body["label"],
                    "idempotency_key": body["idempotency_key"],
                    "created_at": NOW.isoformat(),
                    "updated_at": NOW.isoformat(),
                    "registry_version": 1,
                },
            )
        if request.method == "GET" and request.url.path == "/v1/objects":
            return httpx.Response(
                200,
                json={
                    "schema_version": "1.0",
                    "registry_version": 1,
                    "unchanged": True,
                    "objects": [],
                    "views": [],
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"object_id": "object_1"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    with MemoryClient(transport=httpx.MockTransport(handler)) as client:
        enrolled = client.create_object(label="keys", idempotency_key="register/keys/1")
        gallery = client.list_gallery(since_version=1)
        client.delete_object(enrolled.object_id)

    assert enrolled.object_id == "object_1"
    assert gallery.unchanged is True
    assert requests[1].url.params["since_version"] == "1"


def test_put_object_view_sends_the_typed_upload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            201,
            json=body
            | {
                "view_id": "view_1",
                "object_id": "object_1",
                "crop_reference": "/v1/objects/object_1/views/view_1/crop",
                "created_at": NOW.isoformat(),
                "registry_version": 2,
            },
        )

    upload = ObjectViewUpload(
        view_index=0,
        quality=ObjectViewQuality(
            detection_confidence=0.9,
            box_area_fraction=0.3,
            sharpness_score=1.0,
            mask_box_ratio=0.8,
            quality_score=0.9,
        ),
        embedder_id="fixture-v1",
        pooling="summary+spatial-v1",
        dim=2,
        summary=(0.25, -0.5),
        pooled_spatial=(0.125, -0.25),
        crop_sha256="a" * 64,
        crop_base64="aW1hZ2U=",
    )

    with MemoryClient(transport=httpx.MockTransport(handler)) as client:
        view = client.put_object_view("object_1", upload)

    assert view.summary == (0.25, -0.5)
    assert view.crop_reference.endswith("/crop")
