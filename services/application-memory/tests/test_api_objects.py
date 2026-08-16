"""Personal-object registry API, durable crops, and cross-session identity."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import struct
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from visual_memory_memory_contract.fixtures import SESSION, T0, keys_placed_and_left

from application_memory.config import Settings
from application_memory.main import create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


def an_upload(crop: bytes = b"reference-crop") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "view_index": 0,
        "quality": {
            "detection_confidence": 0.94,
            "box_area_fraction": 0.31,
            "sharpness_score": 1.12,
            "mask_box_ratio": 0.76,
            "quality_score": 0.91,
            "angular_velocity_rad_s": None,
        },
        "embedder_id": "fixture-v1",
        "pooling": "summary+mask-weighted-spatial-v1",
        "dim": 3,
        "summary": [0.1, -0.2, 0.3],
        "pooled_spatial": [-0.4, 0.5, -0.6],
        "crop_sha256": hashlib.sha256(crop).hexdigest(),
        "crop_media_type": "image/jpeg",
        "crop_base64": base64.b64encode(crop).decode("ascii"),
    }


async def test_registry_round_trips_float32_vectors_and_durable_crop(
    app: FastAPI, settings: Settings
) -> None:
    crop = b"reference-crop"
    upload = an_upload(crop)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post(
                "/v1/objects",
                json={"label": "keys", "idempotency_key": "register/keys/1"},
            )
            object_id = created.json()["object_id"]
            view_response = await client.post(f"/v1/objects/{object_id}/views", json=upload)
            view = view_response.json()
            gallery = (await client.get("/v1/objects")).json()
            unchanged = (
                await client.get(
                    "/v1/objects", params={"since_version": gallery["registry_version"]}
                )
            ).json()
            fetched = await client.get(view["crop_reference"])

    assert created.status_code == 201
    assert view_response.status_code == 201
    assert fetched.content == crop
    assert fetched.headers["content-type"] == "image/jpeg"
    stored = gallery["views"][0]
    # Compare the IEEE-754 bytes, not approximately equal Python doubles.
    assert struct.pack("<3f", *stored["summary"]) == struct.pack("<3f", *upload["summary"])
    assert struct.pack("<3f", *stored["pooled_spatial"]) == struct.pack(
        "<3f", *upload["pooled_spatial"]
    )
    assert unchanged["unchanged"] is True
    assert unchanged["objects"] == []
    assert (Path(settings.registration_crop_dir) / object_id).is_dir()


async def test_view_and_object_creation_are_idempotent_and_delete_removes_crops(
    app: FastAPI, settings: Settings
) -> None:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"label": "keys", "idempotency_key": "register/keys/1"}
            first = await client.post("/v1/objects", json=payload)
            second = await client.post("/v1/objects", json=payload)
            object_id = first.json()["object_id"]
            view_1 = await client.post(f"/v1/objects/{object_id}/views", json=an_upload())
            view_2 = await client.post(f"/v1/objects/{object_id}/views", json=an_upload())
            deleted = await client.delete(f"/v1/objects/{object_id}")
            after = await client.get("/v1/objects")

    assert second.status_code == 200
    assert second.json()["object_id"] == object_id
    assert view_1.status_code == 201
    assert view_2.status_code == 200
    assert view_2.json()["view_id"] == view_1.json()["view_id"]
    assert deleted.json()["deleted_views"] == 1
    assert deleted.json()["deleted_crop_files"] == 1
    assert after.json()["objects"] == []
    assert not (Path(settings.registration_crop_dir) / object_id).exists()


async def test_registered_object_survives_session_delete_and_answers_in_a_new_session(
    app: FastAPI,
) -> None:
    session_b = "sess_second"
    placed_a = keys_placed_and_left()[0]
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            enrolled = (
                await client.post(
                    "/v1/objects",
                    json={"label": "keys", "idempotency_key": "register/keys/cross-session"},
                )
            ).json()
            object_id = enrolled["object_id"]
            first = placed_a.model_copy(
                update={"object": placed_a.object.model_copy(update={"object_id": object_id})}
            )
            second = placed_a.model_copy(
                update={
                    "observation_id": "obs_session_b",
                    "idempotency_key": "glasses-01/sess_second/track-2/placed/1",
                    "session_id": session_b,
                    "media_epoch_id": "TR_second",
                    "object": placed_a.object.model_copy(
                        update={"object_id": object_id, "track_id": "track-2"}
                    ),
                    "event": placed_a.event.model_copy(
                        update={"occurred_at": T0 + dt.timedelta(minutes=10)}
                    ),
                }
            )
            for observation in (first, second):
                response = await client.post(
                    "/v1/observations", json=observation.model_dump(mode="json")
                )
                assert response.status_code == 201, response.text
            await client.delete(f"/v1/sessions/{SESSION}")
            answer = (
                await client.post("/v1/query", json={"label": "keys", "session_id": session_b})
            ).json()
            registry = (await client.get(f"/v1/objects/{object_id}")).json()

    assert answer["object_id"] == object_id
    assert answer["answer_status"] == "last_confirmed_only"
    assert answer["last_confirmed_placement"]["surface"] == "coffee_table"
    assert registry["object_id"] == object_id
