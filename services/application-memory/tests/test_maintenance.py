"""Full memory reset clears every store and bumps the gallery version."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

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


async def test_reset_clears_every_store_and_bumps_the_registry_version(
    app: FastAPI, settings: Settings
) -> None:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            enrolled = (
                await client.post(
                    "/v1/objects",
                    json={"label": "keys", "idempotency_key": "register/keys/reset"},
                )
            ).json()
            object_id = enrolled["object_id"]
            await client.post(f"/v1/objects/{object_id}/views", json=an_upload())

            before = (await client.get("/v1/objects")).json()
            crop_dir = Path(settings.registration_crop_dir) / object_id

            reset = await client.post("/v1/maintenance/reset")
            after = (await client.get("/v1/objects")).json()

    assert before["objects"], "precondition: the gallery held the registered object"
    assert crop_dir.parent.exists()

    assert reset.status_code == 200
    body = reset.json()
    assert body["reset"] is True
    # A newer version -- not a reset to zero -- so a vision cache that already
    # saw the old gallery refreshes instead of ignoring a stale-looking number.
    assert body["registry_version"] > before["registry_version"]

    assert after["objects"] == []
    assert after["registry_version"] == body["registry_version"]
    assert not crop_dir.exists()


async def test_reset_is_idempotent_on_an_already_empty_store(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/v1/maintenance/reset")
            second = await client.post("/v1/maintenance/reset")

    assert first.status_code == 200
    assert second.status_code == 200
    # Still monotonic even with nothing to delete.
    assert second.json()["registry_version"] > first.json()["registry_version"]
