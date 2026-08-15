"""The HTTP surface, end to end against a real database and filesystem.

Exercises the whole path a consumer sees: post observations, upload evidence,
ask a question, delete a session. The interesting assertions are about what the
service refuses to claim.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from visual_memory_memory_contract.fixtures import (
    DEVICE,
    EPOCH,
    SESSION,
    T0,
    keys_placed_and_left,
    keys_placed_then_picked_up,
)
from visual_memory_memory_contract.protocol import Observation

from application_memory.config import Settings
from application_memory.main import create_app

pytestmark = pytest.mark.anyio

TOKEN = "an-internal-token-of-at-least-32-chars"


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


async def post_all(client: AsyncClient, observations: object) -> list[dict[str, object]]:
    bodies: list[dict[str, object]] = []
    for observation in observations:  # type: ignore[union-attr]
        assert isinstance(observation, Observation)
        response = await client.post("/v1/observations", json=observation.model_dump(mode="json"))
        assert response.status_code in (200, 201), response.text
        bodies.append(response.json())
    return bodies


async def client_for(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_the_demo_question_gets_a_truthful_answer(app: FastAPI) -> None:
    """The whole product, through HTTP.

    Keys placed, seen, picked up. The answer must name the coffee table *and*
    say it is no longer there.
    """
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await post_all(client, keys_placed_then_picked_up())

            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert answer["answer_status"] == "last_confirmed_only"
    assert answer["current_location"] is None
    assert answer["last_confirmed_placement"]["surface"] == "coffee_table"
    assert "coffee table" in answer["spoken_answer"]
    assert "picked up" in answer["spoken_answer"]


async def test_an_undisturbed_object_is_confirmed_only_with_loadable_evidence(
    app: FastAPI, settings: Settings
) -> None:
    """A confirmed answer requires a frame that can actually be produced.

    The fixtures reference evidence that was never uploaded, so this must come
    back as the weaker claim rather than as `confirmed`.
    """
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await post_all(client, keys_placed_and_left())
            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert answer["answer_status"] == "last_confirmed_only"
    assert "cannot confirm" in answer["spoken_answer"]


async def test_uploading_the_evidence_upgrades_the_answer(app: FastAPI) -> None:
    """With the frame stored and loadable, the same state may be confirmed."""
    observations = list(keys_placed_and_left())
    frame = b"not-a-real-jpeg-but-bytes-all-the-same"
    digest = hashlib.sha256(frame).hexdigest()
    # Point the placement at evidence we are about to upload.
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            upload = await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": digest},
                content=frame,
            )
            evidence_id = upload.json()["evidence_id"]

            first = observations[0]
            with_evidence = first.model_copy(
                update={
                    "evidence": (
                        first.evidence[0].model_copy(
                            update={"evidence_id": evidence_id, "sha256": digest}
                        ),
                    )
                }
            )
            await post_all(client, [with_evidence, observations[1]])
            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert answer["answer_status"] == "confirmed"
    assert answer["current_location"]["surface"] == "coffee_table"


async def test_evidence_with_a_wrong_digest_is_refused(app: FastAPI) -> None:
    """A frame nobody can verify is not evidence."""
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            response = await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": "0" * 64},
                content=b"some bytes",
            )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"


async def test_an_unknown_object_is_admitted_not_guessed(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            answer = (await client.post("/v1/query", json={"label": "wallet"})).json()

    assert answer["answer_status"] == "unknown"
    assert answer["current_location"] is None
    assert "no record" in answer["spoken_answer"]


async def test_a_repeated_observation_reports_the_original_outcome(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            first = keys_placed_and_left()[0].model_dump(mode="json")
            created = await client.post("/v1/observations", json=first)
            repeated = await client.post("/v1/observations", json=first)

    assert created.status_code == 201
    assert repeated.status_code == 200
    assert repeated.json()["duplicate"] is True


async def test_a_placement_without_a_location_is_rejected(app: FastAPI) -> None:
    """The model refuses it, so the API answers 422 rather than storing it."""
    payload = keys_placed_and_left()[0].model_dump(mode="json")
    payload["location"] = None

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            response = await client.post("/v1/observations", json=payload)

    assert response.status_code == 422


async def test_lifecycle_turns_an_in_transit_object_unknown(app: FastAPI) -> None:
    """The gateway names an epoch; Memory fans it out to the objects."""
    envelope = {
        "signal_id": "lc_01JABC",
        "idempotency_key": f"{DEVICE}/{SESSION}/{EPOCH}/track_lost",
        "session_id": SESSION,
        "device_id": DEVICE,
        "signal": {
            "action": "track_lost",
            "source": "media_gateway",
            "occurred_at": (T0 + dt.timedelta(minutes=4)).isoformat().replace("+00:00", "Z"),
            "reason": "track_unsubscribed",
        },
        "scope": {"media_epoch_id": EPOCH, "object_id": None, "track_id": None},
        "provenance": {"component": "media-gateway", "version": "0.1.0"},
    }

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await post_all(client, keys_placed_then_picked_up())
            result = (await client.post("/v1/lifecycle", json=envelope)).json()
            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert result["affected"] == 1
    assert answer["answer_status"] == "last_confirmed_only"
    assert answer["current_status"] == "unknown"


async def test_deleting_a_session_removes_records_and_files(
    app: FastAPI, settings: Settings
) -> None:
    """docs/07: deletion removes database records and evidence together."""
    frame = b"evidence-bytes"
    digest = hashlib.sha256(frame).hexdigest()

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": digest},
                content=frame,
            )
            await post_all(client, keys_placed_and_left())

            deleted = (await client.delete(f"/v1/sessions/{SESSION}")).json()
            after = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert deleted["deleted"]["observations"] == 2
    assert deleted["deleted"]["evidence_files"] == 1
    assert after["answer_status"] == "unknown"
    assert (
        not list((Path(settings.evidence_dir) / SESSION).glob("*"))
        if (Path(settings.evidence_dir) / SESSION).exists()
        else True
    )


async def test_status_reports_the_thresholds_an_evaluation_must_cite(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await post_all(client, keys_placed_and_left())
            body = (await client.get("/v1/status")).json()

    assert body["config"]["promote_min_event_confidence"] == 0.7
    assert body["totals"]["observations"] == 2
    assert body["totals"]["observations_promoted"] == 2
    assert body["objects_by_status"] == {"confirmed_at_location": 1}


async def test_writes_are_refused_without_the_configured_token(settings: Settings) -> None:
    app = create_app(settings.model_copy(update={"internal_api_token": TOKEN}))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            response = await client.post(
                "/v1/observations", json=keys_placed_and_left()[0].model_dump(mode="json")
            )

    assert response.status_code == 401


async def test_an_unlisted_device_cannot_write(settings: Settings) -> None:
    """An allowlist is only a control if the write path checks it."""
    app = create_app(settings.model_copy(update={"device_id_allowlist": ("someone-else",)}))

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            response = await client.post(
                "/v1/observations", json=keys_placed_and_left()[0].model_dump(mode="json")
            )

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


async def test_health_reports_ready_when_the_database_answers(app: FastAPI) -> None:
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            live = await client.get("/health/live")
            ready = await client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


async def test_the_answer_carries_a_working_evidence_url(app: FastAPI) -> None:
    """The URL in an answer must actually fetch the bytes.

    Asserted by following it, not by matching a string: a route that moved
    would leave the string test green and the demo broken.
    """
    observations = list(keys_placed_and_left())
    frame = b"a-frame-standing-in-for-a-clip"
    digest = hashlib.sha256(frame).hexdigest()

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            upload = await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": digest},
                content=frame,
                headers={"content-type": "video/mp4"},
            )
            evidence_id = upload.json()["evidence_id"]

            first = observations[0]
            with_evidence = first.model_copy(
                update={
                    "evidence": (
                        first.evidence[0].model_copy(
                            update={"evidence_id": evidence_id, "sha256": digest}
                        ),
                    )
                }
            )
            await post_all(client, [with_evidence, observations[1]])

            answer = (await client.post("/v1/query", json={"label": "keys"})).json()
            placement = answer["last_confirmed_placement"]
            fetched = await client.get(placement["evidence_url"])

    assert placement["evidence_url"] == f"/v1/evidence/{evidence_id}"
    # A clip is carried exactly like a frame; the client picks the element.
    assert placement["evidence_media_type"] == "video/mp4"
    assert fetched.status_code == 200
    assert fetched.content == frame


async def test_no_url_is_offered_when_the_evidence_is_gone(
    app: FastAPI, settings: Settings
) -> None:
    """Retention deletes files while rows survive.

    A row pointing at a deleted file looks exactly like a valid one, so the
    answer must offer no link at all rather than one that 404s.
    """
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            await post_all(client, keys_placed_and_left())
            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert answer["last_confirmed_placement"]["evidence_url"] is None
    assert answer["answer_status"] == "last_confirmed_only"


async def test_a_public_base_url_is_used_when_configured(settings: Settings) -> None:
    """Same-origin callers get a relative path; anyone else needs an absolute one."""
    app = create_app(settings.model_copy(update={"public_base_url": "https://gn100.local:8081/"}))
    frame = b"bytes"
    digest = hashlib.sha256(frame).hexdigest()
    observations = list(keys_placed_and_left())

    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            upload = await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": digest},
                content=frame,
            )
            evidence_id = upload.json()["evidence_id"]
            first = observations[0]
            await post_all(
                client,
                [
                    first.model_copy(
                        update={
                            "evidence": (
                                first.evidence[0].model_copy(
                                    update={"evidence_id": evidence_id, "sha256": digest}
                                ),
                            )
                        }
                    ),
                    observations[1],
                ],
            )
            answer = (await client.post("/v1/query", json={"label": "keys"})).json()

    assert answer["last_confirmed_placement"]["evidence_url"] == (
        f"https://gn100.local:8081/v1/evidence/{evidence_id}"
    )


async def test_oversized_evidence_is_refused(app: FastAPI, settings: Settings) -> None:
    """`request.body()` reads it all into memory, so the cap must precede that.

    Matters more now that evidence may be a clip rather than a frame.
    """
    small = create_app(settings.model_copy(update={"max_evidence_bytes": 64}))
    payload = b"x" * 128

    async with small.router.lifespan_context(small):
        async with await client_for(small) as client:
            response = await client.post(
                "/v1/evidence",
                params={"session_id": SESSION, "sha256": hashlib.sha256(payload).hexdigest()},
                content=payload,
            )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request"
    assert "larger than the configured limit" in response.json()["message"]


async def test_events_reports_each_ingested_observation_once(app: FastAPI) -> None:
    """`/v1/events` (the publisher dev page's live log) must see every real
    write, in order, and must not double-count a re-delivered duplicate."""
    async with app.router.lifespan_context(app):
        async with await client_for(app) as client:
            observations = keys_placed_then_picked_up()
            await post_all(client, observations)
            # A duplicate re-delivery of the first observation must not add
            # a second entry -- see observations.py's `if result.duplicate`.
            await client.post("/v1/observations", json=observations[0].model_dump(mode="json"))

            body = (await client.get("/v1/events")).json()

    assert [e["action"] for e in body["events"]] == [o.event.action for o in observations]
    assert body["events"][0]["label"] == "keys"
    assert body["events"][-1]["promoted"] is True
