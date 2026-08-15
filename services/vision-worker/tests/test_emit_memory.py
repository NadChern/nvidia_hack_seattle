"""MemoryEmitter: uploading evidence and recording a confirmed candidate as
a canonical Observation.

Uses `httpx.MockTransport`, injected via `MemoryClient(transport=...)`, so
this exercises the real request construction and response parsing MemoryClient
does -- not a mock of MemoryClient's own methods, which would never notice a
real wire-format drift.
"""

from __future__ import annotations

import datetime as dt
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image
from visual_memory_memory_contract.client import MemoryClient
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    Point2D,
    VerifierResult,
)

from vision_worker.emit.memory import MemoryEmitter
from vision_worker.evidence.ring import BufferedFrame

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)

_DETECTOR = DetectorRef(name="yoloe-11s-seg", checkpoint="yoloe-11s-seg.pt", revision="rev-1")
_TRACKER = DetectorRef(name="greedy-iou", checkpoint="n/a", revision="v1")
_VERIFIER = DetectorRef(name="rules", checkpoint="n/a", revision="v1")


def a_jpeg_frame(
    offset_seconds: float, *, width: int = 32, height: int = 24, shade: int = 0
) -> BufferedFrame:
    array = np.full((height, width, 3), shade, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG")
    return BufferedFrame(
        captured_at=T0 + dt.timedelta(seconds=offset_seconds),
        payload=buffer.getvalue(),
        width=width,
        height=height,
    )


def a_candidate(*, action: str = "placed", confidence: float = 0.9) -> CandidateEvent:
    started = T0
    ended = T0 + dt.timedelta(seconds=3)
    return CandidateEvent(
        candidate_id="cand_01JABC",
        session_id="sess_01JAB",
        device_id="glasses-01",
        media_epoch_id="TR_VCabc123",
        track_id="track-42",
        label="keys",
        action=action,  # type: ignore[arg-type]
        window=EvidenceWindow(window_started_at=started, window_ended_at=ended, frame_count=5),
        object_candidate=Detection(
            label="keys",
            confidence=confidence,
            box=BoundingBox(x_min=0.41, y_min=0.52, x_max=0.49, y_max=0.58),
            centroid=Point2D(x=0.45, y=0.55),
        ),
        detector=_DETECTOR,
        tracker=_TRACKER,
        state_machine_version="vision-stability-v1",
        pipeline_version="vision-pipeline-v1",
    )


def a_confirmed_result(candidate: CandidateEvent) -> VerifierResult:
    return VerifierResult(
        candidate_id=candidate.candidate_id,
        outcome="confirmed",
        reason_code="meets_confidence_and_evidence_thresholds",
        latency_ms=1.2,
        verifier=_VERIFIER,
        occurred_at=T0 + dt.timedelta(seconds=3, milliseconds=10),
    )


class RecordingMemory:
    """A fake Memory Service: records every call, returns plausible bodies."""

    def __init__(self) -> None:
        self.evidence_requests: list[httpx.Request] = []
        self.observation_requests: list[httpx.Request] = []
        self._next_evidence_id = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/evidence" and request.method == "POST":
            self._next_evidence_id += 1
            self.evidence_requests.append(request)
            return httpx.Response(200, json={"evidence_id": f"ev_{self._next_evidence_id}"})
        if request.url.path == "/v1/observations" and request.method == "POST":
            self.observation_requests.append(request)
            return httpx.Response(200, json={"state": None})
        return httpx.Response(404, json={"detail": "not found"})  # pragma: no cover


def a_client(memory: RecordingMemory) -> MemoryClient:
    return MemoryClient(transport=httpx.MockTransport(memory.handler))


async def test_emit_uploads_a_still_and_a_clip_before_recording() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    frames = [a_jpeg_frame(i, shade=i * 40) for i in range(5)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    assert len(memory.evidence_requests) == 2
    assert len(memory.observation_requests) == 1
    still_request, clip_request = memory.evidence_requests
    assert still_request.headers["content-type"] == "image/jpeg"
    assert clip_request.headers["content-type"] == "video/mp4"


async def test_the_observation_carries_both_evidence_entries() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    frames = [a_jpeg_frame(i) for i in range(5)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    body = json.loads(memory.observation_requests[0].content)
    assert len(body["evidence"]) == 2
    media_types = {item["media_type"] for item in body["evidence"]}
    assert media_types == {"image/jpeg", "video/mp4"}


async def test_a_placed_candidate_gets_a_location_even_with_no_room_or_surface() -> None:
    """Known limitation until depth/geometry land: null fields, but a real
    Location object -- memory-contract's validator only requires the object
    to exist for a placed action, not that its fields are populated."""
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate(action="placed")
    frames = [a_jpeg_frame(0)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    body = json.loads(memory.observation_requests[0].content)
    assert body["location"] is not None
    assert body["location"]["room"] is None
    assert body["location"]["surface"] is None


async def test_a_non_placed_candidate_has_no_location() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate(action="picked_up")
    frames = [a_jpeg_frame(0)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    body = json.loads(memory.observation_requests[0].content)
    assert body["location"] is None


async def test_no_frames_still_records_with_no_evidence() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()

    await emitter.emit(candidate, a_confirmed_result(candidate), [])

    assert memory.evidence_requests == []
    assert len(memory.observation_requests) == 1
    body = json.loads(memory.observation_requests[0].content)
    assert body["evidence"] == []


async def test_mismatched_frame_sizes_fall_back_to_the_still_frame_alone() -> None:
    """A clip-encode failure must not lose the observation -- it degrades to
    a still-only evidence set, per the module's documented fallback."""
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    frames = [a_jpeg_frame(0, width=32, height=24), a_jpeg_frame(1, width=64, height=48)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    assert len(memory.evidence_requests) == 1
    assert memory.evidence_requests[0].headers["content-type"] == "image/jpeg"
    body = json.loads(memory.observation_requests[0].content)
    assert len(body["evidence"]) == 1


async def test_a_non_confirmed_result_is_refused_before_any_http_call() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    rejected = VerifierResult(
        candidate_id=candidate.candidate_id,
        outcome="rejected",
        reason_code="below_confidence_threshold",
        latency_ms=1.0,
        verifier=_VERIFIER,
        occurred_at=T0,
    )

    with pytest.raises(ValueError, match="not confirmed"):
        await emitter.emit(candidate, rejected, [a_jpeg_frame(0)])

    assert memory.evidence_requests == []
    assert memory.observation_requests == []


async def test_the_idempotency_key_is_deterministic_for_the_same_candidate() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    frames = [a_jpeg_frame(0)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)
    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    first, second = (
        json.loads(request.content)["idempotency_key"] for request in memory.observation_requests
    )
    assert first == second


async def test_detector_and_verifier_provenance_are_carried_through() -> None:
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate()
    frames = [a_jpeg_frame(0)]

    await emitter.emit(candidate, a_confirmed_result(candidate), frames)

    body = json.loads(memory.observation_requests[0].content)
    assert body["provenance"]["detector"]["name"] == "yoloe-11s-seg"
    assert body["provenance"]["verifier"]["name"] == "rules"
    assert body["provenance"]["pipeline_version"] == "vision-pipeline-v1"


# --- A question is not an event ---------------------------------------------


async def test_an_unresolved_vanished_candidate_is_refused() -> None:
    """`vanished` is a question, not a claim. A verifier that confirms one
    without saying what happened has answered nothing, and recording it would
    assert an event the memory contract has no word for.
    """
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate(action="vanished")

    with pytest.raises(ValueError, match="not a memory action"):
        await emitter.emit(candidate, a_confirmed_result(candidate), [a_jpeg_frame(0)])

    assert memory.evidence_requests == [], "nothing may be uploaded for a refused candidate"
    assert memory.observation_requests == []


async def test_a_resolved_vanished_candidate_records_what_the_verifier_decided() -> None:
    """The candidate asked; the verifier answered `picked_up`. That answer,
    not the question, is what memory hears."""
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate(action="vanished")
    result = a_confirmed_result(candidate).model_copy(update={"resolved_action": "picked_up"})

    await emitter.emit(candidate, result, [a_jpeg_frame(0)])

    body = json.loads(memory.observation_requests[0].content)
    assert body["event"]["action"] == "picked_up"
    assert "vanished" not in json.dumps(body)


async def test_the_verifier_description_becomes_the_place() -> None:
    """ "On a white desk next to a tablet" is what a person asked for. It goes
    in `surface`; `room` stays null, because nothing here knows the room and
    guessing it is the failure this service exists to avoid."""
    memory = RecordingMemory()
    emitter = MemoryEmitter(a_client(memory))
    candidate = a_candidate(action="placed")
    result = a_confirmed_result(candidate).model_copy(
        update={"description": "a white desk next to a tablet"}
    )

    await emitter.emit(candidate, result, [a_jpeg_frame(0)])

    body = json.loads(memory.observation_requests[0].content)
    assert body["location"]["surface"] == "a white desk next to a tablet"
    assert body["location"]["room"] is None
