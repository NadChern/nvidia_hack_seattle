"""RuleBasedVerifier: the docs/06 rule 8 boundary, and S04's own stop-condition
fallback -- built as the default, not an afterthought."""

from __future__ import annotations

import datetime as dt

import pytest
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    Point2D,
)

from vision_worker.verify.base import Verifier
from vision_worker.verify.rules import (
    BELOW_CONFIDENCE_THRESHOLD,
    EVIDENCE_MISSING,
    WINDOW_INCOMPLETE,
    RuleBasedVerifier,
    RuleBasedVerifierConfig,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
LATER = T0 + dt.timedelta(seconds=3)

_DETECTOR = DetectorRef(name="yoloe-11s-seg", checkpoint="yoloe-11s-seg.pt", revision="rev-1")
_TRACKER = DetectorRef(name="greedy-iou", checkpoint="n/a", revision="v1")


def a_candidate(*, confidence: float = 0.9, frame_count: int = 12) -> CandidateEvent:
    return CandidateEvent(
        candidate_id="cand_01JABC",
        session_id="sess_01JAB",
        device_id="glasses-01",
        media_epoch_id="TR_VCabc123",
        track_id="track-42",
        label="keys",
        action="placed",
        window=EvidenceWindow(window_started_at=T0, window_ended_at=LATER, frame_count=frame_count),
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


async def test_a_strong_candidate_with_evidence_is_confirmed() -> None:
    verifier = RuleBasedVerifier()

    result = await verifier.verify(a_candidate(), frames=(b"jpeg-bytes",))

    assert result.outcome == "confirmed"
    assert result.candidate_id == "cand_01JABC"
    assert result.verifier.name == "rules"


async def test_no_frames_is_unverified_not_rejected() -> None:
    """Missing evidence is a "we cannot tell", never a confident no."""
    verifier = RuleBasedVerifier()

    result = await verifier.verify(a_candidate(), frames=())

    assert result.outcome == "unverified"
    assert result.reason_code == EVIDENCE_MISSING


async def test_a_window_thinner_than_the_configured_minimum_is_unverified() -> None:
    config = RuleBasedVerifierConfig(min_frame_count=5)
    verifier = RuleBasedVerifier(config)

    result = await verifier.verify(a_candidate(frame_count=2), frames=(b"x",))

    assert result.outcome == "unverified"
    assert result.reason_code == WINDOW_INCOMPLETE


async def test_low_confidence_is_rejected_not_unverified() -> None:
    """Evidence and window are fine; the detection itself is weak -- a
    confident no, distinct from "we cannot tell"."""
    verifier = RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.8))

    result = await verifier.verify(a_candidate(confidence=0.3), frames=(b"x",))

    assert result.outcome == "rejected"
    assert result.reason_code == BELOW_CONFIDENCE_THRESHOLD


async def test_latency_is_measured_and_non_negative() -> None:
    verifier = RuleBasedVerifier()

    result = await verifier.verify(a_candidate(), frames=(b"x",))

    assert result.latency_ms >= 0.0


async def test_the_config_is_exposed_for_status_reporting() -> None:
    config = RuleBasedVerifierConfig(min_confidence=0.75, min_frame_count=3)
    verifier = RuleBasedVerifier(config)

    assert verifier.config.min_confidence == 0.75
    assert verifier.config.min_frame_count == 3


async def test_satisfies_the_verifier_protocol() -> None:
    verifier: Verifier = RuleBasedVerifier()
    assert verifier is not None
