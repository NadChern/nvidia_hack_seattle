"""The world-motion veto, with a scripted reconstruction instead of DA3.

`pose/da3.py` needs a checkpoint, a GPU, and a package this service cannot
even declare (see its module docstring). None of that is needed to test the
decision this module actually makes: given where the object was in the world
across a window, does that contradict what the candidate claims? A fake
`WindowPoseSource` supplies the geometry directly, so the whole veto runs in
milliseconds with no model -- the same discipline `detect/fixture.py` applies
to detection.

The scenario numbers come from real measurements on `media/clips`: keys that
never move back-project to 0.4% of scene-scale drift, which is what
`STATIONARY_DRIFT` stands in for.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    EvidenceWindow,
    Point2D,
    TrackSample,
    VerifierResult,
)

from vision_worker.domain.geometry import CameraIntrinsics, CapturePose, Quaternion
from vision_worker.pose.base import FrameGeometry, WindowGeometry
from vision_worker.verify.rules import RuleBasedVerifier, RuleBasedVerifierConfig
from vision_worker.verify.world_motion import (
    DID_NOT_MOVE,
    STILL_MOVING,
    WORLD_AGREES,
    WorldMotionConfig,
    WorldMotionVerifier,
)

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
SCENE_SCALE = 1.0
#: What a genuinely stationary object measured at, on real footage.
STATIONARY_DRIFT = 0.004
#: Comfortably past `WorldMotionConfig.still_ratio`.
CARRIED_DRIFT = 0.4

_REF = DetectorRef(name="fixture", checkpoint="n/a", revision="v1")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedWindowPose:
    """Returns a reconstruction in which the object sits at `offsets[i]`.

    The camera is placed at the origin looking down +z with a centered
    principal point, so a detection at the frame center back-projects to
    `(0, 0, range)` -- which makes the object's world track exactly the
    range sequence, and the test's intent readable from its inputs.
    """

    def __init__(self, offsets: Sequence[float], *, ready: bool = True) -> None:
        self._offsets = offsets
        self._ready = ready
        self.calls = 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def initialize(self) -> None:
        return None

    async def estimate(self, frames: Sequence[NDArray[np.uint8]]) -> WindowGeometry | None:
        del frames
        self.calls += 1
        views = [
            FrameGeometry(
                capture_pose=CapturePose(position=_point(0.0, 0.0, 0.0), rotation=Quaternion()),
                intrinsics=CameraIntrinsics(focal_px=100.0),
                ranges=np.full((8, 8), 1.0 + offset, dtype=np.float64),
            )
            for offset in self._offsets
        ]
        return WindowGeometry(frames=tuple(views), scene_scale=SCENE_SCALE, is_metric=False)

    def readiness_payload(self) -> Mapping[str, object]:
        return {"pose": "scripted"}

    async def aclose(self) -> None:
        return None


def _point(x: float, y: float, z: float):  # noqa: ANN202
    from visual_memory_vision_contract.protocol import Point3D

    return Point3D(x=x, y=y, z=z)


def a_detection(confidence: float = 0.9) -> Detection:
    return Detection(
        label="keys",
        confidence=confidence,
        box=BoundingBox(x_min=0.45, y_min=0.45, x_max=0.55, y_max=0.55),
        centroid=Point2D(x=0.5, y=0.5),
    )


def a_candidate(action: str) -> CandidateEvent:
    return CandidateEvent(
        candidate_id="cand_1",
        session_id="sess_1",
        device_id="glasses-01",
        media_epoch_id="TR_VCaaa",
        track_id="track-1",
        label="keys",
        action=action,  # type: ignore[arg-type]
        window=EvidenceWindow(
            window_started_at=T0, window_ended_at=T0 + dt.timedelta(seconds=1), frame_count=4
        ),
        object_candidate=a_detection(),
        detector=_REF,
        tracker=_REF,
        state_machine_version="v1",
        pipeline_version="v1",
    )


def some_samples(count: int = 4) -> tuple[TrackSample, ...]:
    return tuple(
        TrackSample(
            track_id="track-1",
            frame_index=i,
            captured_at=T0 + dt.timedelta(seconds=i / 8),
            detection=a_detection(),
        )
        for i in range(count)
    )


def a_verifier(offsets: Sequence[float], *, ready: bool = True) -> WorldMotionVerifier:
    return WorldMotionVerifier(
        RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.5, min_frame_count=1)),
        ScriptedWindowPose(offsets, ready=ready),
        config=WorldMotionConfig(),
    )


def some_frames(count: int = 4) -> tuple[bytes, ...]:
    return tuple(b"jpeg" for _ in range(count))


def decoded(count: int = 4) -> tuple[NDArray[np.uint8], ...]:
    return tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(count))


# --- The veto ---------------------------------------------------------------


async def test_a_pickup_of_something_that_never_moved_is_rejected() -> None:
    """The measured failure: three `picked_up` candidates on footage where
    the keys never left the desk. Every one of them was the camera panning.
    """
    verifier = a_verifier([0.0, STATIONARY_DRIFT, 0.0, STATIONARY_DRIFT])

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "rejected"
    assert result.reason_code == DID_NOT_MOVE


async def test_a_real_pickup_survives() -> None:
    """The object genuinely travels across the window -- exactly what the
    veto must not touch, or it would suppress the demo's own event."""
    verifier = a_verifier([0.0, 0.1, 0.25, CARRIED_DRIFT])

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"
    assert result.reason_code == WORLD_AGREES


async def test_wrapper_preserves_the_inner_verifiers_resolved_fields() -> None:
    """A world-motion relabel must not erase the VLM's corrected action or description."""

    class RichVerifier:
        async def verify(
            self,
            candidate: CandidateEvent,
            *,
            frames: Sequence[bytes],
            samples: Sequence[TrackSample] = (),
            decoded: Sequence[NDArray[np.uint8]] = (),
        ) -> VerifierResult:
            del frames, samples, decoded
            return VerifierResult(
                candidate_id=candidate.candidate_id,
                outcome="confirmed",
                reason_code="vlm_confirmed",
                latency_ms=10.0,
                verifier=DetectorRef(name="fixture-vlm", checkpoint="n/a", revision="v1"),
                occurred_at=T0,
                resolved_action="carried",
                description="beside the laptop",
            )

    verifier = WorldMotionVerifier(
        RichVerifier(), ScriptedWindowPose([0.0, 0.1, 0.25, CARRIED_DRIFT])
    )

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"
    assert result.reason_code == WORLD_AGREES
    assert result.resolved_action == "carried"
    assert result.description == "beside the laptop"


async def test_a_placement_still_travelling_at_the_end_is_rejected() -> None:
    """ "Came to rest" is a claim about the end of the window, so a window
    whose final frames are still moving contradicts it."""
    verifier = a_verifier([0.0, 0.1, 0.3, 0.6])

    result = await verifier.verify(
        a_candidate("placed"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "rejected"
    assert result.reason_code == STILL_MOVING


async def test_a_placement_that_settles_survives() -> None:
    verifier = a_verifier([0.0, 0.3, 0.31, 0.31])

    result = await verifier.verify(
        a_candidate("placed"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"
    assert result.reason_code == WORLD_AGREES


# --- Degrading ---------------------------------------------------------------


async def test_a_candidate_the_rules_rejected_is_returned_untouched() -> None:
    """The veto only ever subtracts. A reconstruction says whether something
    moved; it has no opinion on whether the detection was trustworthy, so it
    must never promote a verdict the rules refused."""
    pose = ScriptedWindowPose([0.0, 0.1, 0.25, CARRIED_DRIFT])
    verifier = WorldMotionVerifier(
        RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.99)), pose
    )

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "rejected"
    assert result.reason_code == "below_confidence_threshold"
    assert pose.calls == 0, "must not pay for a reconstruction it cannot use"


async def test_an_unavailable_pose_source_leaves_the_verdict_alone() -> None:
    """No checkpoint, no CUDA, no `depth-anything-3` installed -- the common
    case, and it must cost accuracy rather than availability."""
    verifier = a_verifier([0.0, 0.0, 0.0, 0.0], ready=False)

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"


async def test_a_window_with_no_samples_leaves_the_verdict_alone() -> None:
    """Without per-frame positions there is no trajectory to judge."""
    verifier = a_verifier([0.0, 0.0, 0.0, 0.0])

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=(), decoded=decoded()
    )

    assert result.outcome == "confirmed"


async def test_a_reconstruction_that_fails_leaves_the_verdict_alone() -> None:
    class FailingPose(ScriptedWindowPose):
        async def estimate(self, frames: Sequence[NDArray[np.uint8]]) -> WindowGeometry | None:
            del frames
            return None

    verifier = WorldMotionVerifier(
        RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.5)), FailingPose([0.0])
    )

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"


# --- Metric vs. ratio thresholds --------------------------------------------


class MetricWindowPose(ScriptedWindowPose):
    """Same scripted geometry, but flagged as anchored to real metres."""

    async def estimate(self, frames: Sequence[NDArray[np.uint8]]) -> WindowGeometry | None:
        window = await super().estimate(frames)
        assert window is not None
        return WindowGeometry(frames=window.frames, scene_scale=window.scene_scale, is_metric=True)


async def test_metric_windows_are_judged_in_metres() -> None:
    """5cm of travel is past the 3cm metric threshold, so a pickup stands --
    even though the same 5cm is under the 2%-of-scene ratio threshold that
    would apply to an unanchored window. The two must not be confused."""
    verifier = WorldMotionVerifier(
        RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.5)),
        MetricWindowPose([0.0, 0.02, 0.04, 0.05]),
        config=WorldMotionConfig(still_m=0.03, still_ratio=0.5),
    )

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "confirmed"


async def test_unanchored_windows_fall_back_to_ratios() -> None:
    """The identical trajectory, without a metric anchor: now the ratio
    threshold governs, and 5cm of a 1.0-unit scene is under it."""
    verifier = WorldMotionVerifier(
        RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.5)),
        ScriptedWindowPose([0.0, 0.02, 0.04, 0.05]),
        config=WorldMotionConfig(still_m=0.03, still_ratio=0.5),
    )

    result = await verifier.verify(
        a_candidate("picked_up"), frames=some_frames(), samples=some_samples(), decoded=decoded()
    )

    assert result.outcome == "rejected"
    assert result.reason_code == DID_NOT_MOVE


def test_thresholds_for_reports_metres_only_when_anchored() -> None:
    config = WorldMotionConfig(still_m=0.03, settled_m=0.08, still_ratio=0.02, settled_ratio=0.06)
    frames = ()

    anchored = WindowGeometry(frames=frames, scene_scale=2.0, is_metric=True)
    native = WindowGeometry(frames=frames, scene_scale=2.0, is_metric=False)

    assert config.thresholds_for(anchored) == (0.03, 0.08)
    # Ratios are relative to the scene, so a deeper scene tolerates more.
    assert config.thresholds_for(native) == (0.04, 0.12)
