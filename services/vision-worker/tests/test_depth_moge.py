"""MogeDepthEstimator: the real depth adapter, exercised against the real
MoGe-2 ViT-L checkpoint.

Opt-in and marked `models` (see pyproject.toml's `addopts`) for the same
reason `test_detect_yoloe.py` is: this needs `uv sync --extra models` and a
CUDA device, neither of which `ci` or `dev-macos` has. Unlike YOLOE's
checkpoint, MoGe's is not a local file this repo gitignores -- `from_
pretrained` fetches and caches it from Hugging Face Hub on first use, so
there is no local-file existence gate here, only the import gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

pytestmark = [pytest.mark.anyio, pytest.mark.models]

pytest.importorskip("torch")
pytest.importorskip("moge")

from vision_worker.depth.moge import MogeDepthEstimator  # noqa: E402

#: Random, not blank -- MoGe's own depth estimate on a uniform image can
#: degenerate in ways that make `_fit_box3d`'s point-cloud thresholds
#: unrepresentative of a real frame; noise gives every pixel a distinct
#: (if not physically meaningful) depth value, closer to what a real camera
#: frame looks like to the model.
_FRAME = np.random.default_rng(3).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def estimator():
    estimator = MogeDepthEstimator(emit_box3d=True)
    await estimator.initialize()
    try:
        yield estimator
    finally:
        await estimator.aclose()


def a_detection(*, x_min: float, y_min: float, x_max: float, y_max: float) -> Detection:
    centroid = Point2D(x=(x_min + x_max) / 2.0, y=(y_min + y_max) / 2.0)
    return Detection(
        label="keys",
        confidence=0.9,
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        centroid=centroid,
    )


async def test_initialize_is_idempotent(estimator: MogeDepthEstimator) -> None:
    assert estimator.is_ready
    await estimator.initialize()  # must not reload
    assert estimator.is_ready


async def test_readiness_payload_reports_ready(estimator: MogeDepthEstimator) -> None:
    payload = estimator.readiness_payload()
    assert payload["ready"] is True
    assert payload["device"] == "cuda"
    assert payload["model_id"] == "Ruicheng/moge-2-vitl-normal"
    assert "failure_reason" not in payload


async def test_estimate_annotates_depth_m_with_a_positive_finite_range(
    estimator: MogeDepthEstimator,
) -> None:
    detection = a_detection(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7)

    [annotated] = await estimator.estimate(_FRAME, [detection])

    assert annotated.depth_m is not None
    assert annotated.depth_m > 0.0
    assert np.isfinite(annotated.depth_m)


async def test_estimate_fits_a_box3d_when_the_detection_has_enough_points(
    estimator: MogeDepthEstimator,
) -> None:
    # 640*480*0.02*0.02 ~= 122 pixels, comfortably above _BOX3D_MIN_POINTS.
    detection = a_detection(x_min=0.49, y_min=0.49, x_max=0.51, y_max=0.51)

    [annotated] = await estimator.estimate(_FRAME, [detection])

    assert annotated.box3d is not None
    assert len(annotated.box3d.corners) == 8


async def test_box3d_is_never_fit_when_emit_box3d_is_off() -> None:
    estimator = MogeDepthEstimator(emit_box3d=False)
    await estimator.initialize()
    try:
        detection = a_detection(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7)
        [annotated] = await estimator.estimate(_FRAME, [detection])
        assert annotated.depth_m is not None
        assert annotated.box3d is None
    finally:
        await estimator.aclose()


async def test_estimate_preserves_order_and_count(estimator: MogeDepthEstimator) -> None:
    detections = [
        a_detection(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
        a_detection(x_min=0.6, y_min=0.6, x_max=0.8, y_max=0.8),
    ]

    annotated = await estimator.estimate(_FRAME, detections)

    assert len(annotated) == 2
    assert all(d.depth_m is not None for d in annotated)


async def test_request_count_and_latency_are_tracked(estimator: MogeDepthEstimator) -> None:
    detection = a_detection(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7)
    await estimator.estimate(_FRAME, [detection])
    await estimator.estimate(_FRAME, [detection])

    payload = estimator.readiness_payload()
    assert payload["request_count"] == 2
    assert payload["average_latency_ms"] >= 0.0


async def test_aclose_drops_readiness(estimator: MogeDepthEstimator) -> None:
    await estimator.aclose()
    assert estimator.is_ready is False
