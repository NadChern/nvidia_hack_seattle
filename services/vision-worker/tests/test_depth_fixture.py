"""FixtureDepthEstimator: the no-GPU, no-model depth path `ci` and
`dev-macos` depend on being real, matching `test_detect_fixture.py`'s role
for detection."""

from __future__ import annotations

import numpy as np
import pytest
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.depth.base import DepthEstimator
from vision_worker.depth.fixture import FixtureDepthEstimator

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


A_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def a_detection(label: str = "keys") -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        box=BoundingBox(x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
        centroid=Point2D(x=0.45, y=0.45),
    )


async def test_annotates_every_detection_with_the_scripted_range() -> None:
    estimator = FixtureDepthEstimator(range_m=2.5)

    [annotated] = await estimator.estimate(A_FRAME, [a_detection()])

    assert annotated.depth_m == 2.5


async def test_a_none_range_leaves_detections_unchanged() -> None:
    estimator = FixtureDepthEstimator(range_m=None)
    detection = a_detection()

    [annotated] = await estimator.estimate(A_FRAME, [detection])

    assert annotated.depth_m is None
    assert annotated == detection


async def test_preserves_order_and_count_across_multiple_detections() -> None:
    estimator = FixtureDepthEstimator(range_m=1.0)

    annotated = await estimator.estimate(A_FRAME, [a_detection("keys"), a_detection("wallet")])

    assert [d.label for d in annotated] == ["keys", "wallet"]
    assert all(d.depth_m == 1.0 for d in annotated)


async def test_an_empty_detection_list_returns_empty() -> None:
    estimator = FixtureDepthEstimator()

    annotated = await estimator.estimate(A_FRAME, [])

    assert annotated == ()


async def test_satisfies_the_depth_estimator_protocol() -> None:
    estimator: DepthEstimator = FixtureDepthEstimator()
    await estimator.initialize()
    assert estimator.readiness_payload()["ready"] is True
    await estimator.aclose()
