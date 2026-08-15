from __future__ import annotations

import numpy as np
import pytest
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.depth.yolo import YoloDepthEstimator

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def a_detection() -> Detection:
    return Detection(
        label="mug",
        confidence=0.9,
        box=BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75),
        centroid=Point2D(x=0.5, y=0.5),
    )


async def test_metric_depth_is_added_to_each_detection() -> None:
    estimator = YoloDepthEstimator()
    estimator._load_state = "ready"  # noqa: SLF001 -- adapter boundary test
    estimator._model = object()  # noqa: SLF001

    async def depth_map(_: np.ndarray) -> np.ndarray:
        return np.full((8, 8), 1.75, dtype=np.float64)

    estimator.depth_map = depth_map  # type: ignore[method-assign]
    [annotated] = await estimator.estimate(np.zeros((8, 8, 3), dtype=np.uint8), [a_detection()])

    assert annotated.depth_m == 1.75
