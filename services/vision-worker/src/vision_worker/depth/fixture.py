"""A depth estimator that annotates every detection with one scripted range
instead of running a model -- the `ci` and `dev-macos` path for `Pipeline`'s
depth wiring, matching `detect/fixture.py`'s role for detection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Detection


class FixtureDepthEstimator:
    """Sets `depth_m=range_m` on every detection passed to `estimate()`.

    `range_m=None` (the default) leaves every detection unchanged, so tests
    can assert the "no depth adapter configured" path and the "always
    annotates" path with the same class.
    """

    def __init__(self, *, range_m: float | None = 1.5) -> None:
        self._range_m = range_m
        self._call_count = 0

    async def initialize(self) -> None:
        return None

    def readiness_payload(self) -> Mapping[str, object]:
        return {
            "depth": "fixture",
            "ready": True,
            "range_m": self._range_m,
            "calls": self._call_count,
        }

    async def estimate(
        self, frame_rgb: NDArray[np.uint8], detections: Sequence[Detection]
    ) -> Sequence[Detection]:
        del frame_rgb  # the fixture estimator never looks at the frame
        self._call_count += 1
        if self._range_m is None:
            return tuple(detections)
        return tuple(
            detection.model_copy(update={"depth_m": self._range_m}) for detection in detections
        )

    async def aclose(self) -> None:
        return None


__all__ = ["FixtureDepthEstimator"]
