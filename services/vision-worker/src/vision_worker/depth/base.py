"""What every depth adapter implements: detections in, the same detections
back out with `depth_m` (and, when the adapter fits one, `box3d`) filled in.

`Detection` is frozen (`visual_memory_vision_contract.protocol._Frozen`), so
an adapter cannot annotate in place the way the prior project's mutable
`Detection` did -- `estimate()` returns a new tuple, one entry per input, in
the same order, matching `Tracker.update`'s own "same order" contract.

Two adapters: `moge.py`, the default (ported from the prior-art `vision/
depth.py`), and `fixture.py`, a scripted no-model stand-in that lets
`Pipeline`'s depth wiring be tested with no GPU -- the same role `detect/
fixture.py` plays for detection. Depth is optional in a way detection is
not: a `Pipeline` with no depth estimator configured (`depth_estimator=
None`) still runs the full image-space stability path, so `initialize()`
failing must never crash the service the way a missing primary detector
would -- see `moge.py`'s own docstring for how it degrades.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Detection


class DepthEstimator(Protocol):
    async def initialize(self) -> None:
        """Load weights and warm up. Called once before the first `estimate()`."""
        ...

    def readiness_payload(self) -> Mapping[str, object]:
        """Reported at `/v1/status` -- matches `Detector.readiness_payload`'s
        role. Includes a `ready` key even when loading failed (see `moge.py`),
        since a depth estimator degrading is expected, not exceptional."""
        ...

    async def estimate(
        self, frame_rgb: NDArray[np.uint8], detections: Sequence[Detection]
    ) -> Sequence[Detection]:
        """Annotate each detection with `depth_m` (and `box3d`, when the
        adapter fits one). Detections a depth estimate could not be produced
        for are returned unchanged, `depth_m=None` -- never dropped; the
        caller's `Detection` count and order must stay stable."""
        ...

    async def aclose(self) -> None: ...


class MetricDepthReference(Protocol):
    """A depth adapter that can also hand over a whole frame's depth, in
    metres, rather than only per-detection ranges.

    This exists for one job: `pose/da3.py` reconstructs a window with
    excellent geometry but arbitrary units, and fitting its scale against a
    genuinely metric depth map turns those units into metres. That needs the
    map, not a handful of sampled ranges -- the fit is a median over every
    pixel valid in both.

    Separate from `DepthEstimator` because not every depth adapter can serve
    as a metric anchor: `fixture.py` scripts a constant and would produce a
    confidently wrong scale factor. Implement this only where the metres are
    real.
    """

    @property
    def is_ready(self) -> bool: ...

    async def depth_map(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.float64] | None:
        """This frame's metric z-depth, `(H, W)` in metres, `NaN` where the
        adapter has no valid estimate. `None` when unavailable entirely."""
        ...


__all__ = ["DepthEstimator", "MetricDepthReference"]
