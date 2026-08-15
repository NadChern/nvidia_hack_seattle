"""What decides "does the camera think it moved" -- the signal
`domain/stability.py` compares an object's own motion against.

Two shapes live here, because the two useful answers arrive on different
schedules.

**Per frame (`PoseSource`).** `image_motion.py`, the default, estimates
background ego-motion from consecutive frames with no model and no extra
dependency; `device.py` (later, gated on the glasses uplink spike) consumes
ARDK 6DoF pose over the relay's data channel instead. Cheap enough to run on
every frame, and feeds `TrackSample.background_motion`.

**Per window (`WindowPoseSource`).** `da3.py` reconstructs a whole span of
frames jointly and returns a real camera pose for each. That cannot be a
`PoseSource`: it costs ~283ms per view against a 125ms frame budget at 8fps,
and -- decisively -- its scale is fixed per *inference call*, so world points
from two separate calls are in different units and cannot be compared to each
other at all. Since judging "did this object move" means comparing positions
over time, the comparison has to happen inside one reconstruction. That makes
it a candidate-window tool (`verify/world_motion.py`), not a streaming one.

Neither shape is imported by `domain/`. Both ultimately feed
`TrackSample.background_motion` or a world position computed through
`domain/geometry.py`, which is what actually decides anything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Point2D

from vision_worker.domain.geometry import CameraIntrinsics, CapturePose


class PoseSource(Protocol):
    """Stateful across frames within one media epoch. `reset()` on every
    `epoch_started` -- comparing this epoch's first frame against the
    previous epoch's last one would estimate motion across a cut that never
    happened.
    """

    def reset(self) -> None: ...

    def observe(self, frame_rgb: NDArray[np.uint8]) -> Point2D | None:
        """Return this frame's background displacement relative to the last
        one, normalized to frame fractions. `None` on the first frame after a
        reset -- there is nothing yet to compare against.
        """
        ...


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """Where the camera was for one frame, and how to range a pixel from it."""

    capture_pose: CapturePose
    intrinsics: CameraIntrinsics
    #: Per-pixel **range along the view ray** -- not a z-depth, matching what
    #: `domain.geometry.compute_world_point` consumes. Indexed `[row, col]` at
    #: the reconstruction's own resolution, which is not the source frame's:
    #: callers address it in normalized coordinates via the helpers below and
    #: never need to know it.
    ranges: NDArray[np.float64]

    def range_in_box(self, x_min: float, y_min: float, x_max: float, y_max: float) -> float | None:
        """Median range over a normalized box, or `None` if nothing valid
        falls inside it.

        Median rather than the centroid pixel, for the reason `depth/moge.py`
        gives: a centroid landing on the object's edge samples the surface
        behind it, and one such frame is enough to fake a jump in world
        position.
        """
        height, width = self.ranges.shape
        col_min = int(np.clip(round(x_min * (width - 1)), 0, width - 1))
        col_max = int(np.clip(round(x_max * (width - 1)), col_min, width - 1))
        row_min = int(np.clip(round(y_min * (height - 1)), 0, height - 1))
        row_max = int(np.clip(round(y_max * (height - 1)), row_min, height - 1))

        patch = self.ranges[row_min : row_max + 1, col_min : col_max + 1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if valid.size else None


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    """One joint reconstruction of a window of frames.

    **Every value here shares one arbitrary scale**, set by this
    reconstruction and meaningful only within it. Two `WindowGeometry`
    objects are not comparable -- not their poses, not their ranges, not
    world points derived from them. `scene_scale` exists so a caller can
    express a judgment as a fraction ("the object drifted 0.5% of the scene")
    rather than in units that mean nothing outside this object.
    """

    frames: Sequence[FrameGeometry]
    #: A representative distance across the reconstruction (the median range),
    #: the denominator that turns a drift into a scale-free ratio.
    scene_scale: float
    #: True only when the adapter's units are real metres. `da3.py`'s
    #: pose-capable checkpoint is *not* metric, so this is normally False and
    #: a caller must not report these numbers as distances.
    is_metric: bool = False


class WindowPoseSource(Protocol):
    """Reconstructs camera geometry across a span of frames at once.

    Lifecycle matches the other model adapters (`detect/`, `depth/`): an
    async `initialize()` that may fail without taking the process down, and
    `aclose()`. `is_ready` stays False when a checkpoint could not be loaded,
    and callers degrade rather than crash -- the pipeline works without this,
    it just cannot tell head motion from object motion as well.
    """

    @property
    def is_ready(self) -> bool: ...

    async def initialize(self) -> None: ...

    async def estimate(self, frames: Sequence[NDArray[np.uint8]]) -> WindowGeometry | None:
        """Reconstruct `frames` jointly. `None` when unavailable or when the
        window is too short to reconstruct."""
        ...

    def readiness_payload(self) -> Mapping[str, object]: ...

    async def aclose(self) -> None: ...


__all__ = [
    "FrameGeometry",
    "PoseSource",
    "WindowGeometry",
    "WindowPoseSource",
]
