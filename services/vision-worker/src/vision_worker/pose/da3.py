"""Depth Anything 3: joint depth *and* camera pose across a window of frames.

The adapter that makes "did it move in the room" answerable without the
glasses. `pose/image_motion.py` compares an object's screen motion against
the background's, which on handheld footage cannot separate a carried object
from a panning head -- measured on `media/clips`, no threshold does: tight
and camera motion reads as object motion, loose and real carrying stops
registering. DA3 sidesteps the question by reconstructing where the camera
actually was, so an object's world position can be compared against itself.

Measured on this machine (RTX 4070 Laptop, 8188 MiB) over 13 frames of
`media/clips/01-placed-on-table.MOV`, where the object never moves:

  - back-projected world drift: 0.4% of scene scale (the image-space path
    reported three pickups on the same footage)
  - 283 ms/view, 1.6 GB weights, 6.0 GB peak at 13 views
  - `DA3-LARGE-1.1` is **not metric**; `DA3METRIC-LARGE` is, but returns no
    extrinsics at all, so it cannot do this job. Metric distance stays
    `depth/moge.py`'s to report.

Two consequences shape everything below.

**Peak VRAM scales with window length**, and 13 views already costs 6.0 GB of
8 GB with a detector also resident. `max_views` is a hard cap, not a hint: a
window longer than that is subsampled evenly rather than reconstructed whole,
because OOM during a demo is worse than a coarser trajectory.

**Scale is per inference call.** Nothing from one `estimate()` can be
compared with anything from another -- see `WindowGeometry`. This is why the
adapter returns `is_metric=False` and a `scene_scale` denominator rather than
pretending to metres.

`torch` and `depth_anything_3` are imported inside `_load_blocking`, not at
module level, so this module stays importable in a profile that never
installed the runtime -- the same discipline `detect/yoloe.py` and
`depth/moge.py` use, for the same reason. Here that is load-bearing rather
than tidy: **`depth-anything-3` is not a declared dependency of this
service**, because it requires `numpy<2` and the workspace pins
`numpy>=2.4.6` (see the note where the extras are declared in
`pyproject.toml`). Install it into a separate environment with the same
`torch==2.6.0` pin; where the import fails, `initialize()` records the error,
`is_ready` stays False, and `verify/world_motion.py` falls back to the rule
verifier.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vision_worker.depth.base import MetricDepthReference
from vision_worker.domain.geometry import CameraIntrinsics, CapturePose, Quaternion
from vision_worker.pose.base import FrameGeometry, WindowGeometry

logger = logging.getLogger(__name__)

#: A reconstruction needs at least two views to have any pose information at
#: all -- one view is a monocular depth estimate with the camera at the origin
#: by definition, which says nothing about whether anything moved.
MIN_VIEWS = 2

#: Below this many pixels valid in both maps, a scale fit is guesswork and the
#: window keeps its native units instead.
_MIN_SCALE_PIXELS = 1000


class Da3WindowPose:
    """Reconstructs a window of frames jointly via Depth Anything 3."""

    def __init__(
        self,
        *,
        model_id: str = "depth-anything/DA3-LARGE-1.1",
        max_views: int = 8,
        process_res: int = 504,
        metric_reference: MetricDepthReference | None = None,
        scale_reference_views: int = 2,
    ) -> None:
        if max_views < MIN_VIEWS:
            raise ValueError(f"max_views must be at least {MIN_VIEWS}, got {max_views}")
        self._model_id = model_id
        self._max_views = max_views
        self._process_res = process_res
        #: Optional metric anchor. With one, this adapter's output is in
        #: metres; without, in units meaningful only inside one window. See
        #: `_fit_scale`.
        self._metric_reference = metric_reference
        self._scale_reference_views = max(1, scale_reference_views)
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._lock = asyncio.Lock()
        self._call_count = 0
        self._average_latency_ms = 0.0
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    async def initialize(self) -> None:
        """Load the checkpoint, degrading to `is_ready=False` on failure.

        A pose-reconstruction failure must not take the service down: the
        pipeline predates this adapter and still works without it, falling
        back to the image-space signal. That is worse, not broken.
        """
        async with self._lock:
            if self._model is not None or self._load_error is not None:
                return
            started = time.perf_counter()
            try:
                await asyncio.to_thread(self._load_blocking)
            except Exception as exc:  # noqa: BLE001 -- see the docstring
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "DA3 unavailable; falling back to image-space motion",
                    extra={"model_id": self._model_id, "error": self._load_error},
                )
                return
            logger.info(
                "DA3 ready",
                extra={
                    "model_id": self._model_id,
                    "device": self._device,
                    "load_ms": round((time.perf_counter() - started) * 1000.0),
                },
            )

    async def estimate(self, frames: Sequence[NDArray[np.uint8]]) -> WindowGeometry | None:
        if self._model is None or len(frames) < MIN_VIEWS:
            return None

        selected = _subsample(frames, self._max_views)
        started = time.perf_counter()
        async with self._lock:
            try:
                raw = await asyncio.to_thread(self._reconstruct_blocking, selected)
            except Exception:
                # Most likely CUDA OOM on a window this card cannot hold.
                # A missing reconstruction degrades the verdict; a raised one
                # would drop a candidate that the rule check could still judge.
                logger.exception("DA3 reconstruction failed", extra={"views": len(selected)})
                return None
        if raw is None:
            return None

        # The scale fit is a ratio *against* this reconstruction, so it can
        # only happen once the reconstruction exists. Outside the lock: the
        # metric adapter has its own GPU serialization, and nesting the two
        # would make a window cost the sum of both.
        depth, intrinsics, extrinsics = raw
        scale = await self._fit_scale(selected, depth)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._call_count += 1
        self._average_latency_ms = (
            self._average_latency_ms * (self._call_count - 1) + elapsed_ms
        ) / self._call_count
        return _build_geometry(depth, intrinsics, extrinsics, scale=scale)

    async def _fit_scale(
        self, frames: Sequence[NDArray[np.uint8]], depth: NDArray[np.float64]
    ) -> float | None:
        """The factor turning this reconstruction's units into metres.

        `median(metric_depth / reconstructed_depth)` over every pixel valid in
        both, on a couple of the window's frames. Median rather than a
        least-squares fit because the two models disagree hardest exactly
        where one of them is wrong -- sky, glass, an object's silhouette --
        and a mean would let those pixels set the scale for the whole window.

        `None` means the window keeps its own arbitrary units, which every
        consumer already handles: no metric adapter configured, its checkpoint
        unavailable, or too few pixels the two models agree exist.

        Measured on `media/clips/01-placed-on-table`: the per-frame factor
        varies only 2.4% across 13 frames, which is what makes one factor per
        window sound -- fitting it per frame would inject that spread as
        noise into the very trajectory being judged.
        """
        reference = self._metric_reference
        if reference is None or not reference.is_ready:
            return None

        ratios: list[float] = []
        for index in _reference_indices(len(frames), self._scale_reference_views):
            metric = await reference.depth_map(frames[index])
            if metric is None:
                continue
            ratio = _median_ratio(metric, depth[index])
            if ratio is not None:
                ratios.append(ratio)

        if not ratios:
            logger.debug("no usable metric reference for this window; keeping native units")
            return None
        return float(np.median(ratios))

    def readiness_payload(self) -> Mapping[str, object]:
        return {
            "pose": "da3",
            "model_id": self._model_id,
            "ready": self.is_ready,
            "device": self._device,
            "max_views": self._max_views,
            "calls": self._call_count,
            "average_latency_ms": round(self._average_latency_ms, 1),
            "load_error": self._load_error,
            "metric_reference": self._metric_reference is not None
            and self._metric_reference.is_ready,
        }

    async def aclose(self) -> None:
        self._model = None

    # ------ Blocking work, always run off the event loop -------------------

    def _load_blocking(self) -> None:
        # Ships with the optional `models` extra, absent in the ci and
        # dev-macos profiles; see `detect/yoloe.py._load_blocking` for why it
        # is bound through `Any` rather than left to resolve.
        import torch  # type: ignore[import-not-found]

        runtime: Any = torch
        self._torch = runtime
        self._device = "cuda" if runtime.cuda.is_available() else "cpu"
        model: Any = _depth_anything_3().from_pretrained(self._model_id)
        self._model = model.to(self._device).eval()

    def _reconstruct_blocking(
        self, frames: Sequence[NDArray[np.uint8]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
        """One forward pass: `(depth, intrinsics, extrinsics)` in DA3's own
        units. Scaling happens above, where the metric reference lives."""
        torch = self._torch
        model = self._model
        assert torch is not None and model is not None

        with torch.inference_mode():
            prediction = model.inference(list(frames), process_res=self._process_res)

        depth = np.asarray(prediction.depth, dtype=np.float64)
        intrinsics = prediction.intrinsics
        extrinsics = prediction.extrinsics
        if extrinsics is None or intrinsics is None:
            # DA3METRIC-LARGE returns depth but no extrinsics -- a valid
            # checkpoint for depth/, useless here. Say so rather than
            # silently producing poses at the origin.
            logger.warning(
                "DA3 checkpoint returned no camera pose; it cannot serve as a window pose source",
                extra={"model_id": self._model_id},
            )
            return None

        return (
            depth,
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(extrinsics, dtype=np.float64),
        )


def _build_geometry(
    depth: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    extrinsics: NDArray[np.float64],
    *,
    scale: float | None,
) -> WindowGeometry | None:
    """Assemble a reconstruction, in metres when `scale` is known.

    **Depth and pose translations scale together.** Applying the factor to
    only one of them would leave a self-consistent-looking geometry in which
    every object sits at the wrong distance from a camera that moved the
    wrong amount -- the failure mode that looks plausible right up until a
    number is reported to a person.
    """
    factor = 1.0 if scale is None else scale
    scaled_depth = depth * factor

    geometries = [
        _frame_geometry(scaled_depth[i], intrinsics[i], extrinsics[i], translation_scale=factor)
        for i in range(len(scaled_depth))
    ]
    finite = scaled_depth[np.isfinite(scaled_depth) & (scaled_depth > 0.0)]
    scene_scale = float(np.median(finite)) if finite.size else 0.0
    if scene_scale <= 0.0:
        return None

    return WindowGeometry(
        frames=tuple(geometries), scene_scale=scene_scale, is_metric=scale is not None
    )


def _median_ratio(metric: NDArray[np.float64], native: NDArray[np.float64]) -> float | None:
    """`median(metric / native)` over pixels valid in both, resampling the
    metric map onto the reconstruction's grid first -- the two models work at
    different resolutions and a ratio needs them pixel-aligned."""
    height, width = native.shape
    rows = np.asarray(_even_indices(metric.shape[0], height), dtype=np.intp)
    cols = np.asarray(_even_indices(metric.shape[1], width), dtype=np.intp)
    resampled = metric[rows[:, None], cols[None, :]]

    usable = np.isfinite(resampled) & np.isfinite(native) & (resampled > 0.0) & (native > 0.0)
    if usable.sum() < _MIN_SCALE_PIXELS:
        return None
    return float(np.median(resampled[usable] / native[usable]))


def _even_indices(source_length: int, wanted: int) -> list[int]:
    """`wanted` indices spread evenly across `source_length`, endpoints
    included -- nearest-neighbour resampling, which is all a scale fit needs
    and avoids interpolating across the depth discontinuities that would
    corrupt the ratio."""
    if wanted <= 1 or source_length <= 1:
        return [0] * max(wanted, 1)
    step = (source_length - 1) / (wanted - 1)
    return [min(round(index * step), source_length - 1) for index in range(wanted)]


def _reference_indices(count: int, wanted: int) -> list[int]:
    """Which views to spend a metric inference on -- spread across the
    window rather than clustered, so a single badly-estimated frame cannot
    set the scale alone."""
    if wanted >= count:
        return list(range(count))
    return list(dict.fromkeys(np.linspace(0, count - 1, wanted).round().astype(int).tolist()))


def _depth_anything_3() -> Any:
    """The DA3 entry point, imported on demand.

    Undeclared by design -- see the module docstring: it requires `numpy<2`
    and this workspace pins `numpy>=2.4.6`, so no type checker can resolve it
    here and none ever will while that stands. `Any` is the honest annotation
    for a class this environment genuinely knows nothing about, rather than a
    suppression of something fixable.
    """
    import importlib

    api = importlib.import_module("depth_anything_3.api")
    return api.DepthAnything3


def _subsample(frames: Sequence[NDArray[np.uint8]], limit: int) -> Sequence[NDArray[np.uint8]]:
    """Evenly thin `frames` to at most `limit`, always keeping the first and
    last -- a window's endpoints are what a drift measurement subtracts."""
    if len(frames) <= limit:
        return frames
    positions = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    unique: list[int] = list(dict.fromkeys(int(index) for index in positions))
    return [frames[index] for index in unique]


def _frame_geometry(
    depth: NDArray[np.float64],
    intrinsics: NDArray[np.float64],
    extrinsics: NDArray[np.float64],
    *,
    translation_scale: float = 1.0,
) -> FrameGeometry:
    """Convert one view's DA3 output into the geometry the domain layer uses.

    Three conversions, each of which is a place to be wrong:

    1. **z-depth to ray range.** DA3's depth is distance along the optical
       axis; `compute_world_point` scales a *unit view ray*, so each pixel's
       z is multiplied by the length of its unnormalized camera-space ray.
       Ignoring this is correct only at the principal point and increasingly
       wrong toward the corners.
    2. **Extrinsics to a capture pose.** DA3 returns world-to-camera
       `[R|t]`; a capture pose is camera-to-world, so the position is
       `-R^T t` and the rotation is `R^T`.
    3. **Matrix to quaternion.** `CapturePose.rotation` is a quaternion in
       Unity's `(w, x, y, z)` order, because that is the form ARDK pose will
       arrive in when the glasses path lands.
    """
    rotation_w2c = extrinsics[:3, :3]
    # Scaled with depth, never on its own -- see `_build_geometry`.
    translation_w2c = extrinsics[:3, 3] * translation_scale
    rotation_c2w = rotation_w2c.T
    position = -rotation_c2w @ translation_w2c

    height, width = depth.shape
    ranges = depth * _ray_length_map(intrinsics, width=width, height=height)

    # A single focal length with a centered principal point, which is what
    # `CameraIntrinsics` models. DA3's own fx/fy agree to well within a
    # percent on this footage; the principal point is assumed centered, the
    # same approximation the prior-art project made for this sensor.
    focal_px = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)

    return FrameGeometry(
        capture_pose=CapturePose(
            position=_as_point(position), rotation=_matrix_to_quaternion(rotation_c2w)
        ),
        intrinsics=CameraIntrinsics(focal_px=focal_px),
        ranges=ranges,
    )


def _ray_length_map(
    intrinsics: NDArray[np.float64], *, width: int, height: int
) -> NDArray[np.float64]:
    """`||K^-1 [u, v, 1]||` for every pixel -- the factor turning a z-depth
    into a range along that pixel's view ray."""
    inverse = np.linalg.inv(intrinsics)
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    ones = np.ones_like(us)
    stacked = np.stack([us, vs, ones], axis=-1)
    rays = stacked @ inverse.T
    return np.linalg.norm(rays, axis=-1)


def _as_point(vector: NDArray[np.float64]):  # noqa: ANN202 -- Point3D, avoids a cycle
    from visual_memory_vision_contract.protocol import Point3D

    return Point3D(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> Quaternion:
    """Shepperd's method: pick the largest diagonal term to divide by, so the
    square root never runs into a near-zero denominator on a rotation the
    naive trace formula cannot handle (any 180-degree turn)."""
    m = matrix
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(
            w=0.25 * s,
            x=float(m[2, 1] - m[1, 2]) / s,
            y=float(m[0, 2] - m[2, 0]) / s,
            z=float(m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        return Quaternion(
            w=float(m[2, 1] - m[1, 2]) / s,
            x=0.25 * s,
            y=float(m[0, 1] + m[1, 0]) / s,
            z=float(m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        return Quaternion(
            w=float(m[0, 2] - m[2, 0]) / s,
            x=float(m[0, 1] + m[1, 0]) / s,
            y=0.25 * s,
            z=float(m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
    return Quaternion(
        w=float(m[1, 0] - m[0, 1]) / s,
        x=float(m[0, 2] + m[2, 0]) / s,
        y=float(m[1, 2] + m[2, 1]) / s,
        z=0.25 * s,
    )


__all__ = ["MIN_VIEWS", "Da3WindowPose"]
