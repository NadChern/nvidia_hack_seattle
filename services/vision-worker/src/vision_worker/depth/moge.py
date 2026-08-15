"""MoGe-2 monocular metric-depth estimator.

Ported from the prior first-person AR project's `vision/depth.py`, which
proved MoGe-2 viable on this exact glasses hardware and task: same model id
default, same warm-start/executor-offload lifecycle as `detect/yoloe.py`,
same median-over-bbox range sampling, same optional oriented-box PCA fit.
Runs on the same RGB frame `detect/` just saw and returns each `Detection`
annotated with `depth_m` -- the metric range (metres) along the detection's
view ray, which `domain/geometry.py`'s `compute_world_point` needs once a
capture pose exists (task #46).

**Graceful degradation is load-bearing, unlike `detect/yoloe.py`.** The
primary detector failing to load should crash the service -- there is no
useful "detector-less" mode. Depth is different: `domain/stability.py`'s
image-space path works with no depth at all, and a `placed` observation
with a null `Location` is already a documented, honest state (see
`emit/memory.py`). So `initialize()` here catches its own load failure,
leaves `is_ready` False, and lets the service start anyway -- annotating
nothing is a degraded pipeline, not a broken one.

`torch` and the `moge` package are imported inside `_load_blocking`, not at
module level, for the same reason `detect/yoloe.py` does: this module stays
importable, and `main.py` can reference `MogeDepthEstimator` unconditionally,
even in a profile that never installed the `models` extra.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Box3D, Detection, Point3D

logger = logging.getLogger(__name__)

_WARMUP_IMAGE_SHAPE = (480, 640, 3)  # H, W, C

#: ViT-L "normal" -- confirmed to exist on Hugging Face by the prior-art
#: project, ~326M params, comfortably under 8GB of VRAM next to a YOLOE-small
#: checkpoint on this class of card (see `docs/02-Model-Landscape.md`).
_DEFAULT_MODEL_ID = "Ruicheng/moge-2-vitl-normal"

# 3D bounding box fit -- ported from `vision/depth.py`'s `_fit_box3d`
# thresholds unchanged, since they were tuned against real detections on
# this exact task and are independent of the runtime they run in.
_BOX3D_MIN_POINTS = 50
_BOX3D_PLANAR_RATIO = 1e-3
_BOX3D_RANGE_MAD_K = 3.0
_BOX3D_PCT_LO = 2.0
_BOX3D_PCT_HI = 98.0
_BOX3D_MAX_HALF_EXTENT_M = 1.0


class MogeDepthEstimator:
    """`DepthEstimator` backed by one warm MoGe-2 checkpoint."""

    def __init__(self, *, model_id: str = _DEFAULT_MODEL_ID, emit_box3d: bool = False) -> None:
        self._model_id = model_id
        self._emit_box3d = emit_box3d

        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "pending"
        self._load_state = "not_started"
        self._failure_reason = ""
        self._load_duration_ms = 0.0
        self._warmup_ms = 0.0
        self._request_count = 0
        self._average_latency_ms = 0.0

    @property
    def is_ready(self) -> bool:
        return self._load_state == "ready" and self._model is not None

    async def initialize(self) -> None:
        if self.is_ready or self._load_state == "loading":
            return
        self._load_state = "loading"
        self._failure_reason = ""
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            await loop.run_in_executor(None, self._load_blocking)
            warmup_started = time.perf_counter()
            await loop.run_in_executor(None, self._warmup_blocking)
            self._warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
            self._load_duration_ms = (time.perf_counter() - started) * 1000.0
            self._load_state = "ready"
            logger.info(
                "moge-2 ready",
                extra={
                    "model_id": self._model_id,
                    "device": self._device,
                    "load_duration_ms": round(self._load_duration_ms, 1),
                    "warmup_ms": round(self._warmup_ms, 1),
                },
            )
        except Exception as error:  # noqa: BLE001 -- degrade, never crash startup
            self._model = None
            self._load_state = "failed"
            self._failure_reason = f"{error.__class__.__name__}: {error}"
            logger.exception(
                "moge-2 depth estimator unavailable -- the pipeline continues on the "
                "image-space stability path with no depth_m or world_point",
                extra={"model_id": self._model_id},
            )

    def readiness_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "depth": "moge",
            "ready": self.is_ready,
            "device": self._device,
            "model_id": self._model_id,
            "load_state": self._load_state,
            "load_duration_ms": round(self._load_duration_ms, 1),
            "warmup_ms": round(self._warmup_ms, 1),
            "request_count": self._request_count,
            "average_latency_ms": round(self._average_latency_ms, 1),
            "emit_box3d": self._emit_box3d,
        }
        if self._failure_reason:
            payload["failure_reason"] = self._failure_reason
        return payload

    async def depth_map(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.float64] | None:
        """This frame's metric z-depth, for use as a scale anchor.

        Satisfies `depth/base.py`'s `MetricDepthReference`: `pose/da3.py`
        fits its arbitrary reconstruction units against these metres. Z, not
        range along the ray -- `estimate()` above wants ranges because it is
        placing points, this wants the axis-aligned depth because that is
        what the other model's depth map is also expressed in, and a ratio
        between two different quantities would be a scale factor for nothing.
        """
        if not self.is_ready:
            return None
        loop = asyncio.get_running_loop()
        points, mask = await loop.run_in_executor(None, self._infer_blocking, frame_rgb)
        if points is None:
            return None
        z = points[..., 2].astype(np.float64)
        if mask is not None:
            z[~mask] = np.nan
        return z

    async def estimate(
        self, frame_rgb: NDArray[np.uint8], detections: Sequence[Detection]
    ) -> Sequence[Detection]:
        if not self.is_ready or not detections:
            return tuple(detections)

        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        points, mask = await loop.run_in_executor(None, self._infer_blocking, frame_rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._request_count += 1
        self._average_latency_ms = (
            self._average_latency_ms * (self._request_count - 1) + elapsed_ms
        ) / self._request_count

        if points is None:
            return tuple(detections)

        height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        ranges = np.linalg.norm(points, axis=2)
        annotated: list[Detection] = []
        for detection in detections:
            update: dict[str, object] = {}
            depth = _sample_range(detection, ranges, mask, width, height)
            if depth is not None:
                update["depth_m"] = depth
            if self._emit_box3d:
                box = _fit_box3d(detection, points, mask, width, height)
                if box is not None:
                    update["box3d"] = box
            annotated.append(detection.model_copy(update=update) if update else detection)
        return tuple(annotated)

    async def aclose(self) -> None:
        self._model = None
        self._load_state = "not_started"

    # ------ Blocking work, always run off the event loop -------------------

    def _load_blocking(self) -> None:
        # Both ship with the optional `models` extra, absent in the ci and
        # dev-macos profiles; see `detect/yoloe.py._load_blocking` for why
        # they are bound through `Any` rather than left to resolve.
        import torch  # type: ignore[import-not-found]
        from moge.model.v2 import MoGeModel  # type: ignore[import-untyped]

        runtime: Any = torch
        self._torch = runtime
        self._device = "cuda" if runtime.cuda.is_available() else "cpu"
        logger.info("loading moge-2", extra={"device": self._device})
        # `moge` ships no type stubs (an unreleased, git-installed package;
        # see the `models` extra in pyproject.toml), so pyright sees this
        # whole chain as Unknown -- matching `detect/yoloe.py`'s treatment
        # of other untyped model-runtime calls.
        self._model = (
            MoGeModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
                self._model_id
            )
            .to(self._device)
            .eval()
        )

    def _warmup_blocking(self) -> None:
        warmup_image = np.zeros(_WARMUP_IMAGE_SHAPE, dtype=np.uint8)
        self._infer_blocking(warmup_image)

    def _infer_blocking(
        self, frame_rgb: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.bool_] | None]:
        """Returns `(points, mask)`: `points` is HxWx3 camera-space metres,
        `mask` is HxW validity. `None, None` if the model produced neither."""
        torch = self._torch
        model = self._model
        assert torch is not None and model is not None
        tensor = torch.tensor(frame_rgb.astype("float32") / 255.0).permute(2, 0, 1).to(self._device)
        with torch.inference_mode():
            output = model.infer(tensor)
        points_t = output.get("points")
        mask_t = output.get("mask")
        points = points_t.detach().cpu().numpy() if points_t is not None else None
        mask = mask_t.detach().cpu().numpy().astype(bool) if mask_t is not None else None
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return points, mask


def _sample_range(
    detection: Detection,
    ranges: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
    width: int,
    height: int,
) -> float | None:
    """Median valid range over the detection's bbox (mask-filtered), with a
    centroid-pixel fallback. Median is robust to a centroid landing on a
    depth discontinuity (object edge)."""
    x1 = int(np.clip(detection.box.x_min * width, 0, width - 1))
    x2 = int(np.clip(detection.box.x_max * width, x1 + 1, width))
    y1 = int(np.clip(detection.box.y_min * height, 0, height - 1))
    y2 = int(np.clip(detection.box.y_max * height, y1 + 1, height))

    region = ranges[y1:y2, x1:x2]
    if region.size:
        values = region[mask[y1:y2, x1:x2]] if mask is not None else region.reshape(-1)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            return float(np.median(values))

    px = int(np.clip(detection.centroid.x * width, 0, width - 1))
    py = int(np.clip(detection.centroid.y * height, 0, height - 1))
    value = ranges[py, px]
    if np.isfinite(value) and value > 0:
        return float(value)
    return None


def _object_pixel_mask(
    detection: Detection, valid: NDArray[np.bool_] | None, width: int, height: int
) -> NDArray[np.bool_]:
    """Boolean HxW mask of the detection's bbox, intersected with MoGe's
    validity mask. No contour to rasterize here -- unlike the prior project,
    this service's `Detection` has no contour field (see `detect/yoloe.py`'s
    docstring), so the box is the best object mask available."""
    x1 = int(np.clip(detection.box.x_min * width, 0, width - 1))
    x2 = int(np.clip(detection.box.x_max * width, x1 + 1, width))
    y1 = int(np.clip(detection.box.y_min * height, 0, height - 1))
    y2 = int(np.clip(detection.box.y_max * height, y1 + 1, height))

    obj = np.zeros((height, width), dtype=bool)
    obj[y1:y2, x1:x2] = True
    if valid is not None:
        obj &= valid
    return obj


def _fit_box3d(
    detection: Detection,
    points: NDArray[np.float64],
    valid: NDArray[np.bool_] | None,
    width: int,
    height: int,
) -> Box3D | None:
    """Oriented 3D bounding box via PCA on the detection's MoGe points, with
    a camera-axis-AABB fallback for near-planar point sets. Returns 8
    corners in camera space (metres, MoGe's Y-down convention) -- front face
    CCW as 0..3, then the matching back face 4..7, matching `Box3D`'s
    documented corner order."""
    object_mask = _object_pixel_mask(detection, valid, width, height)
    pts = points[object_mask]
    if pts.size:
        pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < _BOX3D_MIN_POINTS:
        return None

    ranges = np.linalg.norm(pts, axis=1)
    median = float(np.median(ranges))
    mad = float(np.median(np.abs(ranges - median))) + 1e-6
    pts = pts[np.abs(ranges - median) <= _BOX3D_RANGE_MAD_K * mad]
    if pts.shape[0] < _BOX3D_MIN_POINTS:
        return None

    center = pts.mean(axis=0)
    centered = pts - center
    cov = (centered.T @ centered) / float(pts.shape[0])
    try:
        _, singular_values, axes = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None

    if singular_values[0] <= 0 or (singular_values[2] / singular_values[0]) < _BOX3D_PLANAR_RATIO:
        axes = np.eye(3, dtype=centered.dtype)

    projected = centered @ axes.T
    mins = np.percentile(projected, _BOX3D_PCT_LO, axis=0)
    maxs = np.percentile(projected, _BOX3D_PCT_HI, axis=0)
    mid = (mins + maxs) * 0.5
    half = np.minimum((maxs - mins) * 0.5, _BOX3D_MAX_HALF_EXTENT_M)
    mins, maxs = mid - half, mid + half

    front = [
        (mins[0], mins[1], mins[2]),
        (maxs[0], mins[1], mins[2]),
        (maxs[0], maxs[1], mins[2]),
        (mins[0], maxs[1], mins[2]),
    ]
    back = [(x, y, maxs[2]) for (x, y, _) in front]
    corners_axis = np.array(front + back, dtype=centered.dtype)
    corners_camera = corners_axis @ axes + center

    corners = tuple(Point3D(x=float(c[0]), y=float(c[1]), z=float(c[2])) for c in corners_camera)
    return Box3D(corners=corners)  # type: ignore[arg-type]


__all__ = ["MogeDepthEstimator"]
