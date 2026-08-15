"""Ultralytics YOLO26 metric-depth adapter for the live overlay.

Uses the model runtime already required by YOLOE, avoiding MoGe's additional
~1.3 GB checkpoint and substantially larger resident model on constrained
laptop GPUs. The checkpoint emits metric z-depth in metres. Per detection we
report the median valid depth inside its bounding box, matching the MoGe
adapter's robust sampling semantics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Detection

logger = logging.getLogger(__name__)

_WARMUP_IMAGE_SHAPE = (480, 640, 3)
_DEFAULT_MODEL = "yolo26m-depth.pt"


class YoloDepthEstimator:
    """Metric depth backed by an Ultralytics YOLO26 depth checkpoint."""

    def __init__(self, *, model: str = _DEFAULT_MODEL, device: str | None = None) -> None:
        self._model_name = model
        self._preferred_device = device
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = "pending"
        self._load_state = "not_started"
        self._failure_reason = ""
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
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_blocking)
            warmup = np.zeros(_WARMUP_IMAGE_SHAPE, dtype=np.uint8)
            await loop.run_in_executor(None, self._infer_blocking, warmup)
            self._load_state = "ready"
            logger.info(
                "yolo depth ready",
                extra={"model": self._model_name, "device": self._device},
            )
        except Exception as error:  # noqa: BLE001 -- optional enhancement
            self._model = None
            self._load_state = "failed"
            self._failure_reason = f"{error.__class__.__name__}: {error}"
            logger.exception("yolo depth unavailable; continuing without metric depth")

    def readiness_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "depth": "yolo",
            "ready": self.is_ready,
            "device": self._device,
            "model": self._model_name,
            "load_state": self._load_state,
            "request_count": self._request_count,
            "average_latency_ms": round(self._average_latency_ms, 1),
        }
        if self._failure_reason:
            payload["failure_reason"] = self._failure_reason
        return payload

    async def depth_map(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.float64] | None:
        if not self.is_ready:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._infer_blocking, frame_rgb)

    async def estimate(
        self, frame_rgb: NDArray[np.uint8], detections: Sequence[Detection]
    ) -> Sequence[Detection]:
        if not self.is_ready or not detections:
            return tuple(detections)
        started = time.perf_counter()
        depth = await self.depth_map(frame_rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._request_count += 1
        self._average_latency_ms = (
            self._average_latency_ms * (self._request_count - 1) + elapsed_ms
        ) / self._request_count
        if depth is None:
            return tuple(detections)

        height, width = frame_rgb.shape[:2]
        annotated: list[Detection] = []
        for detection in detections:
            range_m = _sample_depth(detection, depth, width, height)
            annotated.append(
                detection.model_copy(update={"depth_m": range_m})
                if range_m is not None
                else detection
            )
        return tuple(annotated)

    async def aclose(self) -> None:
        self._model = None
        self._load_state = "not_started"

    def _load_blocking(self) -> None:
        import torch  # type: ignore[import-not-found]
        from ultralytics import YOLO  # type: ignore[import-not-found]

        runtime: Any = torch
        self._torch = runtime
        self._device = self._preferred_device or ("cuda" if runtime.cuda.is_available() else "cpu")
        model = YOLO(self._model_name)  # type: ignore[reportUnknownVariableType]
        model.to(self._device)  # type: ignore[reportUnknownMemberType]
        self._model = model

    def _infer_blocking(self, frame_rgb: NDArray[np.uint8]) -> NDArray[np.float64] | None:
        model = self._model
        if model is None:
            return None
        results = model.predict(frame_rgb, device=self._device, verbose=False)
        if not results or results[0].depth is None:
            return None
        data = results[0].depth.data
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        return np.asarray(data, dtype=np.float64)


def _sample_depth(
    detection: Detection, depth: NDArray[np.float64], width: int, height: int
) -> float | None:
    """Median finite positive metric depth inside a normalized box."""
    x1 = int(np.clip(detection.box.x_min * width, 0, width - 1))
    x2 = int(np.clip(detection.box.x_max * width, x1 + 1, width))
    y1 = int(np.clip(detection.box.y_min * height, 0, height - 1))
    y2 = int(np.clip(detection.box.y_max * height, y1 + 1, height))
    values = depth[y1:y2, x1:x2].reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


__all__ = ["YoloDepthEstimator"]
