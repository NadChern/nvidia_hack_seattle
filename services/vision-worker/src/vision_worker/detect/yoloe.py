"""YOLOE detector: warm-start lifecycle, text/prompt-free modes, class-
embedding cache.

Ported from a prior first-person AR project on this exact glasses hardware
(RayNeo X3 Pro) that proved YOLOE viable for this exact task -- two warm
checkpoints, one for known targets via text prompts, one prompt-free for
open vocabulary. See `docs/02-Model-Landscape.md`'s "YOLOE: detector
candidate" for the full context.

Simplified from that port in two ways:
  - No mask-polygon/contour extraction. This service's `Detection` contract
    (`visual_memory_vision_contract.protocol.Detection`) has no contour
    field -- centroid and box are all the stability machine and the
    geometry layer need -- so this drops the `cv2.approxPolyDP` step
    entirely, along with the direct `cv2` dependency it required.
  - No request-concurrency locking. The prior project served concurrent
    HTTP requests and needed to serialize GPU access across them; this
    service's `Pipeline` calls `detect()` sequentially, once per relay
    frame, with nothing else ever calling it at the same time.

`torch` and `ultralytics` are imported inside `_load_blocking`, not at
module level, so this module stays importable -- and `main.py` can
reference `YoloeDetector` unconditionally -- even in a profile that never
installed the `models` extra. The import only runs, and can only fail, when
`initialize()` is actually called because `VMA_DETECTOR_KIND=yoloe` was
selected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.identity.base import SegmentedDetection

logger = logging.getLogger(__name__)

_WARMUP_IMAGE_SHAPE = (480, 640, 3)  # H, W, C
_WARMUP_LABELS = ("item",)


def _select_device(runtime: Any, preferred: str | None = None) -> str:
    """Pick the accelerator this machine actually has, or honour an override.

    CUDA first: it is the deploy target and the only one measured. Then MPS,
    Apple's Metal backend, so a Mac gets its GPU instead of silently running
    a large model on CPU cores -- `sys.platform` is not consulted, because
    `torch.backends.mps.is_available()` is the question that matters and it
    answers False on an Intel Mac, an unsupported macOS, or a torch built
    without it.

    `preferred` exists for the case this cannot detect: MPS silently falls
    back to CPU for operations Metal has no kernel for, and YOLOE's
    text-prompt path runs CLIP, so a Mac that produces wrong results or
    errors needs a way back to `cpu` that is not a code edit. Set
    `VMA_YOLOE_DEVICE=cpu`.
    """
    if preferred is not None:
        return preferred
    if runtime.cuda.is_available():
        return "cuda"
    # `getattr`, because `torch.backends.mps` predates neither every build
    # nor every version this might run against, and an AttributeError here
    # would fail startup on a machine that simply has no Metal.
    mps = getattr(runtime.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _to_numpy(value: Any) -> NDArray[np.float64]:
    """Convert a torch tensor or numpy array to a numpy array on CPU."""
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class YoloeDetector:
    """`Detector` backed by two warm YOLOE checkpoints: text-prompt for known
    targets, prompt-free for open vocabulary -- used when `labels` is empty,
    matching `Detector.detect`'s documented meaning for an empty label set.
    """

    def __init__(
        self,
        *,
        text_model: str = "yoloe-26l-seg.pt",
        prompt_free_model: str = "yoloe-26l-seg-pf.pt",
        score_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self._text_model_name = text_model
        self._prompt_free_model_name = prompt_free_model
        self._score_threshold = score_threshold
        #: `None` means detect it -- see `_select_device`.
        self._preferred_device = device

        self._text_model: Any | None = None
        self._pf_model: Any | None = None
        self._torch: Any | None = None
        self._device = "pending"
        self._load_duration_ms = 0.0
        self._warmup_ms = 0.0
        self._request_count = 0
        self._average_latency_ms = 0.0
        #: Cache: frozenset(labels) -> (ordered_labels, embedding_tensor). A
        #: session's detection labels stay constant for its whole duration,
        #: so this saves a CLIP text-encode on every frame after the first.
        self._embedding_cache: dict[frozenset[str], tuple[list[str], Any]] = {}
        # Per-track identity runs off the frame loop and calls `segment()`.
        # YOLOE mutates its active text classes before inference, so detector
        # and identity forwards must not overlap on the shared warm model.
        self._inference_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self._text_model is not None and self._pf_model is not None

    async def initialize(self) -> None:
        if self.is_ready:
            return
        loop = asyncio.get_running_loop()

        load_started = time.perf_counter()
        await loop.run_in_executor(None, self._load_blocking)
        self._load_duration_ms = (time.perf_counter() - load_started) * 1000.0

        warmup_started = time.perf_counter()
        await loop.run_in_executor(None, self._warmup_blocking)
        self._warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

        logger.info(
            "yoloe ready",
            extra={
                "text_model": self._text_model_name,
                "prompt_free_model": self._prompt_free_model_name,
                "device": self._device,
                "load_duration_ms": round(self._load_duration_ms, 1),
                "warmup_ms": round(self._warmup_ms, 1),
            },
        )

    def readiness_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "detector": "yoloe",
            "ready": self.is_ready,
            "device": self._device,
            "text_model": self._text_model_name,
            "prompt_free_model": self._prompt_free_model_name,
            "score_threshold": self._score_threshold,
            "load_duration_ms": round(self._load_duration_ms, 1),
            "warmup_ms": round(self._warmup_ms, 1),
            "request_count": self._request_count,
            "average_latency_ms": round(self._average_latency_ms, 1),
            "embedding_cache_size": len(self._embedding_cache),
        }
        gpu = self._gpu_memory_mb()
        if gpu is not None:
            payload["gpu_allocated_mb"], payload["gpu_reserved_mb"] = gpu
        return payload

    async def detect(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[Detection]:
        if not self.is_ready:
            raise RuntimeError("YoloeDetector.detect() called before initialize()")

        started = time.perf_counter()
        prompt_free = not labels
        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            result, resolved_labels, model = await loop.run_in_executor(
                None, self._infer_blocking, frame_rgb, list(labels), prompt_free
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._request_count += 1
        self._average_latency_ms = (
            self._average_latency_ms * (self._request_count - 1) + elapsed_ms
        ) / self._request_count

        if result is None:
            return ()
        return tuple(self._postprocess(result, resolved_labels, model))

    async def segment(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[SegmentedDetection]:
        """Run the same `-seg` checkpoint but retain its in-process masks."""
        if not self.is_ready:
            raise RuntimeError("YoloeDetector.segment() called before initialize()")
        prompt_free = not labels
        loop = asyncio.get_running_loop()
        async with self._inference_lock:
            result, resolved_labels, model = await loop.run_in_executor(
                None, self._infer_blocking, frame_rgb, list(labels), prompt_free
            )
        if result is None:
            return ()
        detections = self._postprocess(result, resolved_labels, model)
        masks = getattr(result, "masks", None)
        mask_data = _to_numpy(getattr(masks, "data", None))
        if mask_data.ndim != 3 or len(mask_data) != len(detections):
            return ()
        height, width = frame_rgb.shape[:2]
        return tuple(
            SegmentedDetection(
                detection=detection,
                mask=_resize_mask(mask_data[index] > 0.5, height=height, width=width),
            )
            for index, detection in enumerate(detections)
        )

    async def aclose(self) -> None:
        self._text_model = None
        self._pf_model = None

    # ------ Blocking work, always run off the event loop -------------------

    def _load_blocking(self) -> None:
        # torch and ultralytics arrive with the optional `models` extra, which
        # the ci and dev-macos profiles deliberately never install (see
        # pyproject.toml). Unsuppressed, `uv run pyright` would pass on a
        # machine that happens to have the CUDA wheels and fail on one that
        # does not -- the same gate, two answers. Binding them through `Any`,
        # which is how every other reference in this module already holds
        # them, makes the result identical in every profile.
        import torch  # type: ignore[import-not-found]
        from ultralytics import YOLOE  # type: ignore[import-not-found]

        runtime: Any = torch
        self._torch = runtime
        self._device = _select_device(runtime, self._preferred_device)
        logger.info("loading yoloe", extra={"device": self._device})

        # An `Any` annotation is not enough for these two: pyright narrows the
        # declaration back to `Unknown | Any` from the constructor call, so the
        # suppression has to sit on the lines themselves.
        text_model = YOLOE(self._text_model_name)  # type: ignore[reportUnknownVariableType]
        text_model.to(self._device)  # type: ignore[reportUnknownMemberType]
        pf_model = YOLOE(self._prompt_free_model_name)  # type: ignore[reportUnknownVariableType]
        pf_model.to(self._device)  # type: ignore[reportUnknownMemberType]

        self._text_model = text_model
        self._pf_model = pf_model

    def _warmup_blocking(self) -> None:
        assert self._text_model is not None and self._pf_model is not None
        warmup_image = np.zeros(_WARMUP_IMAGE_SHAPE, dtype=np.uint8)
        warmup_labels: list[str] = list(_WARMUP_LABELS)
        embeddings = self._text_model.get_text_pe(warmup_labels)
        self._text_model.set_classes(warmup_labels, embeddings)
        self._embedding_cache[frozenset(warmup_labels)] = (warmup_labels, embeddings)
        self._text_model.predict(warmup_image, verbose=False)
        self._pf_model.predict(warmup_image, verbose=False)
        self._cuda_synchronize()

    def _infer_blocking(
        self, frame_rgb: NDArray[np.uint8], labels: list[str], prompt_free: bool
    ) -> tuple[Any, list[str] | None, Any]:
        """Select a model, apply class prompts, run one forward pass.

        Returns `(result, resolved_labels, model)` so `_postprocess` can
        resolve labels from index without holding a reference to whichever
        model ran.
        """
        torch = self._torch
        assert torch is not None
        if prompt_free:
            model = self._pf_model
            assert model is not None
            resolved_labels: list[str] | None = None
        else:
            model = self._text_model
            assert model is not None
            cached_labels, embeddings = self._resolve_embeddings(labels)
            model.set_classes(cached_labels, embeddings)
            resolved_labels = cached_labels

        with torch.inference_mode():
            results = model.predict(frame_rgb, conf=self._score_threshold, verbose=False)
        self._cuda_synchronize()

        result = results[0] if results else None
        return result, resolved_labels, model

    def _resolve_embeddings(self, labels: list[str]) -> tuple[list[str], Any]:
        if not labels:
            raise ValueError("text-prompt mode requires at least one label")
        key = frozenset(labels)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        text_model = self._text_model
        assert text_model is not None
        embeddings = text_model.get_text_pe(labels)
        self._embedding_cache[key] = (list(labels), embeddings)
        return list(labels), embeddings

    def _postprocess(self, result: Any, labels: list[str] | None, model: Any) -> list[Detection]:
        """Convert one ultralytics `Results` object into typed `Detection`s.
        Boxes are normalized to `[0, 1]`; the centroid is the box center."""
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        if xyxy.size == 0:
            return []

        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape is None or len(orig_shape) < 2:
            raise ValueError("ultralytics result missing orig_shape (h, w)")
        height, width = int(orig_shape[0]), int(orig_shape[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid orig_shape: {orig_shape!r}")

        confs = _to_numpy(boxes.conf)
        cls_idx = _to_numpy(boxes.cls).astype(int)
        fallback_names = getattr(model, "names", None)

        detections: list[Detection] = []
        for i in range(len(xyxy)):
            label = self._resolve_label(int(cls_idx[i]), labels, fallback_names)
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            box = BoundingBox(
                x_min=x1 / width, y_min=y1 / height, x_max=x2 / width, y_max=y2 / height
            )
            centroid = Point2D(x=(box.x_min + box.x_max) / 2.0, y=(box.y_min + box.y_max) / 2.0)
            detections.append(
                Detection(label=label, confidence=float(confs[i]), box=box, centroid=centroid)
            )
        return detections

    @staticmethod
    def _resolve_label(idx: int, labels: list[str] | None, fallback_names: Any) -> str:
        if labels is not None and 0 <= idx < len(labels):
            return labels[idx]
        # `fallback_names` comes from ultralytics' `model.names` (untyped in
        # its own stubs); the isinstance checks narrow it, but pyright still
        # sees Unknown type parameters, so cast to Any explicitly rather than
        # let strict mode flag the eventual `str(...)` call.
        if isinstance(fallback_names, dict):
            mapping = cast(dict[Any, Any], fallback_names)
            if idx in mapping:
                return str(mapping[idx])
        elif isinstance(fallback_names, list | tuple):
            sequence = cast("list[Any] | tuple[Any, ...]", fallback_names)
            if 0 <= idx < len(sequence):
                return str(sequence[idx])
        return f"class_{idx}"

    # ------ Helpers ----------------------------------------------------------

    def _gpu_memory_mb(self) -> tuple[float, float] | None:
        torch = self._torch
        if self._device != "cuda" or torch is None or not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return round(allocated, 1), round(reserved, 1)

    def _cuda_synchronize(self) -> None:
        torch = self._torch
        if self._device == "cuda" and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()


def _resize_mask(mask: NDArray[np.bool_], *, height: int, width: int) -> NDArray[np.bool_]:
    if mask.shape == (height, width):
        return mask
    source_height, source_width = mask.shape
    rows = np.minimum(
        (np.arange(height, dtype=np.float64) * source_height / height).astype(int),
        source_height - 1,
    )
    columns = np.minimum(
        (np.arange(width, dtype=np.float64) * source_width / width).astype(int),
        source_width - 1,
    )
    return mask[rows[:, None], columns[None, :]]


__all__ = ["YoloeDetector"]
