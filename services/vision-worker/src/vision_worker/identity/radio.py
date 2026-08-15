"""Lazy C-RADIOv4 identity embedder with mask-weighted spatial pooling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any, cast

import numpy as np

from vision_worker.identity.base import EmbeddingVectors, MaskedCrop

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/C-RADIOv4-SO400M"
MODEL_REVISION = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"
POOLING = "summary+mask-weighted-spatial-v1"


def _dynamic(value: Any) -> Any:
    """Collapse untyped remote-model values to one explicit adapter boundary."""
    return value


def _select_device(runtime: Any, preferred: str | None = None) -> str:
    if preferred is not None:
        return preferred
    if runtime.cuda.is_available():
        return "cuda"
    mps = getattr(runtime.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class RadioEmbedder:
    """Transformers-native C-RADIO adapter; load failure degrades identity only."""

    embedder_id = f"{MODEL_ID}@{MODEL_REVISION}"
    pooling = POOLING

    def __init__(self, *, device: str | None = None, batch_size: int = 1) -> None:
        self._preferred_device = device
        self._batch_size = batch_size
        self._device = "pending"
        self._torch: Any | None = None
        self._model: Any | None = None
        self._load_state = "pending"
        self._load_error: str | None = None
        self._load_duration_ms = 0.0
        self._request_count = 0
        self._average_latency_ms = 0.0

    @property
    def is_ready(self) -> bool:
        return self._load_state == "ready" and self._model is not None

    async def initialize(self) -> None:
        if self._load_state in {"ready", "failed"}:
            return
        self._load_state = "loading"
        started = time.perf_counter()
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._load_blocking)
        except Exception as exc:
            # Identity annotates and never vetoes. A failed backbone must not
            # make the detector, event pipeline, or service unavailable.
            self._model = None
            self._load_state = "failed"
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("C-RADIO identity embedder failed to load; identity disabled")
        else:
            self._load_state = "ready"
        self._load_duration_ms = (time.perf_counter() - started) * 1000.0

    async def embed(self, crops: Sequence[MaskedCrop]) -> Sequence[EmbeddingVectors]:
        if not self.is_ready:
            raise RuntimeError("C-RADIO identity embedder is not ready")
        if not crops:
            return ()
        started = time.perf_counter()
        result = await asyncio.get_running_loop().run_in_executor(
            None, self._embed_blocking, tuple(crops)
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self._request_count += len(crops)
        self._average_latency_ms = (
            self._average_latency_ms * (self._request_count - len(crops)) + elapsed
        ) / self._request_count
        return result

    def readiness_payload(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "identity_embedder": "c-radio-v4",
            "ready": self.is_ready,
            "load_state": self._load_state,
            "device": self._device,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "pooling": POOLING,
            "load_duration_ms": round(self._load_duration_ms, 1),
            "request_count": self._request_count,
            "average_latency_ms": round(self._average_latency_ms, 1),
        }
        if self._load_error is not None:
            payload["error"] = self._load_error
        return payload

    async def aclose(self) -> None:
        self._model = None
        runtime = self._torch
        if runtime is not None and self._device == "cuda" and runtime.cuda.is_available():
            runtime.cuda.empty_cache()

    def _load_blocking(self) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModel  # type: ignore[import-not-found]

        runtime: Any = torch
        self._torch = runtime
        self._device = _select_device(runtime, self._preferred_device)
        model = cast(
            Any,
            AutoModel.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
            ),
        )
        model.eval()
        model.to(self._device)
        switch_to_deploy = getattr(model, "switch_to_deploy", None)
        if callable(switch_to_deploy):
            switch_to_deploy()
        self._model = model

    def _embed_blocking(self, crops: Sequence[MaskedCrop]) -> tuple[EmbeddingVectors, ...]:
        runtime = self._torch
        model = self._model
        assert runtime is not None and model is not None
        results: list[EmbeddingVectors] = []
        for start in range(0, len(crops), self._batch_size):
            batch = crops[start : start + self._batch_size]
            pixels = np.stack(
                [np.transpose(crop.image.astype(np.float32) / 255.0, (2, 0, 1)) for crop in batch]
            )
            tensor = runtime.from_numpy(pixels).to(self._device)
            autocast = (
                runtime.autocast(device_type="cuda", dtype=runtime.float16)
                if self._device == "cuda"
                else nullcontext()
            )
            with runtime.inference_mode(), autocast:
                output = _dynamic(model(tensor))
                backbone = _dynamic(output["backbone"] if isinstance(output, dict) else output)
                summaries = _dynamic(backbone.summary.float())
                features = _dynamic(backbone.features.float())
                feature_dim = int(_dynamic(features.shape[2]))
                summary_dim = int(_dynamic(summaries.shape[1]))
                if summary_dim != feature_dim:
                    if summary_dim % feature_dim:
                        raise ValueError(
                            f"C-RADIO summary dim {summary_dim} is not compatible with "
                            f"spatial dim {feature_dim}"
                        )
                    # C-RADIOv4 exposes one summary per distilled teacher and
                    # flattens them. Store one stable 1,152-d vector by mean-
                    # pooling those summary tokens, matching the spatial dim.
                    summaries = _dynamic(
                        summaries.reshape(summaries.shape[0], -1, feature_dim).mean(dim=1)
                    )
                token_count = int(_dynamic(features.shape[1]))
                grid = round(token_count**0.5)
                if grid * grid != token_count:
                    raise ValueError(f"C-RADIO returned non-square spatial tokens: {token_count}")
                masks = runtime.from_numpy(
                    np.stack([crop.mask.astype(np.float32) for crop in batch])[:, None]
                ).to(self._device)
                weights = (
                    runtime.nn.functional.interpolate(masks, size=(grid, grid), mode="nearest")
                    .flatten(2)
                    .transpose(1, 2)
                )
                pooled = _dynamic(
                    (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                )
                summaries = _dynamic(runtime.nn.functional.normalize(summaries, dim=1))
                pooled = _dynamic(runtime.nn.functional.normalize(pooled, dim=1))

            summary_np = summaries.cpu().numpy().astype(np.float32, copy=False)
            pooled_np = pooled.cpu().numpy().astype(np.float32, copy=False)
            for summary, spatial in zip(summary_np, pooled_np, strict=True):
                if summary.shape != spatial.shape:
                    raise ValueError(
                        f"C-RADIO summary/spatial dims differ: {summary.shape} vs {spatial.shape}"
                    )
                results.append(
                    EmbeddingVectors(
                        embedder_id=self.embedder_id,
                        pooling=self.pooling,
                        summary=summary,
                        pooled_spatial=spatial,
                    )
                )
        return tuple(results)


__all__ = ["MODEL_ID", "MODEL_REVISION", "POOLING", "RadioEmbedder", "_select_device"]
