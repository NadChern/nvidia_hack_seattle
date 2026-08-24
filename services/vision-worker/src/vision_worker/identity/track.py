"""Grounder-free localisation for the register button.

The register button answers the grounder's question -- "which object?" --
physically: the wearer holds the item centred and presses. So localisation is a
fixed **centre box** propagated across the presentation by a video segmenter,
not a per-frame VLM call. This module owns that: a :class:`Tracker` protocol the
enroller depends on, and a :class:`Sam2Tracker` that anchors one centre-box seed
on the middle frame and propagates it both directions.

The heavy ``transformers`` SAM2 import is deferred to first use so importing this
module (for the protocol, or with a fake tracker in tests) stays cheap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import BoundingBox

logger = logging.getLogger(__name__)

#: A held object fills the middle of the frame during a register gesture. The
#: centre-frac sweep found a *generous* box is the robust default: the wallet's
#: looseness at 0.45 was the anchor, not the object, and 0.60 rescued it while
#: the keychain stayed flat. See docs/spikes/register-no-speech.
DEFAULT_CENTRE_FRAC = 0.60

#: SAM2.1-tiny: the register case is the *easy* VOS case -- ~10 s, object
#: deliberately presented, held close, dominant in frame -- so the smallest
#: mature tracker suffices, and it co-resides with C-RADIO on 8 GiB.
DEFAULT_SAM2_ID = "facebook/sam2.1-hiera-tiny"


def centre_box(frac: float) -> BoundingBox:
    """A normalised centre box of side ``frac``, the SAM2 seed and the HUD reticle."""
    half = frac / 2.0
    return BoundingBox(x_min=0.5 - half, y_min=0.5 - half, x_max=0.5 + half, y_max=0.5 + half)


@runtime_checkable
class Tracker(Protocol):
    """Propagate one seed box across a presentation into per-frame boxes.

    Returns ``{frame_index: box}`` in the *input* frame indexing; frames the
    tracker lost the object on are simply absent. Boxes are normalised.
    """

    async def track(
        self, frames: Sequence[NDArray[np.uint8]], seed: BoundingBox
    ) -> dict[int, BoundingBox]: ...


class Sam2Tracker:
    """SAM2 video segmenter, box-prompted from the centre seed.

    Graduated from ``scripts/enroll_offline.py``'s ``track`` (spike:
    register-no-speech), trimmed to operate on already-decoded RGB frames rather
    than a clip, and made awaitable by running the blocking propagation in a
    worker thread.
    """

    def __init__(self, *, model_id: str = DEFAULT_SAM2_ID, device: str | None = None) -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        # Deferred so the module imports without transformers>=4.57 / a GPU, and
        # so tests using a fake Tracker never touch SAM2.
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        # The transformers SAM2 classes ship only partial types, so under strict
        # pyright every method on the model/session reads as unknown. Launder
        # them through Any exactly as the C-RADIO adapter does for the same
        # libraries, keeping the type surface at this one boundary.
        runtime: Any = torch
        model_cls: Any = Sam2VideoModel
        processor_cls: Any = Sam2VideoProcessor
        device = self._device or ("cuda" if runtime.cuda.is_available() else "cpu")
        if self._model is None or self._processor is None:
            self._processor = processor_cls.from_pretrained(self._model_id)
            self._model = (
                model_cls.from_pretrained(self._model_id, dtype=runtime.bfloat16).to(device).eval()
            )
            logger.info("SAM2 tracker loaded", extra={"model_id": self._model_id, "device": device})
        return runtime, self._model, self._processor

    async def track(
        self, frames: Sequence[NDArray[np.uint8]], seed: BoundingBox
    ) -> dict[int, BoundingBox]:
        if not frames:
            return {}
        return await asyncio.to_thread(self._track_sync, list(frames), seed)

    def _track_sync(
        self, frames: list[NDArray[np.uint8]], seed: BoundingBox
    ) -> dict[int, BoundingBox]:
        torch, model, processor = self._ensure_loaded()
        height, width = frames[0].shape[:2]
        anchor_idx = len(frames) // 2  # middle frame: object most likely presented
        box_px = [
            seed.x_min * width,
            seed.y_min * height,
            seed.x_max * width,
            seed.y_max * height,
        ]
        session = processor.init_video_session(
            video=frames,
            inference_device=model.device,
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        processor.add_inputs_to_inference_session(
            session,
            frame_idx=anchor_idx,
            obj_ids=1,
            input_boxes=[[box_px]],
            original_size=(height, width),
        )
        tracked: dict[int, BoundingBox] = {}
        with torch.inference_mode():
            # Forward from the anchor, then backward, so the whole presentation
            # is covered rather than only its second half.
            for reverse in (False, True):
                for out in model.propagate_in_video_iterator(
                    session, start_frame_idx=anchor_idx, reverse=reverse
                ):
                    mask = processor.post_process_masks(
                        [out.pred_masks], original_sizes=[[height, width]], binarize=True
                    )[0][0, 0]
                    ys, xs = np.nonzero(mask.cpu().numpy())
                    if xs.size:
                        tracked[out.frame_idx] = BoundingBox(
                            x_min=float(xs.min()) / width,
                            y_min=float(ys.min()) / height,
                            x_max=float(xs.max() + 1) / width,
                            y_max=float(ys.max() + 1) / height,
                        )
        return tracked


__all__ = ["DEFAULT_CENTRE_FRAC", "DEFAULT_SAM2_ID", "Sam2Tracker", "Tracker", "centre_box"]
