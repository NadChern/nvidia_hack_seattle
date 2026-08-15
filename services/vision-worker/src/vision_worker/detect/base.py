"""What every object detector implements.

Two adapters exist behind this interface: `fixture.py` replays a scripted
detection sequence -- the `ci` profile and the no-GPU-laptop path -- and
`yoloe.py` (task #38) runs a real model. Nothing above this interface -- the
tracker, the stability machine, the verifier -- can tell which one is
running, which is what makes the `ci` and `dev-macos` profiles in the plan's
Context real rather than aspirational.

Per `docs/08-Development-and-Deployment.md`:30, only `detect/`, `depth/`,
`track/`, and `pose/` may import a model runtime; everything that consumes a
`Detector` must not need to. `tests/test_domain_isolation.py` extends that
rule to this whole service, not just `domain/`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Detection


class Detector(Protocol):
    """A stateful detector: `initialize()` once, then `detect()` per frame.

    Matches the warm-start lifecycle the prior cooking_assist project's
    `vision/detector.py:YoloeDetector` uses: a blocking model load happens
    once, off the event loop, and `detect()` stays cheap enough to call every
    frame afterward.
    """

    async def initialize(self) -> None:
        """Load weights and warm up. Called once before the first `detect()`."""
        ...

    def readiness_payload(self) -> Mapping[str, object]:
        """Reported at `/v1/status` -- load time, warmup time, request count,
        average inference latency, or whatever else this detector wants
        observable."""
        ...

    async def detect(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[Detection]:
        """Detect `labels` in one frame.

        `labels` empty means open-vocabulary / prompt-free: return whatever
        the detector finds. No track association happens here -- that is
        `track/`'s job, run on this method's output.
        """
        ...

    async def aclose(self) -> None:
        """Release model resources. A no-op for adapters that hold none."""
        ...


__all__ = ["Detector"]
