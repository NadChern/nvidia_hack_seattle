"""A deterministic `WindowReasoner` for tests and GPU-free runs.

The pipeline is wired entirely through the `WindowReasoner` Protocol, so a test
-- or a laptop with no Cosmos to talk to -- drives the whole
window -> identity -> memory path by scripting exactly what each window
"sees", with no model, no HTTP, and no torch.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from visual_memory_vision_contract.protocol import BoundingBox, DetectorRef

from vision_worker.reason.base import LocalizedFrame, WindowEvent

_CENTER_BOX = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

_FIXTURE_REF = DetectorRef(
    name="cosmos-fixture", checkpoint="fixture", revision="reason-fixture-v1"
)


class FixtureReasoner:
    """Returns scripted `WindowEvent`s, one list per `analyze` call.

    Pass `script` to return a different result for each successive window (the
    queue is consumed left to right); once it is empty, `default` is returned
    for every further call. Every call's `(frame_count, labels)` is recorded on
    `calls` so a test can assert the pipeline actually invoked the reasoner.
    """

    def __init__(
        self,
        *,
        script: Sequence[Sequence[WindowEvent]] = (),
        default: Sequence[WindowEvent] = (),
        ref: DetectorRef = _FIXTURE_REF,
        localize_box: BoundingBox | None = _CENTER_BOX,
        reference_valid: bool = True,
    ) -> None:
        self._script: deque[Sequence[WindowEvent]] = deque(script)
        self._default = tuple(default)
        self._ref = ref
        self._localize_box = localize_box
        self._reference_valid = reference_valid
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    @property
    def ref(self) -> DetectorRef:
        return self._ref

    async def analyze(
        self, frames: Sequence[bytes], *, labels: Sequence[str]
    ) -> Sequence[WindowEvent]:
        self.calls.append((len(frames), tuple(labels)))
        if self._script:
            return tuple(self._script.popleft())
        return self._default

    async def localize_sequence(
        self, frames: Sequence[bytes], label: str
    ) -> Sequence[LocalizedFrame]:
        if self._localize_box is None:
            return ()
        return tuple(
            LocalizedFrame(index=index, box=self._localize_box) for index in range(len(frames))
        )

    async def localize(self, frame: bytes, label: str) -> BoundingBox | None:
        return self._localize_box

    async def validate_reference(self, crop: bytes, label: str) -> bool:
        return self._reference_valid


__all__ = ["FixtureReasoner"]
