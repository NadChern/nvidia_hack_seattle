"""The window-reasoner boundary: one VLM look at a short video window.

This replaces the old per-frame `detect -> track -> stability -> verify` chain
with a single question asked of a vision-language model over a few consecutive
frames: *what personal objects are in the last frame, where are they, and what
just happened to them.* The model both localizes (a box) and classifies the
event, so there is no separate detector, tracker, or stability machine behind
it -- see the package docstring and the plan.

Everything here is pure Python and torch-free: a `WindowReasoner` talks to a
model over HTTP (`cosmos.py`) or returns a script (`fixture.py`), so
`pipeline.py` and `tests/test_domain_isolation.py` stay free of any model
runtime, exactly as they were with the verifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateAction,
    DetectorRef,
)

#: What the reasoner can say happened to an object. The first four are memory
#: vocabulary and become an `Observation.event.action` unchanged. The last two
#: are how the model declines: `nothing_happened` (it looked and no event
#: occurred -- the answer that keeps a false pickup out of memory) and
#: `unknown` (it could not tell). The pipeline drops both before any write, so
#: they are represented but never promoted.
ReasonAction = CandidateAction | str


@dataclass(frozen=True, slots=True)
class WindowEvent:
    """One object the reasoner found in a window's final frame.

    `box` is in the **last frame's** normalized coordinates (top-left origin,
    0..1), because that is the frame the pipeline decodes and crops for the
    C-RADIOv4 identity check. `action` describes what happened to it across the
    window; `location_description` is the model's own words for where it is,
    which flows straight into `Location.surface` in memory.
    """

    label: str
    box: BoundingBox
    action: ReasonAction
    location_description: str | None
    confidence: float

    @property
    def is_memory_event(self) -> bool:
        """Whether this is a real event worth a memory write.

        `nothing_happened` and `unknown` are honest non-answers, not events --
        the pipeline logs and drops them, the same discipline the old verifier
        applied to `vlm_saw_no_event`.
        """
        return self.action in ("placed", "picked_up", "carried")


class WindowReasoner(Protocol):
    """Looks at a window of frames and reports the personal objects in it."""

    @property
    def ref(self) -> DetectorRef:
        """What produced these events, precisely enough to record on an
        `Observation`'s provenance -- mirrors the old `DetectorRef`."""
        ...

    async def analyze(
        self, frames: Sequence[bytes], *, labels: Sequence[str]
    ) -> Sequence[WindowEvent]:
        """Return one `WindowEvent` per object of interest visible in the final
        frame. `frames` are JPEG bytes in capture order; `labels` is the set of
        registered object labels to look for (nothing else is reported)."""
        ...


class Localizer(Protocol):
    """Finds one named object's box in one frame -- registration's front end.

    Enrollment localizes the object being rotated the same way the pipeline
    localizes it at query time (both via the reasoner), so an enrolled crop is
    framed like the crops it will later be matched against -- the crop-parity
    the identity cosine depends on.
    """

    async def localize(self, frame: bytes, label: str) -> BoundingBox | None: ...


__all__ = ["Localizer", "ReasonAction", "WindowEvent", "WindowReasoner"]
