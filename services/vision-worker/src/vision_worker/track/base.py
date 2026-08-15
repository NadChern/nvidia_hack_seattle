"""What every tracker implements: per-frame detections in, a `track_id`
assigned to each one out.

Identity assignment only. A tracker decides whether this frame's "keys" box
is the same physical object as last frame's -- it says nothing about whether
that object is at rest or moving, which is `domain/stability.py`'s job on the
`TrackSample`s this pipeline builds from the tracker's output.

Two adapters exist: `greedy_iou.py`, pure numpy and the default, and
`botsort.py` (a later upgrade), which trades a torch/ultralytics dependency
for global motion compensation -- a Kalman filter tuned for a head-worn
camera, where a stationary object can jump hundreds of pixels in-frame
because the head moved. Nothing above this interface can tell which is
running, matching `detect/base.py`'s reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from visual_memory_vision_contract.protocol import Detection


class Tracker(Protocol):
    """Stateful across frames within one media epoch. `reset()` on every
    `epoch_started` -- `track_id` is only ever meaningful within one
    `(session_id, media_epoch_id)`; carrying identity across a reconnect is
    the exact trap `docs/06-Data-Contract.md`:110 warns about.
    """

    def reset(self) -> None: ...

    def update(self, detections: Sequence[Detection]) -> Sequence[tuple[str, Detection]]:
        """Assign a `track_id` to each of this frame's detections.

        Returns one `(track_id, detection)` pair per input detection, in the
        same order they were given. A detection that cannot be matched to a
        prior frame mints a fresh id; this is the only place a new `track_id`
        is ever created.
        """
        ...


__all__ = ["Tracker"]
