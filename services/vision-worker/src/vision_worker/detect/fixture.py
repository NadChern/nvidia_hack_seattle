"""A detector that replays a scripted sequence instead of running a model.

The `ci` profile and anyone developing on a machine with no compatible GPU run
the *entire* pipeline -- relay consumer, tracker, stability machine, evidence,
verifier, emission -- against this. It never inspects `frame_rgb`; the script
supplies whatever `Detection`s a caller wants on each call, in order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Detection

from vision_worker.identity.base import SegmentedDetection


class FixtureDetector:
    """Replays `script`, one entry per `detect()` call.

    `loop=True` (the default) restarts from the beginning once exhausted,
    matching `ScriptedMediaSource`'s "paced and looping" precedent in the
    Media Gateway: a fixture pipeline left running should keep producing
    rather than going silent after one pass.
    """

    def __init__(self, script: Sequence[Sequence[Detection]], *, loop: bool = True) -> None:
        if not script:
            raise ValueError("a fixture detector needs at least one scripted frame")
        self._script = [tuple(frame) for frame in script]
        self._loop = loop
        self._cursor = 0
        self._call_count = 0
        self._last_detections: tuple[Detection, ...] = ()

    async def initialize(self) -> None:
        return None

    def readiness_payload(self) -> Mapping[str, object]:
        return {
            "detector": "fixture",
            "script_length": len(self._script),
            "calls": self._call_count,
            "loop": self._loop,
        }

    async def detect(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[Detection]:
        del frame_rgb  # the fixture detector never looks at the frame
        if self._cursor >= len(self._script):
            if not self._loop:
                return ()
            self._cursor = 0

        scripted = self._script[self._cursor]
        self._cursor += 1
        self._call_count += 1

        if not labels:
            self._last_detections = scripted
        else:
            wanted = set(labels)
            self._last_detections = tuple(
                detection for detection in scripted if detection.label in wanted
            )
        return self._last_detections

    async def segment(
        self, frame_rgb: NDArray[np.uint8], *, labels: Sequence[str]
    ) -> Sequence[SegmentedDetection]:
        wanted = set(labels)
        height, width = frame_rgb.shape[:2]
        segmented: list[SegmentedDetection] = []
        for detection in self._last_detections:
            if wanted and detection.label not in wanted:
                continue
            box = detection.box
            x1 = max(0, min(width, round(box.x_min * width)))
            x2 = max(0, min(width, round(box.x_max * width)))
            y1 = max(0, min(height, round(box.y_min * height)))
            y2 = max(0, min(height, round(box.y_max * height)))
            mask = np.zeros((height, width), dtype=np.bool_)
            mask[y1:y2, x1:x2] = True
            segmented.append(SegmentedDetection(detection=detection, mask=mask))
        return tuple(segmented)

    async def aclose(self) -> None:
        return None


__all__ = ["FixtureDetector"]
