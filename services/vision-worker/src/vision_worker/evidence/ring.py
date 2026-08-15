"""A bounded, time-scoped ring of already-sampled frames.

Never raw media -- the Media Gateway's `raw_buffer_seconds = 0` commitment is
about what *it* retains, not what it relays. This ring holds only frames the
gateway already sampled, dimension-guarded, and relayed, and only for
`max_duration`, an explicit retention value reported at `/v1/status` per
docs/07 rather than a constant buried in code.

One ring per session/epoch, shared across every object in view -- not one
per track. A candidate's `EvidenceWindow` selects a temporal slice out of it
by timestamp, which is what lets `evidence/clip.py` and `Verifier.verify()`
both work from "whatever was on camera during this window" rather than a
per-object recording.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """One already-sampled frame, held only long enough to become evidence."""

    captured_at: dt.datetime
    payload: bytes
    width: int
    height: int


class EvidenceRing:
    """Evicts anything older than `max_duration` relative to the newest
    frame's own `captured_at` -- driven by the video's timestamps, not wall
    clock, so eviction is deterministic and testable without a real clock.
    """

    def __init__(self, max_duration: dt.timedelta) -> None:
        self._max_duration = max_duration
        self._frames: deque[BufferedFrame] = deque()

    @property
    def max_duration(self) -> dt.timedelta:
        return self._max_duration

    def push(self, frame: BufferedFrame) -> None:
        self._frames.append(frame)
        self._evict_stale(relative_to=frame.captured_at)

    def reset(self) -> None:
        """Drop everything. Call on `epoch_started` -- a new epoch's frames
        must never be evidenced by the previous epoch's buffered bytes."""
        self._frames.clear()

    def window(
        self, *, started_at: dt.datetime, ended_at: dt.datetime
    ) -> tuple[BufferedFrame, ...]:
        """Every currently buffered frame captured within `[started_at,
        ended_at]`, oldest first. Frames older than `max_duration` are
        already gone by the time this is called -- a window reaching further
        back than the ring's retention returns only what survived."""
        return tuple(frame for frame in self._frames if started_at <= frame.captured_at <= ended_at)

    def __len__(self) -> int:
        return len(self._frames)

    def _evict_stale(self, *, relative_to: dt.datetime) -> None:
        cutoff = relative_to - self._max_duration
        while self._frames and self._frames[0].captured_at < cutoff:
            self._frames.popleft()


__all__ = ["BufferedFrame", "EvidenceRing"]
