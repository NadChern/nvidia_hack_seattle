"""Detects gaps in audio continuity from `pts_samples` arithmetic.

`pts_samples` is the cumulative sample count since the epoch began (see
`docs/12-Media-Relay-Contract.md`, "Audio"). If two consecutive chunks are
contiguous, the later one's `pts_samples` equals the earlier one's
`pts_samples + samples`. A larger value means samples were lost between them.
Message count and `sequence` are not reliable signals for this -- both stay
perfectly contiguous across a gap, which is the whole reason this file, and
the `audio_session_basic` fixture it is tested against, exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_memory_media_contract.protocol import AudioChunk


@dataclass(frozen=True, slots=True)
class ContinuityGap:
    """A detected discontinuity between two consecutive audio chunks."""

    before: AudioChunk
    after: AudioChunk
    lost_samples: int

    @property
    def lost_seconds(self) -> float:
        return self.lost_samples / self.after.sample_rate


def gap_between(previous: AudioChunk, current: AudioChunk) -> ContinuityGap | None:
    """Return the gap between two consecutive chunks, or `None` if contiguous."""
    expected_pts = previous.pts_samples + previous.samples
    lost = current.pts_samples - expected_pts
    if lost > 0:
        return ContinuityGap(before=previous, after=current, lost_samples=lost)
    return None


class ContinuityTracker:
    """One-chunk-at-a-time continuity check for a single epoch's chunk stream.

    Feed chunks in arrival order through `check`. Call `reset` on every
    `epoch_started` -- comparing a chunk against the last chunk of a
    *previous* epoch is meaningless, since `pts_samples` restarts at zero
    each epoch.
    """

    def __init__(self) -> None:
        self._previous: AudioChunk | None = None

    def reset(self) -> None:
        self._previous = None

    def check(self, chunk: AudioChunk) -> ContinuityGap | None:
        gap = gap_between(self._previous, chunk) if self._previous is not None else None
        self._previous = chunk
        return gap


__all__ = ["ContinuityGap", "ContinuityTracker", "gap_between"]
