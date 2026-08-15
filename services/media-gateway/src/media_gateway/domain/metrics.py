"""Counters and latency sampling.

docs/03-Hackathon-Stack.md makes queue depth, dropped frames, verifier
outcomes, and latency release metrics rather than debugging aids, so they are
recorded here and surfaced by the status endpoint.

Latency is kept in a bounded reservoir: an unbounded list would grow without
limit over a long session, and the target hardware has a unified-memory budget
to respect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class LatencyReservoir:
    """Recent latency samples, bounded, with percentiles."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._samples: deque[float] = deque(maxlen=capacity)
        self.count = 0

    def observe(self, seconds: float) -> None:
        self._samples.append(seconds)
        self.count += 1

    def percentile(self, fraction: float) -> float | None:
        """Nearest-rank percentile of the retained samples."""
        if not self._samples:
            return None
        ordered = sorted(self._samples)
        index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "retained": len(self._samples),
            "p50_ms": _ms(self.percentile(0.50)),
            "p95_ms": _ms(self.percentile(0.95)),
            "max_ms": _ms(max(self._samples) if self._samples else None),
        }


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 3)


@dataclass
class StreamMetrics:
    """Per-stream-kind counters."""

    received: int = 0
    admitted: int = 0
    rejected_dimensions: int = 0
    dropped_before_sampling: int = 0
    relayed: int = 0
    #: Audio only. Non-zero here means a subscriber was closed rather than
    #: silently losing audio, which would corrupt transcription invisibly.
    subscribers_closed_for_backpressure: int = 0
    relay_latency: LatencyReservoir = field(default_factory=LatencyReservoir)

    def snapshot(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "admitted": self.admitted,
            "rejected_dimensions": self.rejected_dimensions,
            "dropped_before_sampling": self.dropped_before_sampling,
            "relayed": self.relayed,
            "subscribers_closed_for_backpressure": (self.subscribers_closed_for_backpressure),
            "relay_latency": self.relay_latency.snapshot(),
        }


@dataclass
class MetricsRegistry:
    """Everything the status endpoint reports."""

    video: StreamMetrics = field(default_factory=StreamMetrics)
    audio: StreamMetrics = field(default_factory=StreamMetrics)
    sessions_created: int = 0
    sessions_ended: int = 0
    sessions_expired: int = 0
    epochs_started: int = 0
    epochs_ended: int = 0
    tokens_issued: int = 0
    lifecycle_signals_emitted: int = 0

    def for_stream(self, stream_kind: str) -> StreamMetrics:
        return self.video if stream_kind == "video" else self.audio

    def snapshot(self) -> dict[str, Any]:
        return {
            "sessions": {
                "created": self.sessions_created,
                "ended": self.sessions_ended,
                "expired": self.sessions_expired,
            },
            "epochs": {"started": self.epochs_started, "ended": self.epochs_ended},
            "tokens_issued": self.tokens_issued,
            "lifecycle_signals_emitted": self.lifecycle_signals_emitted,
            "video": self.video.snapshot(),
            "audio": self.audio.snapshot(),
        }


__all__ = ["LatencyReservoir", "MetricsRegistry", "StreamMetrics"]
