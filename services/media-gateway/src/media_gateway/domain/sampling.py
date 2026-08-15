"""Bounded sampling and the dimension guard.

This is the S01 spike's proven core, extracted from the LiveKit consumer so it
can be tested without a server. Three cooperating pieces:

- `LatestSlot` holds at most one frame and never blocks the producer, so slow
  inference downstream cannot apply backpressure to media ingest.
- `DimensionGuard` rejects frames whose size is not the expected one. The spike
  observed transient 8x8 frames during simulcast adaptation and track teardown
  making up roughly a quarter of all frames; forwarding one to a detector is a
  silent correctness bug.
- `Pacer` drives the consumer on absolute deadlines so sampling does not drift.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from media_gateway.config import DimensionGuardMode

T = TypeVar("T")

logger = logging.getLogger(__name__)


class LatestSlot(Generic[T]):
    """A one-item slot with latest-wins semantics.

    A plain slot rather than an `asyncio.Queue(maxsize=1)`: the producer must
    never block and the consumer is time-driven rather than event-driven, so
    queue blocking semantics would be misleading. Within one event loop no
    locking is required.
    """

    __slots__ = ("_item", "_occupied", "dropped", "offered", "taken")

    def __init__(self) -> None:
        self._item: T | None = None
        self._occupied = False
        self.offered = 0
        self.taken = 0
        #: Frames displaced before anyone sampled them. Reported downstream as
        #: `dropped_since_previous` so consumers can measure their own gaps.
        self.dropped = 0

    def offer(self, item: T) -> None:
        """Store an item, displacing any unread one. Never blocks."""
        self.offered += 1
        if self._occupied:
            self.dropped += 1
        self._item = item
        self._occupied = True

    def take(self) -> T | None:
        """Remove and return the pending item, or None when empty."""
        if not self._occupied:
            return None
        item = self._item
        self._item = None
        self._occupied = False
        self.taken += 1
        return item

    def clear(self) -> None:
        """Discard any pending item without counting it as taken."""
        self._item = None
        self._occupied = False

    @property
    def occupied(self) -> bool:
        return self._occupied


#: How many consecutive frames of one size make it the size in force, under
#: `sustained`. At 30fps of arriving track frames this is a little under a
#: second -- long enough to outlast a rung of an encoder's ramp-up, short
#: enough that a real stream starts flowing promptly.
SUSTAINED_RUN = 24


@dataclass
class DimensionGuard:
    """Admit only frames of the expected size, counting everything seen.

    `strict` compares against the configured size, matching the spike.

    `sustained` latches the first size that *persists*, and re-latches when a
    different one does. This is what a real publisher needs.

    `first_frame_wins` latches whatever the first frame happens to be. It is
    retained because it is what the spike specified, but **it is the wrong
    choice for anything that ramps**, which in practice means anything
    publishing through LiveKit: an encoder starts small and climbs to the
    negotiated resolution over tens of seconds, so the first frame is the
    bottom rung of that climb rather than the size the stream will settle at.
    Measured against a browser publishing 720p, it latched 320x180 and then
    rejected 5002 of 5581 frames -- for the whole session, silently, with a
    consumer downstream seeing a couple of percent of the video and finding
    nothing in it.
    """

    mode: DimensionGuardMode
    expected_width: int
    expected_height: int
    #: Every size seen, tallied *before* the admission decision, so the
    #: histogram is a complete record of what the track actually produced.
    histogram: dict[str, int] = field(default_factory=dict[str, int])
    admitted: int = 0
    rejected: int = 0
    #: How many times `sustained` changed its mind. Non-zero at the start of a
    #: stream is the encoder ramping and expected; climbing later means the
    #: publisher keeps changing resolution, which is worth knowing.
    relatched: int = 0
    _latched: tuple[int, int] | None = field(default=None, repr=False)
    _run_size: tuple[int, int] | None = field(default=None, repr=False)
    _run_length: int = field(default=0, repr=False)

    def admit(self, width: int, height: int) -> bool:
        """Record a frame's size and report whether it may be sampled."""
        key = f"{width}x{height}"
        self.histogram[key] = self.histogram.get(key, 0) + 1

        if self.mode == "sustained":
            self._observe_run(width, height)

        expected = self._expected(width, height)
        if (width, height) != expected:
            self.rejected += 1
            return False
        self.admitted += 1
        return True

    def _observe_run(self, width: int, height: int) -> None:
        """Track how long the current size has held, and latch once it sticks.

        Latching on a *run* rather than a single frame is the whole point: the
        first frame of a ramping encoder is not representative of anything,
        while a size that has held for a second is what the stream is actually
        doing.
        """
        size = (width, height)
        if size == self._run_size:
            self._run_length += 1
        else:
            self._run_size = size
            self._run_length = 1

        if self._run_length < SUSTAINED_RUN or size == self._latched:
            return

        # Re-latching mid-epoch is deliberate. Consumers tolerate it:
        # `ImageMotionPose` resets on a shape change and `evidence/clip.py`
        # falls back to a still when a window spans two sizes. Refusing to
        # re-latch would instead reject every frame for the rest of the epoch,
        # which is the failure this mode exists to prevent.
        previous = self._latched
        self._latched = size
        if previous is not None:
            self.relatched += 1
            logger.info(
                "dimension guard re-latched to a new sustained size",
                extra={
                    "previous": f"{previous[0]}x{previous[1]}",
                    "latched": f"{width}x{height}",
                    "relatched": self.relatched,
                },
            )

    def _expected(self, width: int, height: int) -> tuple[int, int]:
        if self.mode == "strict":
            return (self.expected_width, self.expected_height)
        if self.mode == "sustained":
            # Nothing has held long enough yet. Admitting the current frame
            # would be `first_frame_wins` by another name, so the ramp is
            # rejected -- counted, and visible in the histogram.
            return self._latched if self._latched is not None else (-1, -1)
        if self._latched is None:
            self._latched = (width, height)
        return self._latched

    def reset(self) -> None:
        """Forget per-epoch state. Called when a new epoch starts.

        The latched size must not survive an epoch, because a rejoin may bring
        a different camera resolution. Counters reset too, since they are
        reported per epoch.
        """
        self.histogram = {}
        self.admitted = 0
        self.rejected = 0
        self.relatched = 0
        self._latched = None
        self._run_size = None
        self._run_length = 0

    @property
    def latched(self) -> tuple[int, int] | None:
        """The size in force, once known."""
        if self.mode == "strict":
            return (self.expected_width, self.expected_height)
        return self._latched


class Pacer:
    """Paces a loop on absolute deadlines.

    Deadlines are absolute rather than `sleep(interval)` so scheduling delay
    does not accumulate: at 2 FPS over a thirty-minute session, relative sleeps
    drift by seconds.

    The clock and sleep function are injectable so pacing is testable without
    real time passing.
    """

    def __init__(
        self,
        interval_s: float,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = interval_s
        self._now = now
        self._sleep = sleep or _asyncio_sleep
        self._next_deadline: float | None = None
        #: Ticks skipped because the loop fell more than one interval behind.
        self.missed = 0

    async def wait(self) -> None:
        """Sleep until the next deadline."""
        current = self._now()
        if self._next_deadline is None:
            self._next_deadline = current + self.interval_s

        delay = self._next_deadline - current
        if delay > 0:
            await self._sleep(delay)
            self._next_deadline += self.interval_s
            return

        # Behind schedule. Skip the deadlines already missed rather than
        # bursting to catch up, which would defeat the sampling rate.
        behind = int(-delay // self.interval_s) + 1
        self.missed += behind
        self._next_deadline += self.interval_s * behind
        await self._sleep(0)

    def reset(self) -> None:
        self._next_deadline = None


async def _asyncio_sleep(delay: float) -> None:
    import asyncio

    await asyncio.sleep(delay)


__all__ = ["DimensionGuard", "LatestSlot", "Pacer"]
