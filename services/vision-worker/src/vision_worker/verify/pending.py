"""Running verification off the frame loop.

**Nothing here may block frame handling.** `submit` is synchronous, never
awaits, and never raises: it drops a unit of work into a bounded queue and
returns. A background worker awaits the verifier.

Why this exists rather than awaiting the verifier inline. The relay hands
frames to `Pipeline.video_frame` one at a time, and the gateway's hub
(`media_gateway.relay.hub`) keeps a *single* latest-frame slot per subscriber:
a frame that arrives while the previous one is still being handled displaces
the unread one and increments `dropped_since_previous`. So a verifier that
takes twenty seconds does not slow the stream down -- it makes the gateway
discard roughly twenty seconds of frames, and the pipeline resumes with a hole
it cannot see. `observed_fps` then measures the wrong thing, `frames_since_seen`
stops meaning what `StabilityConfig` built it to mean, and unrelated tracks trip
`vanished` because they went unseen for a gap nobody was in frame for.

That is the failure this module prevents: an inline verifier does not merely
make the pipeline slow, it corrupts the state machine that produced the
candidate being verified.

The queue is bounded and drops the *oldest* entry when full, the same policy
and for the same reason as the gateway's `MemorySink`: blocking would stall the
caller, and dropping the newest would discard the most recent thing that
happened, which is the one that matters. A drop here is more serious than a
dropped lifecycle signal -- it means a real pickup is never recorded -- so it is
counted, logged, and reported at `/v1/status` rather than swallowed.

Queued work holds its evidence frames, which is why the depth is small: a
window is tens of JPEGs, so a depth of 8 is a few tens of megabytes at worst,
bounded and flat. Sizing it larger would trade an unbounded memory footprint
for the ability to work through a backlog whose candidates are already stale.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

#: One unit of deferred work. Deliberately opaque: this module schedules, it
#: does not know what verification is.
Work = Callable[[], Awaitable[None]]


class PendingVerifications:
    """A bounded queue of deferred work, run by a fixed pool of workers.

    Workers start on the first `submit`, so a `Pipeline` that never proposes a
    candidate -- every test that only checks frame handling, for one -- never
    creates a task it would then have to clean up.
    """

    def __init__(self, *, depth: int, concurrency: int) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        self._queue: asyncio.Queue[Work] = asyncio.Queue(maxsize=depth)
        self._concurrency = concurrency
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False
        #: Work accepted but not yet finished -- queued plus in flight. The
        #: queue's own `qsize` misses the in-flight ones, which are exactly the
        #: slow ones worth seeing at `/v1/status`.
        self._in_flight = 0
        self.completed = 0
        self.dropped = 0
        self.failed = 0

    @property
    def pending(self) -> int:
        """Accepted and not yet finished, including work currently running."""
        return self._queue.qsize() + self._in_flight

    def submit(self, work: Work) -> None:
        """Queue a unit of work. Never blocks, never raises."""
        if self._closed:
            # Shutting down. Accepting work nothing will ever run would make
            # `drain` wait for a worker that is already gone.
            self.dropped += 1
            return

        self._ensure_workers()
        try:
            self._queue.put_nowait(work)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                # The displaced item never runs, so its `join()` accounting has
                # to be settled here or `drain` would wait on it forever.
                self._queue.task_done()
            self.dropped += 1
            logger.warning(
                "verification queue full; dropped the oldest candidate -- "
                "verification is not keeping up with the stream",
                extra={"dropped": self.dropped, "depth": self._queue.maxsize},
            )
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(work)

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._run(), name=f"verification-worker-{index}")
            for index in range(self._concurrency)
        ]

    async def _run(self) -> None:
        while True:
            work = await self._queue.get()
            self._in_flight += 1
            try:
                await work()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.failed += 1
                # Logged and abandoned rather than retried. A retry would need
                # the evidence window held even longer, and a verifier failing
                # once on a window will fail on it again.
                logger.exception("verification failed", extra={"failed": self.failed})
            else:
                self.completed += 1
            finally:
                self._in_flight -= 1
                self._queue.task_done()

    async def drain(self) -> None:
        """Wait until every accepted unit of work has finished.

        The sync point a batch run needs: `scripts/replay_clip.py` reaches the
        end of a clip long before the last verification answers, and printing a
        summary at that moment would report a pipeline that had not finished
        thinking. A live service calls this on shutdown for the same reason.
        """
        if not self._workers:
            return
        await self._queue.join()

    async def aclose(self) -> None:
        """Finish what was accepted, then stop the workers."""
        self._closed = True
        await self.drain()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()


__all__ = ["PendingVerifications", "Work"]
