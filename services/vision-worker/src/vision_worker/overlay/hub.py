"""Fan-out of per-frame detections to viewers.

**Nothing here may block frame handling.** `publish` is synchronous, never
awaits, and never raises. This is the same rule `verify/pending.py` enforces for
verification, and it exists for the same measured reason: the gateway's hub
(`media_gateway.relay.hub`) keeps a single latest-frame slot per subscriber, so
a pipeline that blocks does not slow the stream down -- it makes the gateway
discard frames the stability machine never learns were missing.

A browser tab on a slow laptop is exactly the kind of consumer that would
otherwise cause that, which is why a viewer can never apply backpressure here.

Each subscriber holds one latest-wins slot. An overlay is a snapshot of what a
frame looked like, so a stale one has no value once a newer one exists --
dropping it is strictly better than delaying the new one. That makes this
simpler than `relay/hub.py`, which additionally has ordered control messages
that must never be dropped; overlays are entirely droppable, and there is no
`epoch_started` equivalent to protect.

Carrying no pixels is what makes the whole approach viable: a viewer publishing
its own camera already has the frames, so this sends coordinates. Kilobytes per
second against megabits, and no second encode on a machine that is already
running a detector.
"""

from __future__ import annotations

import asyncio
import logging

from visual_memory_vision_contract.protocol import OverlayFrame

logger = logging.getLogger(__name__)


class OverlaySubscriber:
    """One connected viewer. Holds at most one pending overlay."""

    def __init__(self, *, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.sent = 0
        self.dropped = 0
        self._pending: OverlayFrame | None = None
        self._closed = False
        self._wakeup = asyncio.Event()

    # --- Producer side ---------------------------------------------------

    def offer(self, frame: OverlayFrame) -> None:
        """Queue an overlay, displacing any unread one. Never blocks."""
        if self._closed:
            return
        if self._pending is not None:
            self.dropped += 1
        self._pending = frame
        self.sent += 1
        self._wakeup.set()

    def close(self) -> None:
        """Stop the subscriber and wake its send loop."""
        self._closed = True
        self._wakeup.set()

    # --- Consumer side ---------------------------------------------------

    async def next(self) -> OverlayFrame | None:
        """Await the next overlay, or None once closed and drained."""
        while True:
            if self._pending is not None:
                frame, self._pending = self._pending, None
                return frame
            if self._closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()


class OverlayHub:
    """Tracks viewers and fans overlays out to them."""

    def __init__(self, *, max_subscribers: int) -> None:
        self._max_subscribers = max_subscribers
        self._subscribers: set[OverlaySubscriber] = set()
        #: Overlays discarded because no viewer had read the previous one.
        #: Unlike a dropped candidate this costs nothing real -- it is a frame
        #: a browser would not have drawn anyway -- but a number that climbs
        #: fast says the viewer, not the pipeline, is the slow part.
        self.dropped = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def has_subscribers(self) -> bool:
        """Whether building an overlay is worth the work at all.

        The pipeline checks this before assembling a frame's tracks, so a
        deployment with no viewer attached -- which is every deployment, most of
        the time -- pays nothing.
        """
        return bool(self._subscribers)

    def subscribe(self, *, session_id: str | None = None) -> OverlaySubscriber | None:
        """Register a viewer, optionally scoped to one Gateway session, or None
        when the limit is already reached.

        `None` rather than an exception: refusing an extra debugging viewer is
        an ordinary outcome the endpoint reports with a close code, not an
        error condition anything needs to unwind from.
        """
        if len(self._subscribers) >= self._max_subscribers:
            return None
        subscriber = OverlaySubscriber(session_id=session_id)
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: OverlaySubscriber) -> None:
        self._subscribers.discard(subscriber)
        subscriber.close()

    def publish(self, frame: OverlayFrame) -> None:
        """Offer an overlay to viewers of this session. Never blocks or raises.

        An unscoped subscriber remains available for diagnostics, but the Admin
        Console always supplies the selected Gateway session. That prevents a
        second publisher or a stale previous selection from putting boxes over
        unrelated video.
        """
        for subscriber in self._subscribers:
            if subscriber.session_id is not None and subscriber.session_id != frame.session_id:
                continue
            before = subscriber.dropped
            subscriber.offer(frame)
            self.dropped += subscriber.dropped - before

    def close(self) -> None:
        """Close every viewer, for shutdown."""
        for subscriber in tuple(self._subscribers):
            subscriber.close()
        self._subscribers.clear()


__all__ = ["OverlayHub", "OverlaySubscriber"]
