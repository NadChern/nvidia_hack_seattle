"""Fan-out to local relay subscribers.

Two invariants have to hold at once, which is why this is a tagged deque rather
than an `asyncio.Queue`:

- Video frames are latest-wins. A slow consumer loses stale frames and sees a
  rising `dropped_since_previous`; it never stalls media ingest.
- Control messages are never dropped or reordered. `docs/12` guarantees that an
  `epoch_started` cannot be processed after a frame belonging to the epoch it
  starts, so a consumer can safely reset tracker state on it.

A single size-one queue cannot do both: the second control message would
displace the first. Eviction therefore targets the queued *frame* specifically
and leaves control messages alone.

Audio is never dropped at all. A subscriber that cannot keep up has its socket
closed, because silently losing audio corrupts transcription invisibly while a
closed socket is a failure someone notices.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Final

from visual_memory_media_contract.protocol import StreamKind, VideoEncoding

from media_gateway.errors import CapacityError

logger = logging.getLogger(__name__)

#: Close reason sent when an audio subscriber cannot keep up.
AUDIO_BACKPRESSURE: Final = "audio_backpressure"

#: How many control messages may queue behind a stalled video consumer before
#: it is treated as gone. Control is rare -- epoch and session boundaries plus
#: keepalives -- so this is generous.
CONTROL_BACKLOG: Final = 64


@dataclass(frozen=True, slots=True)
class _Item:
    payload: bytes
    is_frame: bool


class Subscriber:
    """One connected consumer of one stream."""

    def __init__(
        self,
        *,
        stream_kind: StreamKind,
        encoding: VideoEncoding | None,
        capacity: int,
    ) -> None:
        self.stream_kind: StreamKind = stream_kind
        self.encoding: VideoEncoding | None = encoding
        self.capacity = capacity
        self.sent = 0
        self.dropped = 0
        self.close_reason: str | None = None
        self._items: deque[_Item] = deque()
        self._frames_queued = 0
        self._wakeup = asyncio.Event()

    # --- Producer side ---------------------------------------------------

    def offer_frame(self, payload: bytes) -> None:
        """Queue a frame, displacing any unread frame. Never blocks.

        Only the pending frame is displaced; queued control messages survive.
        """
        if self.close_reason is not None:
            return
        if self._frames_queued:
            self._evict_pending_frame()
        self._items.append(_Item(payload, is_frame=True))
        self._frames_queued += 1
        self.sent += 1
        self._wake()

    def offer_ordered(self, payload: bytes) -> bool:
        """Queue a message that must not be dropped.

        Returns False when the subscriber was closed because it is too far
        behind to accept more.
        """
        if self.close_reason is not None:
            return False
        if len(self._items) >= self.capacity:
            self.close(AUDIO_BACKPRESSURE)
            return False
        self._items.append(_Item(payload, is_frame=False))
        self.sent += 1
        self._wake()
        return True

    def _evict_pending_frame(self) -> None:
        for index in range(len(self._items) - 1, -1, -1):
            if self._items[index].is_frame:
                del self._items[index]
                self._frames_queued -= 1
                self.dropped += 1
                return

    def close(self, reason: str) -> None:
        """Mark the subscriber closed and wake its send loop.

        Queued messages are deliberately kept. A consumer must still receive
        the `epoch_ended` and `session_ended` that explain why the stream is
        stopping; discarding them turns a clean shutdown into an unexplained
        disconnect. `next` drains what is left and then reports the close.
        """
        if self.close_reason is not None:
            return
        self.close_reason = reason
        self._wake()

    def _wake(self) -> None:
        self._wakeup.set()

    # --- Consumer side ---------------------------------------------------

    async def next(self) -> bytes | None:
        """Await the next wire frame, or None once the subscriber is closed."""
        while True:
            if self._items:
                item = self._items.popleft()
                if item.is_frame:
                    self._frames_queued -= 1
                return item.payload
            if self.close_reason is not None:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()

    def drain_nowait(self) -> list[bytes]:
        """Take everything queued without awaiting. For tests and shutdown."""
        payloads = [item.payload for item in self._items]
        self._items.clear()
        self._frames_queued = 0
        return payloads

    @property
    def depth(self) -> int:
        return len(self._items)


class RelayHub:
    """Tracks subscribers and fans encoded frames out to them."""

    def __init__(self, *, max_subscribers: int, audio_queue_chunks: int) -> None:
        self._max_subscribers = max_subscribers
        self._audio_queue_chunks = audio_queue_chunks
        self._subscribers: set[Subscriber] = set()

    def subscribe(
        self,
        *,
        stream_kind: StreamKind,
        encoding: VideoEncoding | None = None,
    ) -> Subscriber:
        if len(self._subscribers) >= self._max_subscribers:
            raise CapacityError(
                "too many relay subscribers",
                active=len(self._subscribers),
                limit=self._max_subscribers,
            )
        subscriber = Subscriber(
            stream_kind=stream_kind,
            encoding=encoding,
            # Video holds one frame plus a control backlog; audio buffers a
            # couple of seconds of chunks.
            capacity=CONTROL_BACKLOG if stream_kind == "video" else self._audio_queue_chunks,
        )
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)

    def subscribers(self, stream_kind: StreamKind | None = None) -> list[Subscriber]:
        return [
            subscriber
            for subscriber in self._subscribers
            if stream_kind is None or subscriber.stream_kind == stream_kind
        ]

    def required_video_encodings(self) -> set[VideoEncoding]:
        """Encodings at least one subscriber wants, so we encode no more."""
        encodings: set[VideoEncoding] = set()
        for subscriber in self._subscribers:
            if subscriber.stream_kind == "video" and subscriber.encoding is not None:
                encodings.add(subscriber.encoding)
        return encodings

    def publish_control(self, frame: bytes, stream_kind: StreamKind) -> int:
        """Deliver a control message in order with that stream's media."""
        delivered = 0
        for subscriber in self.subscribers(stream_kind):
            if subscriber.offer_ordered(frame):
                delivered += 1
            else:
                logger.warning(
                    "relay subscriber closed while queueing a control message",
                    extra={"stream_kind": stream_kind, "reason": subscriber.close_reason},
                )
        return delivered

    def publish_video(self, frames: dict[VideoEncoding, bytes]) -> int:
        """Fan out a sampled frame, latest-wins per subscriber."""
        delivered = 0
        for subscriber in self.subscribers("video"):
            encoding = subscriber.encoding
            if encoding is None:  # pragma: no cover - always set for video
                continue
            frame = frames.get(encoding)
            if frame is None:  # pragma: no cover - encodings queried first
                continue
            subscriber.offer_frame(frame)
            delivered += 1
        return delivered

    def publish_audio(self, frame: bytes) -> int:
        """Fan out audio, closing any subscriber that cannot keep up."""
        delivered = 0
        for subscriber in self.subscribers("audio"):
            if subscriber.offer_ordered(frame):
                delivered += 1
            else:
                logger.warning(
                    "audio subscriber closed for backpressure",
                    extra={"queued": subscriber.depth, "sent": subscriber.sent},
                )
        return delivered

    def close_all(self, reason: str) -> None:
        for subscriber in list(self._subscribers):
            subscriber.close(reason)

    def __len__(self) -> int:
        return len(self._subscribers)


__all__ = [
    "AUDIO_BACKPRESSURE",
    "CONTROL_BACKLOG",
    "RelayHub",
    "Subscriber",
]
