"""Consumes the media relay's video stream and drives the vision pipeline.

Wraps `MediaClient`, which already implements the reconnect and epoch
semantics this module only needs to react to: per its own docstring, "a
reconnect is at least as strong a reset as an epoch change... a consumer that
dropped mid-epoch resets rather than resuming against stale tracker state."

This module's only job is to translate relay messages into the two things the
rest of the pipeline needs: a reset on every `epoch_started`, and every
`VideoFrame` handed to a sink with `device_id` resolved -- `VideoFrame` and
`EpochStarted` both carry `session_id` but neither carries `device_id`, so
resolving it here, once, is what keeps that join out of every downstream
consumer.

**Reading is separated from processing, and stale frames are dropped.**

Without that split, a detector slower than the frame rate does not merely run
late -- it falls progressively further behind, forever. `media_gateway.relay.
hub` drops on its *send* side, keeping one latest-frame slot per subscriber,
but once bytes are on the wire they sit in this process's receive buffer.
Iterating the client directly and awaiting the sink per message consumes that
buffer in arrival order, so a 20ms-per-frame shortfall accumulates into
seconds of lag that never drains.

Measured, before this existed: YOLOE at 720p took ~145ms against a 125ms
budget at 8fps, and the gap between a frame being relayed and its overlay
being emitted grew past seven seconds and kept climbing. The pipeline's own
`observed_fps` read exactly 8.0 throughout, because it measures the
`captured_at` stamps of frames it processes -- and it processed every one of
them, just later and later. The metric built to catch rate problems is blind
to this one.

So a reader task drains the socket as fast as messages arrive, and a
dispatcher processes them. Between the two sits a queue with the same policy
the gateway applies one hop upstream: **a video frame displaces an older
unprocessed frame, and control messages are never dropped or reordered.**
Dropping is right because a stale frame describes a moment that has already
passed; processing it late is strictly worse than not processing it, and
`docs/12` guarantees an `epoch_started` is never seen after a frame of the
epoch it starts.

Audio is deliberately not handled here. `services/speech` consumes the same
relay and must *not* drop, because losing audio corrupts a transcript
invisibly -- the reason `relay/hub.py` closes a slow audio subscriber rather
than dropping for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from visual_memory_media_contract.client import MediaClient
from visual_memory_media_contract.protocol import (
    EpochEnded,
    EpochStarted,
    LifecycleSignal,
    SessionEnded,
    SessionStarted,
    VideoFrame,
)

logger = logging.getLogger(__name__)

#: Whatever `MediaClient` yields. Deliberately loose: this queue does not care
#: what a message is beyond whether it may be dropped.
RelayMessage: TypeAlias = object

#: How many messages may queue before the oldest is discarded. Only one video
#: frame is ever pending (a newer one displaces it), so in practice this
#: bounds *control* backlog, which is rare -- session and epoch boundaries
#: plus keepalives, all of which the sink handles without touching a model.
#: Reaching this means the sink is wedged, not merely slow.
INBOX_CAPACITY = 64


class FrameSink(Protocol):
    """What the rest of the pipeline implements to receive relay events."""

    async def epoch_started(self, *, session_id: str, device_id: str, epoch_id: str) -> None:
        """Reset per-track state. Called before any frame of the new epoch.

        `epoch_id` is the LiveKit track SID (docs/06: "consumers MUST reset
        per-track state on this message"). `track_id` is only ever meaningful
        within one `(session_id, epoch_id)` -- see
        `vision_worker.domain.stability.TrackRegistry.reset`.
        """
        ...

    async def epoch_ended(self, *, session_id: str, epoch_id: str) -> None: ...

    async def video_frame(
        self, *, session_id: str, device_id: str, epoch_id: str, frame: VideoFrame
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _Item:
    message: RelayMessage
    #: Whether this may be displaced by a newer one of its kind.
    is_frame: bool


class _Inbox:
    """Messages read but not yet processed. Frames are latest-wins.

    A tagged deque rather than an `asyncio.Queue`, for the same reason
    `media_gateway.relay.hub` uses one: two invariants have to hold at once,
    and a size-one queue cannot express both. Frames must be droppable so a
    slow sink never accumulates lag; control messages must not be, because a
    lost `epoch_started` leaves tracker state unreset against a new epoch.
    """

    def __init__(self, capacity: int = INBOX_CAPACITY) -> None:
        self._items: deque[_Item] = deque()
        self._capacity = capacity
        self._frames_queued = 0
        self._wakeup = asyncio.Event()
        self._closed = False
        #: Frames superseded before anything looked at them. Reported at
        #: `/v1/status`: this is the number that says the detector cannot keep
        #: up with the stream, and it is the one that was missing while the
        #: lag grew unnoticed.
        self.frames_dropped = 0
        self.control_dropped = 0

    def offer(self, message: RelayMessage, *, is_frame: bool) -> None:
        """Queue a message. Never blocks, never raises."""
        if self._closed:
            return
        if is_frame and self._frames_queued:
            self._evict_pending_frame()
        elif len(self._items) >= self._capacity:
            # Unreachable while the sink is merely slow: only one frame is
            # ever pending, and control messages are handled without touching
            # a model. Getting here means the sink is stuck, and continuing to
            # grow would trade a visible failure for an invisible one.
            self._drop_oldest()
        self._items.append(_Item(message, is_frame))
        if is_frame:
            self._frames_queued += 1
        self._wakeup.set()

    def _evict_pending_frame(self) -> None:
        """Remove the newest queued frame, leaving control messages in place.

        Scanning from the back finds the frame the incoming one supersedes;
        everything else keeps its relative order, so a frame is never
        reordered across an epoch boundary.
        """
        for index in range(len(self._items) - 1, -1, -1):
            if self._items[index].is_frame:
                del self._items[index]
                self._frames_queued -= 1
                self.frames_dropped += 1
                return

    def _drop_oldest(self) -> None:
        item = self._items.popleft()
        if item.is_frame:
            self._frames_queued -= 1
            self.frames_dropped += 1
        else:
            self.control_dropped += 1
            logger.error(
                "relay inbox full; dropped a control message -- the pipeline is not consuming",
                extra={"control_dropped": self.control_dropped},
            )

    def close(self) -> None:
        self._closed = True
        self._wakeup.set()

    async def next(self) -> RelayMessage | None:
        """The next message, or None once closed and drained."""
        while True:
            if self._items:
                item = self._items.popleft()
                if item.is_frame:
                    self._frames_queued -= 1
                return item.message
            if self._closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()


class RelayConsumer:
    """Drives `sink` from one relay video stream."""

    def __init__(self, client: MediaClient, sink: FrameSink) -> None:
        self._client = client
        self._sink = sink
        self._device_by_session: dict[str, str] = {}
        self._inbox = _Inbox()

    @property
    def frames_dropped(self) -> int:
        """Frames superseded before being processed -- the detector falling
        behind the stream, made visible."""
        return self._inbox.frames_dropped

    @property
    def control_dropped(self) -> int:
        return self._inbox.control_dropped

    async def run(self) -> None:
        """Read and dispatch until the client is closed or ends for good.

        `MediaClient` already reconnects on a transient drop; each reconnect
        re-delivers `session_started` and a fresh `epoch_started` for every
        still-active epoch, so no state needs preserving across one.
        """
        reader = asyncio.create_task(self._read(), name="relay-reader")
        try:
            await self._dispatch()
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        # Surface a reader failure rather than reporting a clean exit: main's
        # readiness check treats an unexpected exit as not-ready, and it can
        # only do that if the exception reaches it.
        if not reader.cancelled():
            failure = reader.exception()
            if failure is not None:
                raise failure

    async def _read(self) -> None:
        """Drain the socket as fast as messages arrive. Never awaits the sink."""
        try:
            async for message in self._client:
                self._inbox.offer(message, is_frame=isinstance(message, VideoFrame))
        finally:
            # Let the dispatcher finish what is queued and then stop, rather
            # than leaving it waiting on a socket nobody is reading.
            self._inbox.close()

    async def _dispatch(self) -> None:
        while True:
            message = await self._inbox.next()
            if message is None:
                return
            await self._handle(message)

    async def _handle(self, message: RelayMessage) -> None:
        if isinstance(message, SessionStarted):
            self._device_by_session[message.session_id] = message.device_id
        elif isinstance(message, SessionEnded):
            self._device_by_session.pop(message.session_id, None)
        elif isinstance(message, EpochStarted):
            await self._sink.epoch_started(
                session_id=message.session_id,
                device_id=self._device_for(message.session_id),
                epoch_id=message.epoch_id,
            )
        elif isinstance(message, EpochEnded):
            await self._sink.epoch_ended(session_id=message.session_id, epoch_id=message.epoch_id)
        elif isinstance(message, VideoFrame):
            await self._sink.video_frame(
                session_id=message.session_id,
                device_id=self._device_for(message.session_id),
                epoch_id=message.epoch_id,
                frame=message,
            )
        elif isinstance(message, LifecycleSignal):
            # The gateway's in-band copy of what it already posts to
            # Memory over HTTP. Vision has no use for it beyond
            # debugging -- Memory is the component that fans a lifecycle
            # signal out to affected objects.
            logger.debug(
                "relay lifecycle signal",
                extra={
                    "action": message.envelope.signal.action,
                    "session_id": message.envelope.session_id,
                },
            )
        # StreamHello and Keepalive carry nothing this pipeline acts on.

    def _device_for(self, session_id: str) -> str:
        """Fall back to the session_id itself if `session_started` was missed
        -- e.g. this consumer connected mid-session after a restart. A frame
        must still be attributable to *something*; silently dropping real
        perception data because a lookup came up empty would be worse than an
        imprecise `device_id`.
        """
        return self._device_by_session.get(session_id, session_id)


__all__ = ["FrameSink", "RelayConsumer"]
