"""Typed HUD events and bounded per-session fan-out."""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from media_gateway.errors import CapacityError


class TranscriptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: Literal["transcript"] = "transcript"
    text: str = Field(min_length=1, max_length=2_000)
    epoch_id: str
    pts_samples_start: int = Field(ge=0)
    samples: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    occurred_at: dt.datetime


class ReplyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: Literal["reply"] = "reply"
    question: str = Field(min_length=1, max_length=2_000)
    reply: str = Field(min_length=1, max_length=2_000)
    answer_status: Literal["confirmed", "last_confirmed_only", "unknown", "ambiguous_object"] | None
    object_id: str | None
    guard: Literal[
        "passed",
        "vetoed:1",
        "vetoed:2",
        "vetoed:3",
        "vetoed:4",
        "vetoed:5",
        "vetoed:6",
    ]
    latency_ms: int = Field(ge=0)
    occurred_at: dt.datetime


DeviceEvent = Annotated[TranscriptEvent | ReplyEvent, Field(discriminator="type")]


@dataclass(eq=False)
class DeviceEventSubscriber:
    session_id: str
    queue_size: int
    queue: asyncio.Queue[TranscriptEvent | ReplyEvent] = field(init=False)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    close_reason: str | None = None

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.queue_size)

    def push(self, event: TranscriptEvent | ReplyEvent) -> None:
        if self.closed.is_set():
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.close_reason = "event_backpressure"
            self.closed.set()

    def close(self, reason: str) -> None:
        self.close_reason = reason
        self.closed.set()


class DeviceEventHub:
    """Fan out immutable text events without blocking the Agent's POST."""

    def __init__(self, *, queue_size: int, max_subscribers: int) -> None:
        self._queue_size = queue_size
        self._max_subscribers = max_subscribers
        self._subscribers: set[DeviceEventSubscriber] = set()

    def subscribe(self, session_id: str) -> DeviceEventSubscriber:
        if len(self._subscribers) >= self._max_subscribers:
            raise CapacityError("too many device event subscribers", limit=self._max_subscribers)
        subscriber = DeviceEventSubscriber(session_id=session_id, queue_size=self._queue_size)
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: DeviceEventSubscriber) -> None:
        self._subscribers.discard(subscriber)

    def publish(self, session_id: str, event: TranscriptEvent | ReplyEvent) -> int:
        delivered = 0
        for subscriber in tuple(self._subscribers):
            if subscriber.session_id != session_id:
                continue
            subscriber.push(event)
            if not subscriber.closed.is_set():
                delivered += 1
        return delivered

    def close_session(self, session_id: str, reason: str = "session_ended") -> None:
        for subscriber in tuple(self._subscribers):
            if subscriber.session_id == session_id:
                subscriber.close(reason)

    def close_all(self, reason: str = "gateway_shutdown") -> None:
        for subscriber in tuple(self._subscribers):
            subscriber.close(reason)

    def __len__(self) -> int:
        return len(self._subscribers)


__all__ = [
    "DeviceEvent",
    "DeviceEventHub",
    "DeviceEventSubscriber",
    "ReplyEvent",
    "TranscriptEvent",
]
