"""Remote-assist request registry: one arm/accept/end cycle per session.

Shaped like `ManualTriggerRegistry` on purpose -- in-memory, an injected clock,
no globals -- but a request carries more state than a bare arm/consume flag
because it is listed by an operator before it is accepted, and its outcome
(accepted vs. expired) has to survive that listing.

Keyed by `session_id`, not `request_id`: only one remote-assist call can be in
flight for a given wearer at a time, the same way only one manual trigger can.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from media_gateway.domain.ids import new_assist_request_id
from media_gateway.errors import CapacityError

#: Matches the HUD's `assist` device event one-for-one (docs/12) -- the
#: request registry and the event pushed to the wearer's glasses must agree on
#: the same three words, or the two sides of the contract silently drift.
AssistState = Literal["requested", "accepted", "ended"]


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class AssistRequest:
    request_id: str
    session_id: str
    device_id: str
    state: AssistState
    requested_at: dt.datetime
    expires_at: dt.datetime

    def is_expired(self, *, now: dt.datetime) -> bool:
        return now >= self.expires_at


class AssistRequestRegistry:
    """Tracks at most one live remote-assist request per session."""

    def __init__(
        self,
        *,
        ttl_s: int,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._ttl_s = ttl_s
        self._now = now
        self._requests: dict[str, AssistRequest] = {}

    def request(self, *, session_id: str, device_id: str) -> AssistRequest:
        """Raise a request, or return the one already ringing.

        Idempotent while a request is still `requested`: the wearer pressing
        the button twice before anyone answers must not mint a second request
        with a different id and a reset expiry, which would make "the"
        request ambiguous for whoever is about to accept it.
        """
        now = self._now()
        existing = self._requests.get(session_id)
        if (
            existing is not None
            and existing.state == "requested"
            and not existing.is_expired(now=now)
        ):
            return existing

        created = AssistRequest(
            request_id=new_assist_request_id(),
            session_id=session_id,
            device_id=device_id,
            state="requested",
            requested_at=now,
            expires_at=now + dt.timedelta(seconds=self._ttl_s),
        )
        self._requests[session_id] = created
        return created

    def pending(self) -> list[AssistRequest]:
        """Every currently-requested (not yet accepted, not expired) call."""
        now = self._now()
        self._sweep(now=now)
        return [r for r in self._requests.values() if r.state == "requested"]

    def accept(self, session_id: str) -> AssistRequest | None:
        """Consume-once: `None` if there is nothing left to accept."""
        now = self._now()
        request = self._requests.get(session_id)
        if request is None or request.state != "requested" or request.is_expired(now=now):
            return None
        request.state = "accepted"
        return request

    def end(self, session_id: str) -> AssistRequest | None:
        """Mark an accepted (or still-pending) call ended, and stop tracking it."""
        request = self._requests.pop(session_id, None)
        if request is None:
            return None
        request.state = "ended"
        return request

    def is_active(self, session_id: str) -> bool:
        """True while a call is accepted -- what the Agent must stay quiet for."""
        request = self._requests.get(session_id)
        return request is not None and request.state == "accepted"

    def clear(self, session_id: str) -> None:
        self._requests.pop(session_id, None)

    def _sweep(self, *, now: dt.datetime) -> None:
        expired = [
            session_id
            for session_id, request in self._requests.items()
            if request.state == "requested" and request.is_expired(now=now)
        ]
        for session_id in expired:
            del self._requests[session_id]


#: Pushed on `WS /v1/assist/events`. `keepalive` is not part of this union --
#: like the device-events channel, it is written directly by the WS pump so an
#: idle connection is distinguishable from a dead one without touching the hub.
class AssistRequestedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: Literal["assist_requested"] = "assist_requested"
    request_id: str
    session_id: str
    occurred_at: dt.datetime


class AssistAcceptedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: Literal["assist_accepted"] = "assist_accepted"
    request_id: str
    session_id: str
    occurred_at: dt.datetime


class AssistEndedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    type: Literal["assist_ended"] = "assist_ended"
    request_id: str
    session_id: str
    occurred_at: dt.datetime


AssistNotification = Annotated[
    AssistRequestedEvent | AssistAcceptedEvent | AssistEndedEvent,
    Field(discriminator="type"),
]


@dataclass(eq=False)
class AssistEventSubscriber:
    queue_size: int
    queue: asyncio.Queue[AssistNotification] = field(init=False)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    close_reason: str | None = None

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.queue_size)

    def push(self, event: AssistNotification) -> None:
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


class AssistEventHub:
    """Fans out assist notifications to every connected operator.

    Deliberately global rather than session-scoped, matching the frozen
    contract: an unpaired helper does not know which session it is joining
    until it has already seen the request that names it. Scoping this to a
    specific device is an open item, not settled here -- see
    `role-prompts/Jacky-Remote-Assist.md`.
    """

    def __init__(self, *, queue_size: int, max_subscribers: int) -> None:
        self._queue_size = queue_size
        self._max_subscribers = max_subscribers
        self._subscribers: set[AssistEventSubscriber] = set()

    def subscribe(self) -> AssistEventSubscriber:
        if len(self._subscribers) >= self._max_subscribers:
            raise CapacityError("too many assist event subscribers", limit=self._max_subscribers)
        subscriber = AssistEventSubscriber(queue_size=self._queue_size)
        self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: AssistEventSubscriber) -> None:
        self._subscribers.discard(subscriber)

    def publish(self, event: AssistNotification) -> int:
        delivered = 0
        for subscriber in tuple(self._subscribers):
            subscriber.push(event)
            if not subscriber.closed.is_set():
                delivered += 1
        return delivered

    def close_all(self, reason: str = "gateway_shutdown") -> None:
        for subscriber in tuple(self._subscribers):
            subscriber.close(reason)

    def __len__(self) -> int:
        return len(self._subscribers)


__all__ = [
    "AssistAcceptedEvent",
    "AssistEndedEvent",
    "AssistEventHub",
    "AssistEventSubscriber",
    "AssistNotification",
    "AssistRequest",
    "AssistRequestRegistry",
    "AssistRequestedEvent",
    "AssistState",
]
