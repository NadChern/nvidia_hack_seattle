"""In-memory activity ring for a human watching ingestion happen.

Not the canonical record -- the `audit` table is -- and not durable across a
restart. Backs `GET /v1/events`, the publisher dev page's live log, matching
`vision_worker.pipeline`'s identical `PipelineEvent`/`_RECENT_EVENTS_MAXLEN`
pattern on the producer side.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass

#: Bounded so a long-running dev session's memory footprint stays flat.
_RECENT_EVENTS_MAXLEN = 200


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """One observation's ingestion outcome."""

    at: dt.datetime
    label: str
    action: str
    object_id: str | None
    promoted: bool
    current_status: str | None


class ActivityLog:
    """One per process, held on `app.state.activity`."""

    def __init__(self) -> None:
        self._events: deque[MemoryEvent] = deque(maxlen=_RECENT_EVENTS_MAXLEN)

    def record(self, event: MemoryEvent) -> None:
        self._events.append(event)

    @property
    def recent(self) -> tuple[MemoryEvent, ...]:
        """Oldest first."""
        return tuple(self._events)


__all__ = ["ActivityLog", "MemoryEvent"]
