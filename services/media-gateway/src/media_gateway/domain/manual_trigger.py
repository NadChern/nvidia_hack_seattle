"""Short-lived, consume-once manual question arms."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class ManualTriggerRegistry:
    def __init__(
        self,
        *,
        ttl_s: int,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._ttl_s = ttl_s
        self._now = now
        self._armed: dict[str, dt.datetime] = {}

    def arm(self, session_id: str) -> dt.datetime:
        expires_at = self._now() + dt.timedelta(seconds=self._ttl_s)
        self._armed[session_id] = expires_at
        return expires_at

    def consume(self, session_id: str) -> bool:
        expires_at = self._armed.pop(session_id, None)
        return expires_at is not None and expires_at > self._now()

    def clear(self, session_id: str) -> None:
        self._armed.pop(session_id, None)


__all__ = ["ManualTriggerRegistry"]
