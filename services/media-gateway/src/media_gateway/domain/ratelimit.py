"""A small fixed-window rate limiter.

Token minting is the most abuse-sensitive surface the gateway exposes, and
docs/07-Privacy-and-Security.md asks for rate limits on internal APIs on the
grounds that a trusted LAN is not an authentication mechanism.

In-process and per-client only. One gateway process owns the media plane, so
there is nothing to share state with.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class FixedWindowLimiter:
    """Allow at most `limit` events per client per window."""

    def __init__(
        self,
        *,
        limit: int,
        window_s: float = 60.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.limit = limit
        self.window_s = window_s
        self._now = now
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(self, client: str) -> bool:
        """Record an attempt, returning whether it is within the limit."""
        current = self._now()
        started, count = self._windows.get(client, (current, 0))

        if current - started >= self.window_s:
            started, count = current, 0

        if count >= self.limit:
            self._windows[client] = (started, count)
            return False

        self._windows[client] = (started, count + 1)
        return True

    def retry_after_s(self, client: str) -> float:
        """Seconds until the client's window resets."""
        started, _ = self._windows.get(client, (self._now(), 0))
        return max(0.0, self.window_s - (self._now() - started))

    def forget(self, client: str) -> None:
        self._windows.pop(client, None)


__all__ = ["FixedWindowLimiter"]
