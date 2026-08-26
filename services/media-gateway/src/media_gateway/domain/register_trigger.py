"""Short-lived, consume-once register-button arms.

The register button is the grounder-free, speech-free enrollment trigger: the
wearer focuses "Register" on the HUD and taps, the glasses arm this, and the
agent consumes it and drives a center-anchor capture. It mirrors
:class:`~media_gateway.domain.manual_trigger.ManualTriggerRegistry` exactly, but
carries an optional label -- the button may name the item (a HUD picker) or
leave it for the agent to allocate a placeholder.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class RegisterArm:
    """A consumed register press: its optional label (``None`` = placeholder)."""

    label: str | None


class RegisterTriggerRegistry:
    def __init__(
        self,
        *,
        ttl_s: int,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._ttl_s = ttl_s
        self._now = now
        self._armed: dict[str, tuple[dt.datetime, str | None]] = {}

    def arm(self, session_id: str, label: str | None = None) -> dt.datetime:
        expires_at = self._now() + dt.timedelta(seconds=self._ttl_s)
        self._armed[session_id] = (expires_at, label)
        return expires_at

    def consume(self, session_id: str) -> RegisterArm | None:
        entry = self._armed.pop(session_id, None)
        if entry is None:
            return None
        expires_at, label = entry
        if expires_at <= self._now():
            return None
        return RegisterArm(label=label)

    def clear(self, session_id: str) -> None:
        self._armed.pop(session_id, None)


__all__ = ["RegisterArm", "RegisterTriggerRegistry"]
