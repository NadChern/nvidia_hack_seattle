"""Readiness tracking.

Readiness answers "can this process serve traffic", not "is a device
connected". With a room per session there is no LiveKit room until someone
starts a session, so making readiness depend on a publisher would deadlock a
`depends_on: service_healthy` graph and would report an idle gateway as broken.

Checks are registered as they come into existence, so this stays honest about
what is actually verified rather than asserting health it cannot observe.
"""

from __future__ import annotations

from collections.abc import Callable

#: A check returns None when healthy, or a short reason when not.
ReadinessCheck = Callable[[], str | None]


class Readiness:
    """A named set of readiness checks plus the shutdown flag."""

    def __init__(self) -> None:
        self._checks: dict[str, ReadinessCheck] = {}
        self._shutting_down = False

    def register(self, name: str, check: ReadinessCheck) -> None:
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        self._checks.pop(name, None)

    def begin_shutdown(self) -> None:
        """Fail readiness immediately so traffic drains before teardown."""
        self._shutting_down = True

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def evaluate(self) -> str | None:
        """Return None when ready, else the first failing reason."""
        if self._shutting_down:
            return "shutting_down"
        for name, check in self._checks.items():
            reason = check()
            if reason is not None:
                return f"{name}: {reason}"
        return None


__all__ = ["Readiness", "ReadinessCheck"]
