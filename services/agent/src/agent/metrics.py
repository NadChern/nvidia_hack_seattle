"""Small in-process counters for guard and hands-free observability."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agent.models import GuardVerdict


@dataclass(slots=True)
class AgentMetrics:
    queries: int = 0
    guard_passed: int = 0
    guard_vetoed: dict[str, int] = field(default_factory=dict[str, int])
    hands_free_ignored: int = 0
    hands_free_triggered: int = 0
    hands_free_replies: int = 0
    hands_free_errors: int = 0
    registrations_started: int = 0
    registrations_succeeded: int = 0
    registrations_failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_guard(self, verdict: GuardVerdict) -> None:
        if verdict.startswith("registration:"):
            return
        with self._lock:
            self.queries += 1
            if verdict == "passed":
                self.guard_passed += 1
            else:
                self.guard_vetoed[verdict] = self.guard_vetoed.get(verdict, 0) + 1

    def record_hands_free_ignored(self) -> None:
        with self._lock:
            self.hands_free_ignored += 1

    def record_hands_free_triggered(self) -> None:
        with self._lock:
            self.hands_free_triggered += 1

    def record_hands_free_reply(self) -> None:
        with self._lock:
            self.hands_free_replies += 1

    def record_hands_free_error(self) -> None:
        with self._lock:
            self.hands_free_errors += 1

    def record_registration_started(self) -> None:
        with self._lock:
            self.registrations_started += 1

    def record_registration_succeeded(self) -> None:
        with self._lock:
            self.registrations_succeeded += 1

    def record_registration_failed(self) -> None:
        with self._lock:
            self.registrations_failed += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "queries": self.queries,
                "guard_passed": self.guard_passed,
                "guard_vetoed": dict(sorted(self.guard_vetoed.items())),
                "hands_free_ignored": self.hands_free_ignored,
                "hands_free_triggered": self.hands_free_triggered,
                "hands_free_replies": self.hands_free_replies,
                "hands_free_errors": self.hands_free_errors,
                "registrations_started": self.registrations_started,
                "registrations_succeeded": self.registrations_succeeded,
                "registrations_failed": self.registrations_failed,
            }


__all__ = ["AgentMetrics"]
