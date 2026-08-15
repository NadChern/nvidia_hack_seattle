"""Session registry.

A session is one authorized device's media run. The registry is bounded and
swept by TTL so an abandoned session cannot hold a room or a slot forever --
the gateway may only ever hold a couple of concurrent sessions on the target
hardware.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from media_gateway.config import Settings
from media_gateway.domain.ids import new_session_id
from media_gateway.errors import CapacityError, ForbiddenError, NotFoundError


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class Session:
    """One device's media session."""

    session_id: str
    device_id: str
    room: str
    created_at: dt.datetime
    last_seen_at: dt.datetime
    ended_at: dt.datetime | None = None
    #: True while a publisher is actually connected. Reported in status, and
    #: deliberately not part of readiness.
    publisher_present: bool = False
    #: True once a publisher has *ever* joined. Distinct from
    #: `publisher_present`, which drops back to False on any disconnect: a
    #: wearer who briefly loses the link must not be mistaken for a session
    #: nobody ever used. See `SessionRegistry.sweep`.
    ever_published: bool = False
    metadata: dict[str, str] = field(default_factory=dict[str, str])

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def touch(self, *, at: dt.datetime | None = None) -> None:
        self.last_seen_at = at or _utcnow()

    def is_expired(self, ttl_s: int, *, now: dt.datetime | None = None) -> bool:
        reference = now or _utcnow()
        return (reference - self.last_seen_at).total_seconds() > ttl_s


class SessionRegistry:
    """Bounded, TTL-swept store of active sessions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}

    def create(
        self,
        *,
        device_id: str,
        session_id: str | None = None,
        at: dt.datetime | None = None,
    ) -> Session:
        """Create a session for an authorized device.

        Sweeps expired sessions first, so a crashed publisher does not
        permanently consume the concurrency budget.
        """
        self._require_allowed(device_id)
        now = at or _utcnow()
        self.sweep(now=now)

        if len(self._sessions) >= self._settings.max_concurrent_sessions:
            raise CapacityError(
                "no session slots available",
                active=len(self._sessions),
                limit=self._settings.max_concurrent_sessions,
            )

        resolved = session_id or new_session_id()
        session = Session(
            session_id=resolved,
            device_id=device_id,
            room=f"{self._settings.room_prefix}-{resolved}",
            created_at=now,
            last_seen_at=now,
        )
        self._sessions[resolved] = session
        return session

    def _require_allowed(self, device_id: str) -> None:
        allowlist = self._settings.device_id_allowlist
        if allowlist and device_id not in allowlist:
            # Do not echo the allowlist back; that would leak which device ids
            # exist to an unauthenticated caller.
            raise ForbiddenError("device is not authorized", device_id=device_id)

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError("unknown session", session_id=session_id)
        return session

    def find(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def end(self, session_id: str, *, at: dt.datetime | None = None) -> Session:
        session = self.get(session_id)
        session.ended_at = at or _utcnow()
        session.publisher_present = False
        self._sessions.pop(session_id, None)
        return session

    def sweep(self, *, now: dt.datetime | None = None) -> list[Session]:
        """Remove sessions idle for longer than the TTL.

        Sessions nobody ever joined expire on a much shorter clock. A device
        that cannot reach LiveKit asks for a *new* session on each retry, and
        with the default budget of two, two failed joins lock the gateway out
        for the full hour of `session_ttl_s` -- the wearer then sees only
        `429 capacity_exhausted` and no amount of retrying recovers. Observed
        repeatedly on the X3 Pro while the ICE path was broken.

        `ever_published` rather than `publisher_present`, so a wearer whose
        link drops keeps the long TTL and their slot.
        """
        reference = now or _utcnow()
        unclaimed_ttl = self._settings.unclaimed_session_ttl_s
        expired = [
            session
            for session in self._sessions.values()
            if session.is_expired(self._settings.session_ttl_s, now=reference)
            or (not session.ever_published and session.is_expired(unclaimed_ttl, now=reference))
        ]
        for session in expired:
            session.ended_at = reference
            session.publisher_present = False
            self._sessions.pop(session.session_id, None)
        return expired

    def active(self) -> list[Session]:
        return list(self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)


__all__ = ["Session", "SessionRegistry"]
