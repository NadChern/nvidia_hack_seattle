"""Owns the room workers and the LiveKit reachability probe.

Readiness reports whether this process can serve traffic, which for the LiveKit
path means the control plane is reachable. It deliberately does not depend on a
publisher being connected: with a room per session there is no room until
someone starts one, so gating on a publisher would report an idle gateway as
broken and deadlock a `depends_on: service_healthy` graph.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from urllib.parse import urlparse

from media_gateway.config import Settings
from media_gateway.domain.session import Session
from media_gateway.transport.room_worker import RoomWorker
from media_gateway.transport.source import MediaSink

logger = logging.getLogger(__name__)

DEFAULT_PORTS = {"ws": 80, "wss": 443, "http": 80, "https": 443}


def livekit_endpoint(url: str) -> tuple[str, int]:
    """Return the host and port to probe for a LiveKit URL."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or DEFAULT_PORTS.get(parsed.scheme, 7880)
    return host, port


class SessionSupervisor:
    """Joins a room per session and keeps the reachability probe fresh."""

    def __init__(
        self,
        *,
        settings: Settings,
        sink: MediaSink,
        on_participant_left: Callable[[str, str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._sink = sink
        self._on_participant_left = on_participant_left
        self._workers: dict[str, RoomWorker] = {}
        self._probe_task: asyncio.Task[None] | None = None
        self._reachable: bool | None = None

    # --- Readiness -------------------------------------------------------

    def readiness(self) -> str | None:
        """None when LiveKit is reachable, else a short reason."""
        if self._settings.media_source != "livekit":
            return None
        if self._reachable is None:
            return "probe pending"
        return None if self._reachable else "livekit unreachable"

    async def _probe_once(self) -> bool:
        host, port = livekit_endpoint(self._settings.livekit_url)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._settings.livekit_connect_timeout_s,
            )
        except (OSError, TimeoutError):
            return False
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True

    async def _probe_loop(self) -> None:
        while True:
            reachable = await self._probe_once()
            if reachable != self._reachable:
                logger.info(
                    "livekit reachability changed",
                    extra={"reachable": reachable, "livekit_url": self._settings.livekit_url},
                )
            self._reachable = reachable
            await asyncio.sleep(self._settings.livekit_probe_interval_s)

    # --- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._settings.media_source != "livekit":
            return
        self._probe_task = asyncio.create_task(self._probe_loop(), name="livekit-probe")

    async def stop(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._probe_task
            self._probe_task = None

        for session_id in list(self._workers):
            await self.leave(session_id)

    # --- Rooms -----------------------------------------------------------

    async def join(self, session: Session) -> RoomWorker:
        """Join the room for a session, replacing any existing worker."""
        await self.leave(session.session_id)

        worker = RoomWorker(
            settings=self._settings,
            session=session,
            sink=self._sink,
            on_participant_left=self._on_participant_left,
        )
        await worker.start()
        self._workers[session.session_id] = worker
        return worker

    async def leave(self, session_id: str) -> None:
        worker = self._workers.pop(session_id, None)
        if worker is not None:
            await worker.stop("session_ended")

    def worker_for(self, session_id: str) -> RoomWorker | None:
        return self._workers.get(session_id)

    def __len__(self) -> int:
        return len(self._workers)


__all__ = ["DEFAULT_PORTS", "SessionSupervisor", "livekit_endpoint"]
