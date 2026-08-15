"""Posting lifecycle signals to the Memory Service.

**Nothing here may block the media path.** `emit` is synchronous, never awaits,
and never raises: it drops the envelope into a bounded queue and returns. A
background worker does the HTTP. If Memory is slow, down, or missing entirely,
frames keep flowing and audio keeps flowing — a memory service outage must not
become a media outage.

The queue is bounded and drops the *oldest* entry when full. Blocking would
stall the room worker; dropping the newest would discard the most recent state
change, which is the one that matters. Either way the drop is counted and
logged, because silently losing a signal means an object stays `in_transit`
forever with nothing to explain why.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import httpx
from visual_memory_media_contract.protocol import LifecycleEnvelope

from media_gateway.config import Settings

logger = logging.getLogger(__name__)

#: Enough to absorb a teardown burst -- every epoch in a session ending at
#: once -- without letting an unreachable Memory grow unboundedly.
QUEUE_DEPTH = 64


class MemorySink:
    """Fire-and-forget delivery of lifecycle envelopes."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = settings.lifecycle_sink_url
        self._queue: asyncio.Queue[LifecycleEnvelope] = asyncio.Queue(maxsize=QUEUE_DEPTH)
        self._worker: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self.delivered = 0
        self.dropped = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        """False until a sink URL is configured, which is the default.

        The gateway is useful with no Memory Service at all, so this stays off
        rather than filling logs with connection errors nobody asked for.
        """
        return self._url is not None

    async def start(self) -> None:
        if not self.enabled:
            logger.info("lifecycle sink disabled; set VMA_LIFECYCLE_SINK_URL to enable")
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.lifecycle_sink_timeout_s),
            headers=self._auth_headers(),
        )
        self._worker = asyncio.create_task(self._drain(), name="lifecycle-sink")
        logger.info("lifecycle sink enabled", extra={"sink": self._url})

    def _auth_headers(self) -> dict[str, str]:
        token = self._settings.internal_api_token
        if token is None:
            return {}
        # The token is a SecretStr; it reaches the header and never a log line.
        return {"authorization": f"Bearer {token.get_secret_value()}"}

    def emit(self, envelope: LifecycleEnvelope) -> None:
        """Queue an envelope. Never blocks, never raises."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(envelope)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
            logger.warning(
                "lifecycle sink queue full; dropped the oldest signal",
                extra={"dropped": self.dropped},
            )
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(envelope)

    async def _drain(self) -> None:
        assert self._client is not None
        while True:
            envelope = await self._queue.get()
            try:
                await self._post(envelope)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.failed += 1
                # Logged and abandoned rather than retried forever. The
                # idempotency key makes a retry safe, but an unbounded retry
                # loop against a dead Memory would outlive the session it
                # describes.
                logger.warning(
                    "could not deliver a lifecycle signal",
                    extra={"action": envelope.signal.action, "failed": self.failed},
                )
            finally:
                self._queue.task_done()

    async def _post(self, envelope: LifecycleEnvelope) -> None:
        assert self._client is not None and self._url is not None
        response = await self._client.post(self._url, json=envelope.model_dump(mode="json"))
        response.raise_for_status()
        self.delivered += 1
        logger.info(
            "delivered a lifecycle signal",
            extra={
                "action": envelope.signal.action,
                "reason": envelope.signal.reason,
                "media_epoch_id": envelope.scope.media_epoch_id,
            },
        )

    async def stop(self) -> None:
        """Give queued signals a brief chance to land, then stop.

        A session ending is the moment the last signals are produced, so
        cancelling immediately would discard exactly the ones that matter.
        """
        if self._worker is None:
            return
        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
            await asyncio.wait_for(
                self._queue.join(), timeout=self._settings.lifecycle_sink_timeout_s
            )
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        if self._client is not None:
            await self._client.aclose()

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "failed": self.failed,
            "queued": self._queue.qsize(),
        }


async def register_session(settings: Settings, *, session_id: str, device_id: str) -> str:
    """Register a session with Memory and adopt the id it returns.

    docs/06 splits minting from owning: the gateway is the only component
    present when a session starts, and Memory owns what the session *was*.
    Adopting the returned id is what lets Memory start minting its own later
    with no code change on either side.

    A registry that is configured but unreachable is **not** fatal. Refusing
    the session would make a Memory outage stop the glasses from connecting,
    which trades a recoverable problem for an unrecoverable one.
    """
    if settings.session_registry_url is None:
        return session_id

    headers: dict[str, str] = {}
    if settings.internal_api_token is not None:
        headers["authorization"] = f"Bearer {settings.internal_api_token.get_secret_value()}"

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.lifecycle_sink_timeout_s), headers=headers
        ) as client:
            response = await client.post(
                settings.session_registry_url,
                json={"session_id": session_id, "device_id": device_id},
            )
            response.raise_for_status()
            adopted = str(response.json().get("session_id") or session_id)
    except Exception:
        logger.warning(
            "could not register the session with memory; using the local id",
            extra={"session_id": session_id},
        )
        return session_id

    if adopted != session_id:
        logger.info(
            "adopted the session id returned by memory",
            extra={"session_id": adopted, "local_session_id": session_id},
        )
    return adopted


__all__ = ["MemorySink", "register_session"]
