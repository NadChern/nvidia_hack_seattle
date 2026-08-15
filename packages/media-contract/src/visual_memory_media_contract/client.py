"""Async client for the media relay.

Consumers iterate messages and never touch LiveKit:

    async for message in MediaClient("ws://localhost:8080/v1/stream/video"):
        match message:
            case EpochStarted():
                tracker.reset(message.epoch_id)
            case VideoFrame():
                tracker.step(message.rgb, message.captured_at)
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

import websockets
from websockets.asyncio.client import ClientConnection

from visual_memory_media_contract.framing import FramingError, decode_message
from visual_memory_media_contract.protocol import RelayMessage

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

#: A close code the client must not retry. The relay uses it for an
#: unauthorized or otherwise refused connection, and retrying would just repeat
#: the refusal while the consumer sees an empty stream and no explanation.
POLICY_VIOLATION = 1008


class MediaClientError(RuntimeError):
    """The client could not deliver a usable stream."""


class ReconnectPolicy:
    """Exponential backoff with jitter.

    Jitter matters because Vision and Speech both reconnect when the gateway
    restarts; without it they would retry in lockstep forever.
    """

    def __init__(
        self,
        *,
        initial_seconds: float = 0.5,
        max_seconds: float = 30.0,
        multiplier: float = 2.0,
        jitter: float = 0.25,
    ) -> None:
        self.initial_seconds = initial_seconds
        self.max_seconds = max_seconds
        self.multiplier = multiplier
        self.jitter = jitter

    def delay_for(self, attempt: int) -> float:
        """Return the delay before retry number `attempt` (1-based)."""
        raw = self.initial_seconds * (self.multiplier ** max(0, attempt - 1))
        capped = min(raw, self.max_seconds)
        spread = capped * self.jitter
        return max(0.0, capped + random.uniform(-spread, spread))  # noqa: S311


class MediaClient:
    """Reconnecting async iterator over one relay stream.

    A reconnect is at least as strong a reset as an epoch change: the gateway
    re-sends `stream_hello` and a fresh `epoch_started` for every still-active
    epoch, so a consumer that dropped mid-epoch resets rather than resuming
    against stale tracker state.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        reconnect: bool = True,
        policy: ReconnectPolicy | None = None,
        open_timeout: float = 10.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.url = url
        self.token = token
        self.reconnect = reconnect
        self.policy = policy or ReconnectPolicy()
        self.open_timeout = open_timeout
        self.max_message_bytes = max_message_bytes
        self._closed = False
        self._connection: ClientConnection | None = None

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop iterating and close the underlying socket."""
        self._closed = True
        connection = self._connection
        if connection is not None:
            await connection.close()

    async def __aiter__(self) -> AsyncIterator[RelayMessage]:
        attempt = 0
        while not self._closed:
            try:
                async for message in self._stream_once():
                    attempt = 0
                    yield message
            except (OSError, websockets.WebSocketException) as exc:
                if self._closed or not self.reconnect:
                    return
                attempt += 1
                delay = self.policy.delay_for(attempt)
                logger.warning(
                    "media relay disconnected, reconnecting",
                    extra={
                        "url": self.url,
                        "attempt": attempt,
                        "delay_seconds": round(delay, 3),
                        "error": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay)
                continue

            # The server closed cleanly.
            if not self.reconnect:
                return
            attempt += 1
            await asyncio.sleep(self.policy.delay_for(attempt))

    async def _stream_once(self) -> AsyncIterator[RelayMessage]:
        async with websockets.connect(
            self.url,
            additional_headers=self._headers,
            open_timeout=self.open_timeout,
            max_size=self.max_message_bytes,
        ) as connection:
            self._connection = connection
            try:
                async for frame in self._frames(connection):
                    if isinstance(frame, str):
                        raise MediaClientError(
                            "relay sent a text frame; the protocol is binary only"
                        )
                    try:
                        yield decode_message(frame)
                    except FramingError as exc:
                        # A malformed frame means the peer is not speaking this
                        # protocol; reconnecting would just repeat the failure.
                        raise MediaClientError(f"malformed relay frame: {exc}") from exc
            finally:
                self._connection = None

    async def _frames(self, connection: ClientConnection) -> AsyncIterator[bytes | str]:
        """Yield raw frames, turning a refusal into a clear error.

        Without this an unauthorized consumer sees an empty stream and no
        explanation, which is indistinguishable from "no publisher yet".
        """
        try:
            async for frame in connection:
                yield frame
        except websockets.ConnectionClosed as exc:
            received = exc.rcvd
            if received is not None and received.code == POLICY_VIOLATION:
                raise MediaClientError(
                    f"relay refused the connection: {received.reason or 'policy violation'}"
                ) from exc
            raise


__all__ = ["DEFAULT_MAX_MESSAGE_BYTES", "MediaClient", "MediaClientError", "ReconnectPolicy"]
