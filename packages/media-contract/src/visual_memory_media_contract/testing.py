"""Test doubles for consumers of the media relay.

`replay_server` serves a recorded fixture over a real WebSocket, so a consumer
can exercise `MediaClient` end to end with no gateway, no LiveKit, and no
hardware:

    async with replay_server("video_session_basic") as url:
        async for message in MediaClient(url, reconnect=False):
            ...
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from visual_memory_media_contract.fixtures import load_fixture
from visual_memory_media_contract.framing import decode_message
from visual_memory_media_contract.protocol import RelayMessage

Handler = Callable[[ServerConnection], Coroutine[Any, Any, None]]


async def _send_all(connection: ServerConnection, frames: Sequence[bytes], delay: float) -> None:
    try:
        for frame in frames:
            await connection.send(frame)
            if delay:
                await asyncio.sleep(delay)
    except websockets.ConnectionClosed:  # pragma: no cover - client hung up early
        return


def _bound_port(server: Server) -> int:
    """Return the ephemeral port a server bound to.

    `getsockname` is typed as Any because its shape depends on the address
    family, so narrow it explicitly rather than trusting the index.
    """
    address = cast(tuple[object, ...], server.sockets[0].getsockname())
    port = address[1] if len(address) >= 2 else None
    if not isinstance(port, int):  # pragma: no cover - non-INET socket
        raise RuntimeError(f"expected an INET socket address, got {address!r}")
    return port


@asynccontextmanager
async def _serving(handler: Handler, host: str) -> AsyncGenerator[str]:
    """Run `handler` on an ephemeral port and yield its URL.

    Ephemeral so parallel tests never collide on a fixed port.
    """
    server = await serve(handler, host, 0)
    try:
        yield f"ws://{host}:{_bound_port(server)}"
    finally:
        server.close()
        await server.wait_closed()


def _frames_for(fixture: str | Sequence[bytes]) -> list[bytes]:
    return load_fixture(fixture) if isinstance(fixture, str) else list(fixture)


@asynccontextmanager
async def replay_server(
    fixture: str | Sequence[bytes],
    *,
    delay: float = 0.0,
    close_after: bool = True,
    host: str = "127.0.0.1",
) -> AsyncGenerator[str]:
    """Serve a fixture over a real WebSocket and yield its URL.

    When `close_after` is false the connection stays open after the last
    frame, which is how you exercise a consumer's idle handling.
    """
    frames = _frames_for(fixture)

    async def handler(connection: ServerConnection) -> None:
        await _send_all(connection, frames, delay)
        if close_after:
            await connection.close()
        else:  # pragma: no cover - only used by idle tests
            await connection.wait_closed()

    async with _serving(handler, host) as url:
        yield url


@asynccontextmanager
async def flaky_replay_server(
    fixture: str | Sequence[bytes],
    *,
    drop_after: int,
    host: str = "127.0.0.1",
) -> AsyncGenerator[str]:
    """Serve a fixture but hang up mid-stream on the first connection.

    Exercises the reconnect path: the first connection receives `drop_after`
    frames and is then dropped without a close handshake; later connections
    receive the whole fixture.
    """
    frames = _frames_for(fixture)
    connections = 0

    async def handler(connection: ServerConnection) -> None:
        nonlocal connections
        connections += 1
        if connections == 1:
            await _send_all(connection, frames[:drop_after], 0.0)
            # Abort without a close handshake so the client sees a transport
            # error rather than a clean end of stream.
            connection.transport.abort()
            return
        await _send_all(connection, frames, 0.0)
        await connection.close()

    async with _serving(handler, host) as url:
        yield url


def messages_of(frames: Sequence[bytes]) -> list[RelayMessage]:
    """Decode wire frames into messages."""
    return [decode_message(frame) for frame in frames]


def assert_matches_fixture(observed: Sequence[RelayMessage], fixture: str) -> None:
    """Assert an observed message sequence equals a recorded fixture.

    Both the gateway and its consumers assert against the same files, so a
    provider change that breaks a consumer fails on both sides.
    """
    expected = messages_of(load_fixture(fixture))
    if len(observed) != len(expected):
        observed_types = [message.type for message in observed]
        expected_types = [message.type for message in expected]
        raise AssertionError(
            f"fixture {fixture!r} has {len(expected)} messages, observed {len(observed)}\n"
            f"  expected: {expected_types}\n"
            f"  observed: {observed_types}"
        )
    for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
        if actual != wanted:
            raise AssertionError(
                f"fixture {fixture!r} message {index} ({wanted.type}) differs\n"
                f"  expected: {wanted!r}\n"
                f"  observed: {actual!r}"
            )


__all__ = [
    "assert_matches_fixture",
    "flaky_replay_server",
    "messages_of",
    "replay_server",
]
