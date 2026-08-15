"""Run the real app under uvicorn for WebSocket tests.

`starlette.testclient` warns that pairing it with httpx is deprecated, and it
exercises a test shim rather than the real ASGI stack. Serving on an ephemeral
port and connecting with an ordinary WebSocket client tests what actually ships.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI


def _port_of(server: uvicorn.Server) -> int:
    for bound in server.servers:
        for socket in bound.sockets:
            address = cast(tuple[object, ...], socket.getsockname())
            if len(address) >= 2 and isinstance(address[1], int):
                return address[1]
    raise RuntimeError("server is not bound to an INET socket")  # pragma: no cover


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncGenerator[str]:
    """Serve `app` on an ephemeral port and yield its base URL."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="on",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name="test-uvicorn")

    for _ in range(500):
        if server.started:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover - startup failure
        server.should_exit = True
        await task
        raise RuntimeError("server did not start")

    try:
        yield f"127.0.0.1:{_port_of(server)}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)


__all__ = ["serve"]
