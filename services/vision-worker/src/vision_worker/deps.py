"""Shared request dependencies.

`docs/07-Privacy-and-Security.md` is explicit that "a trusted LAN is not an
authentication mechanism", so internal surfaces authenticate even though they
are only published to the local network. Matches `media_gateway.deps`.
"""

from __future__ import annotations

import hmac

from fastapi import Request, WebSocket

from vision_worker.config import Settings
from vision_worker.errors import UnauthorizedError

BEARER = "bearer"


def _presented_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != BEARER or not value:
        return None
    return value.strip()


def check_internal_token(settings: Settings, authorization: str | None) -> None:
    """Authorize an internal caller, or raise.

    When no token is configured the surface is open, which is the local
    development default.
    """
    expected = settings.internal_api_token
    if expected is None:
        return

    presented = _presented_token(authorization)
    if presented is None:
        raise UnauthorizedError("missing bearer token")
    # Constant time: a timing oracle on a shared secret is cheap to avoid.
    if not hmac.compare_digest(presented, expected.get_secret_value()):
        raise UnauthorizedError("invalid bearer token")


def authorize_request(request: Request) -> None:
    check_internal_token(request.app.state.settings, request.headers.get("authorization"))


def authorize_websocket(websocket: WebSocket) -> None:
    """Authorize a WebSocket, accepting the token in the query string.

    **A browser cannot set headers on a WebSocket handshake.** The browser
    WebSocket API takes a URL and an optional subprotocol list and nothing else,
    so a viewer could never present a bearer header no matter how it was
    written. Header-only auth here would mean the overlay stream works in
    development -- where no token is configured and the surface is open -- and
    is unreachable in deploy, which is precisely where a demo runs.

    So `?token=` is accepted as well, and only here: every other surface stays
    header-only, because every other caller is a service that can set one.

    The cost is that the token appears in the URL, and URLs reach places headers
    do not -- server logs, browser history, `Referer`. That is acceptable for a
    LAN-scoped internal token guarding a read-only telemetry stream, and would
    not be for anything that authorizes a write.
    """
    presented = websocket.headers.get("authorization")
    if presented is None:
        token = websocket.query_params.get("token")
        if token:
            presented = f"Bearer {token}"
    check_internal_token(websocket.app.state.settings, presented)


__all__ = ["authorize_request", "authorize_websocket", "check_internal_token"]
