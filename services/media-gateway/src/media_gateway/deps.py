"""Shared request dependencies.

`docs/07-Privacy-and-Security.md` is explicit that "a trusted LAN is not an
authentication mechanism", so internal surfaces authenticate even though they
are only published to the local network.
"""

from __future__ import annotations

import hmac

from fastapi import Request, WebSocket

from media_gateway.config import Settings
from media_gateway.errors import UnauthorizedError

BEARER = "bearer"


def _presented_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != BEARER or not value:
        return None
    return value.strip()


def _matches_internal_token(settings: Settings, presented: str | None) -> bool:
    expected = settings.internal_api_token
    if expected is None:
        return True
    return presented is not None and hmac.compare_digest(presented, expected.get_secret_value())


def check_internal_token(settings: Settings, authorization: str | None) -> None:
    """Authorize an internal caller, or raise.

    When no token is configured the surface is open, which is the local
    development default. Deploy configuration refuses to start without one, so
    this cannot silently be open in production.
    """
    expected = settings.internal_api_token
    if expected is None:
        return

    presented = _presented_token(authorization)
    if presented is None:
        raise UnauthorizedError("missing bearer token")
    # Constant time: a timing oracle on a shared secret is cheap to avoid.
    if not _matches_internal_token(settings, presented):
        raise UnauthorizedError("invalid bearer token")


def authorize_request(request: Request) -> None:
    check_internal_token(request.app.state.settings, request.headers.get("authorization"))


def authorize_websocket(websocket: WebSocket) -> None:
    authorization = websocket.headers.get("authorization")
    if authorization is None:
        query_token = websocket.query_params.get("token")
        authorization = f"Bearer {query_token}" if query_token else None
    check_internal_token(websocket.app.state.settings, authorization)


def _is_device_credential(presented: str | None) -> bool:
    """Shape check only; `verify` is what actually authorizes."""
    return presented is not None and presented.startswith("v1.")


def authorize_device_request(request: Request, *, device_id: str) -> None:
    """Accept the internal operator or a credential for exactly one device."""
    settings: Settings = request.app.state.settings
    presented = _presented_token(request.headers.get("authorization"))
    # A presented device credential is always verified, even on the open local
    # development surface. Otherwise device scoping exists only where nobody
    # tests it, and the first time it runs for real is in front of an audience.
    if _matches_internal_token(settings, presented) and not _is_device_credential(presented):
        return
    if presented is None:
        raise UnauthorizedError("missing bearer token")
    claimed_device = request.app.state.device_credentials.verify(presented)
    if not hmac.compare_digest(claimed_device, device_id):
        raise UnauthorizedError("device credential does not match the requested device")


def authorize_device_websocket(websocket: WebSocket, *, device_id: str) -> None:
    """WebSocket form of :func:`authorize_device_request`.

    A device credential is accepted **only** from the `Authorization` header.
    The query string exists because a browser cannot set headers on a
    WebSocket, and the console has an operator token already; a device
    credential is long-lived, so widening that concession to it would put a
    week-long secret into every URL that anything might log.
    """
    settings: Settings = websocket.app.state.settings
    header_token = _presented_token(websocket.headers.get("authorization"))
    presented = header_token or websocket.query_params.get("token")
    if _matches_internal_token(settings, presented) and not _is_device_credential(presented):
        return
    if presented is None:
        raise UnauthorizedError("missing bearer token")
    if header_token is None:
        raise UnauthorizedError("device credentials must use the authorization header")
    claimed_device = websocket.app.state.device_credentials.verify(header_token)
    if not hmac.compare_digest(claimed_device, device_id):
        raise UnauthorizedError("device credential does not match the requested device")


__all__ = [
    "authorize_device_request",
    "authorize_device_websocket",
    "authorize_request",
    "authorize_websocket",
    "check_internal_token",
]
