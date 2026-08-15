"""Request-scoped helpers shared by the routers."""

from __future__ import annotations

import hmac

from fastapi import Request
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from application_memory.config import Settings
from application_memory.errors import ForbiddenError, UnauthorizedError


def settings_of(request: Request) -> Settings:
    resolved: Settings = request.app.state.settings
    return resolved


def session_factory_of(request: Request) -> sessionmaker[DbSession]:
    factory: sessionmaker[DbSession] = request.app.state.sessions
    return factory


def authorize_request(request: Request) -> None:
    """Refuse an unauthenticated caller when a token is configured.

    Compared with `compare_digest` so a wrong token cannot be discovered one
    character at a time from response timing.
    """
    settings = settings_of(request)
    if settings.internal_api_token is None:
        return

    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        raise UnauthorizedError("a bearer token is required")
    if not hmac.compare_digest(presented, settings.internal_api_token.get_secret_value()):
        raise UnauthorizedError("the bearer token is not valid")


def authorize_device(request: Request, device_id: str) -> None:
    """Refuse writes from a device nobody listed.

    An allowlist is only a control if it is checked on the write path; checking
    it at session creation alone would leave the ingestion endpoint open.
    """
    allowlist = settings_of(request).device_id_allowlist
    if allowlist and device_id not in allowlist:
        raise ForbiddenError("device is not allowed to write observations", device_id=device_id)


__all__ = ["authorize_device", "authorize_request", "session_factory_of", "settings_of"]
