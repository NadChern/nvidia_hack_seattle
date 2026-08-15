"""Typed errors mapped to explicit HTTP responses.

Every service must "return explicit unavailable, invalid, unauthorized, and
ambiguous results" (docs/11-Engineering-Standards.md), so failures carry a
stable machine-readable `code` rather than a prose message a caller has to
pattern-match on.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """Base class for errors with a defined external representation."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            payload["context"] = self.context
        return payload


class InvalidRequestError(GatewayError):
    status_code = 400
    code = "invalid_request"


class UnauthorizedError(GatewayError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(GatewayError):
    """Authenticated, but not permitted -- an unlisted device, for example."""

    status_code = 403
    code = "forbidden"


class NotFoundError(GatewayError):
    status_code = 404
    code = "not_found"


class ConflictError(GatewayError):
    """The request is valid but conflicts with current state."""

    status_code = 409
    code = "conflict"


class CapacityError(GatewayError):
    """A bounded resource is exhausted -- sessions or relay subscribers."""

    status_code = 429
    code = "capacity_exhausted"


class UnavailableError(GatewayError):
    """A dependency the request needs is not reachable."""

    status_code = 503
    code = "unavailable"


__all__ = [
    "CapacityError",
    "ConflictError",
    "ForbiddenError",
    "GatewayError",
    "InvalidRequestError",
    "NotFoundError",
    "UnauthorizedError",
    "UnavailableError",
]
