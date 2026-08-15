"""Typed errors mapped to explicit HTTP responses.

docs/11-Engineering-Standards.md requires every service to "return explicit
unavailable, invalid, unauthorized, and ambiguous results", so failures carry a
stable machine-readable `code` rather than prose a caller has to pattern-match.
"""

from __future__ import annotations

from typing import Any


class MemoryServiceError(Exception):
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


class InvalidRequestError(MemoryServiceError):
    status_code = 400
    code = "invalid_request"


class UnauthorizedError(MemoryServiceError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(MemoryServiceError):
    """Authenticated, but not permitted -- an unlisted device, for example."""

    status_code = 403
    code = "forbidden"


class NotFoundError(MemoryServiceError):
    status_code = 404
    code = "not_found"


class ConflictError(MemoryServiceError):
    """Valid, but irreconcilable with what is already stored."""

    status_code = 409
    code = "conflict"


__all__ = [
    "ConflictError",
    "ForbiddenError",
    "InvalidRequestError",
    "MemoryServiceError",
    "NotFoundError",
    "UnauthorizedError",
]
