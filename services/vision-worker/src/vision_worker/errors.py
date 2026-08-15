"""Typed errors mapped to explicit HTTP responses.

Every service must "return explicit unavailable, invalid, unauthorized, and
ambiguous results" (docs/11-Engineering-Standards.md), so failures carry a
stable machine-readable `code` rather than a prose message a caller has to
pattern-match on. Matches `media_gateway.errors`.
"""

from __future__ import annotations

from typing import Any


class VisionError(Exception):
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


class UnauthorizedError(VisionError):
    status_code = 401
    code = "unauthorized"


__all__ = ["UnauthorizedError", "VisionError"]
