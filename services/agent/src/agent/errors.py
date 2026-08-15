"""Errors with stable HTTP representations."""

from __future__ import annotations

from typing import Any


class AgentServiceError(Exception):
    status_code = 500
    code = "agent_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.context}


class UnauthorizedError(AgentServiceError):
    status_code = 401
    code = "unauthorized"


class DependencyUnavailableError(AgentServiceError):
    status_code = 503
    code = "dependency_unavailable"


__all__ = ["AgentServiceError", "DependencyUnavailableError", "UnauthorizedError"]
