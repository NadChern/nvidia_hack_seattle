"""Structured JSON logging with redaction.

docs/07-Privacy-and-Security.md requires that logs never contain raw media,
transcripts, tokens, or precise evidence paths. Redaction is enforced by a
filter rather than by reviewer discipline, because the failure mode -- a JWT or
a frame in a log file -- is silent and permanent.

Mirrors the existing service logging modules
exactly, not just in spirit: this is a repository-wide pattern, not a
per-service one, and the three should stay identical.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Any

#: Attributes the stdlib puts on every record; anything else is context we add.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

#: Uvicorn attaches an ANSI-coloured duplicate of its own message; it is pure
#: noise in structured output.
_DROPPED_ATTRS = frozenset({"color_message"})

#: Keys whose values are always secret regardless of content.
SENSITIVE_KEYS = frozenset(
    {
        "api_secret",
        "authorization",
        "credential",
        "credentials",
        "livekit_api_secret",
        "password",
        "secret",
        "token",
    }
)

#: A JWT is three base64url segments; the header almost always starts `eyJ`.
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")

#: `?token=...` in a URL. A browser cannot set headers on a WebSocket, so the
#: console authenticates in the query string and the value reaches anything that
#: logs a request line. Device credentials are `v1.<payload>.<sig>` and internal
#: tokens are opaque, so neither is caught by the JWT shape above.
_QUERY_SECRET = re.compile(
    r"(?i)\b(token|credential|api_key|apikey|secret|password|code)=[^&\s\"']+"
)

REDACTED = "[redacted]"
_MAX_STRING = 512


def _redact_value(key: str, value: Any, *, truncate: bool = True) -> Any:
    """Redact by key name, by shape, and by type.

    `truncate` is off for the log message itself: a message is often a
    traceback, and cutting one off destroys the only thing that makes a
    startup failure diagnosable.
    """
    if key.lower() in SENSITIVE_KEYS:
        return REDACTED
    if isinstance(value, bytes | bytearray):
        # Never serialize payloads. Log the size instead.
        return f"<{len(value)} bytes>"
    if isinstance(value, memoryview):
        return f"<{value.nbytes} bytes>"
    if isinstance(value, str):
        redacted = _JWT.sub(REDACTED, value)
        redacted = _QUERY_SECRET.sub(lambda match: f"{match.group(1)}={REDACTED}", redacted)
        if truncate and len(redacted) > _MAX_STRING:
            return redacted[:_MAX_STRING] + "...[truncated]"
        return redacted
    if isinstance(value, dict):
        return {str(k): _redact_value(str(k), v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(value, list | tuple):
        return [_redact_value(key, item) for item in value]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return value


class RedactionFilter(logging.Filter):
    """Redact secrets from the message and from every extra field."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_value("msg", record.msg, truncate=False)
        # Uvicorn's access logger puts the request line in `args`, not `msg`, so
        # redacting the message alone leaves `?token=` in the formatted output.
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_value("args", item, truncate=False) for item in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: _redact_value(str(key), value, truncate=False)
                for key, value in record.args.items()  # pyright: ignore[reportUnknownVariableType]
            }
        for key, value in list(record.__dict__.items()):
            if key in _STANDARD_ATTRS or key in _DROPPED_ATTRS or key.startswith("_"):
                continue
            record.__dict__[key] = _redact_value(key, value)
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, with context fields inlined."""

    def __init__(self, *, service: str, version: str) -> None:
        super().__init__()
        self.service = service
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "version": self.version,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key in _DROPPED_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(*, level: str, service: str, version: str) -> None:
    """Install the JSON formatter and redaction filter on the root logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(service=service, version=version))
    handler.addFilter(RedactionFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


__all__ = [
    "REDACTED",
    "SENSITIVE_KEYS",
    "JsonFormatter",
    "RedactionFilter",
    "configure_logging",
]
