"""Redaction guarantees.

docs/07-Privacy-and-Security.md requires that logs never contain raw media,
transcripts, tokens, or precise evidence paths. These tests are the
enforcement: the failure mode is silent and permanent, so it cannot rest on
reviewer discipline.
"""

import json
import logging

import pytest

from media_gateway.logging import REDACTED, JsonFormatter, RedactionFilter

A_JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiJnbGFzc2VzLTAxIiwicm9vbSI6InZtYS0xIn0"
    ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
)


def emit(**extra: object) -> dict[str, object]:
    """Format one record through the real filter and formatter."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=extra.pop("msg", "a message"),
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)

    assert RedactionFilter().filter(record)
    formatted = JsonFormatter(service="media-gateway", version="0.1.0").format(record)
    parsed = json.loads(formatted)
    assert isinstance(parsed, dict)
    return parsed


def test_record_is_valid_json_with_service_context() -> None:
    payload = emit()

    assert payload["service"] == "media-gateway"
    assert payload["version"] == "0.1.0"
    assert payload["level"] == "INFO"
    assert payload["message"] == "a message"


def test_context_fields_are_inlined() -> None:
    payload = emit(session_id="sess_01", media_epoch_id="TR_VC1")

    assert payload["session_id"] == "sess_01"
    assert payload["media_epoch_id"] == "TR_VC1"


@pytest.mark.parametrize("key", ["token", "secret", "api_secret", "authorization"])
def test_sensitive_keys_are_redacted(key: str) -> None:
    payload = emit(**{key: "super-secret-value"})

    assert payload[key] == REDACTED
    assert "super-secret-value" not in json.dumps(payload)


def test_a_jwt_is_redacted_even_under_an_innocuous_key() -> None:
    payload = emit(detail=f"issued {A_JWT} for glasses-01")

    assert A_JWT not in json.dumps(payload)
    assert REDACTED in str(payload["detail"])


def test_a_jwt_in_the_message_is_redacted() -> None:
    payload = emit(msg=f"minted {A_JWT}")

    assert A_JWT not in json.dumps(payload)


@pytest.mark.parametrize(
    "secret",
    [
        "v1.eyBub3QtYS1qd3QgfQ.7Zq8kR3nVc0",  # device credential: not a JWT shape
        "an-internal-token-of-at-least-32-chars",  # opaque operator token
    ],
)
def test_a_query_string_token_is_redacted(secret: str) -> None:
    """A browser cannot set headers on a WebSocket, so tokens ride the URL."""
    payload = emit(msg=f"GET /v1/device/s-1/events?token={secret} HTTP/1.1")

    assert secret not in json.dumps(payload)
    assert REDACTED in str(payload["message"])


def test_a_token_in_access_log_args_is_redacted() -> None:
    """Uvicorn's access logger puts the request line in `args`, not `msg`."""
    secret = "an-internal-token-of-at-least-32-chars"
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:0", "GET", f"/v1/stream/video?token={secret}", "1.1", 200),
        exc_info=None,
    )

    assert RedactionFilter().filter(record)
    formatted = JsonFormatter(service="media-gateway", version="0.1.0").format(record)

    assert secret not in formatted
    assert REDACTED in formatted


def test_bytes_are_never_serialized() -> None:
    frame = b"\xff\xd8\xff\xe0 jpeg bytes"
    payload = emit(frame=frame)

    assert payload["frame"] == f"<{len(frame)} bytes>"
    assert "jpeg" not in json.dumps(payload)


def test_nested_secrets_are_redacted() -> None:
    payload = emit(grant={"room": "vma-1", "token": A_JWT})

    assert A_JWT not in json.dumps(payload)
    assert payload["grant"] == {"room": "vma-1", "token": REDACTED}


def test_secrets_inside_a_list_are_redacted() -> None:
    payload = emit(issued=[{"token": A_JWT}, {"token": A_JWT}])

    assert A_JWT not in json.dumps(payload)


def test_long_strings_are_truncated() -> None:
    payload = emit(blob="x" * 5000)

    assert isinstance(payload["blob"], str)
    assert len(payload["blob"]) < 600
    assert payload["blob"].endswith("[truncated]")


def test_non_serializable_values_do_not_break_a_record() -> None:
    payload = emit(when=object())

    assert isinstance(payload["when"], str)


def test_the_message_itself_is_not_truncated() -> None:
    """A message is often a traceback; truncating one hides the cause."""
    traceback = "Traceback (most recent call last):\n" + ("  File x, line 1\n" * 200)
    payload = emit(msg=traceback)

    assert isinstance(payload["message"], str)
    assert "[truncated]" not in payload["message"]
    assert len(payload["message"]) == len(traceback)


def test_a_jwt_is_still_redacted_from_an_untruncated_message() -> None:
    payload = emit(msg=f"{'padding ' * 200}{A_JWT}")

    assert A_JWT not in json.dumps(payload)


def test_uvicorn_color_message_is_dropped() -> None:
    payload = emit(color_message="Started server \x1b[36m%d\x1b[0m")

    assert "color_message" not in payload
    assert "\x1b" not in json.dumps(payload)
