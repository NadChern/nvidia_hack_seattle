"""Redaction guarantees.

docs/07-Privacy-and-Security.md requires that logs never contain raw media,
transcripts, tokens, or precise evidence paths. These tests are the
enforcement: the failure mode is silent and permanent, so it cannot rest on
reviewer discipline. Mirrors `media_gateway/tests/test_logging.py` and
`application_memory/tests/test_logging.py` -- same module, same guarantees.
"""

import json
import logging

import pytest

from speech.logging import REDACTED, JsonFormatter, RedactionFilter

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
    formatted = JsonFormatter(service="speech", version="0.1.0").format(record)
    parsed = json.loads(formatted)
    assert isinstance(parsed, dict)
    return parsed


def test_record_is_valid_json_with_service_context() -> None:
    payload = emit()

    assert payload["service"] == "speech"
    assert payload["version"] == "0.1.0"
    assert payload["level"] == "INFO"
    assert payload["message"] == "a message"


def test_context_fields_are_inlined() -> None:
    payload = emit(session_id="sess_01", epoch_id="TR_VC1")

    assert payload["session_id"] == "sess_01"
    assert payload["epoch_id"] == "TR_VC1"


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


def test_bytes_are_never_serialized() -> None:
    """The generic guarantee this service leans on for `AudioSegment.pcm`/
    `SpeechAudio.pcm` -- raw audio collapses to a byte count, never content,
    even if a call site ever passed it as an `extra` field by mistake.
    """
    pcm = b"\x00\x01\xff\xfe raw audio bytes"
    payload = emit(pcm=pcm)

    assert payload["pcm"] == f"<{len(pcm)} bytes>"
    assert "audio" not in json.dumps(payload)


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
