"""Deterministic checks for security assertions used by the LiveKit round trip."""

from __future__ import annotations

import jwt
import pytest

from tests.integration.roundtrip import _tamper_jwt_signature


@pytest.mark.parametrize(
    "signature",
    (
        "A_signature_that_does_not_end_in_A",
        "A_signature_that_ends_in_A",
        "Z_signature",
    ),
)
def test_signature_tampering_always_changes_the_token(signature: str) -> None:
    token = f"header.claims.{signature}"

    tampered = _tamper_jwt_signature(token)

    assert tampered != token
    assert tampered.rsplit(".", 1)[0] == token.rsplit(".", 1)[0]


def test_tampered_signed_token_fails_signature_verification() -> None:
    secret = "integration-test-secret-at-least-32-bytes"
    token = jwt.encode({"sub": "integration"}, secret, algorithm="HS256")

    tampered = _tamper_jwt_signature(token)

    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(tampered, secret, algorithms=["HS256"])


@pytest.mark.parametrize("token", ("", "one-segment", "two.segments"))
def test_malformed_token_cannot_be_tampered(token: str) -> None:
    with pytest.raises(ValueError, match="three-segment JWT"):
        _tamper_jwt_signature(token)
