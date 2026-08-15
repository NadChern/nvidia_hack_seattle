"""Single-use pairing codes and signed, device-scoped credentials."""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass

from media_gateway.errors import CapacityError, UnauthorizedError


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


@dataclass(frozen=True)
class IssuedPairingCode:
    code: str
    expires_at: dt.datetime


@dataclass(frozen=True)
class IssuedDeviceCredential:
    credential: str
    device_id: str
    expires_at: dt.datetime


class DeviceCredentialSigner:
    """HMAC credentials that survive restart when the internal token is stable."""

    def __init__(
        self,
        *,
        secret: bytes,
        ttl_s: int,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("device credential signing secret must be at least 32 bytes")
        self._secret = secret
        self._ttl_s = ttl_s
        self._now = now

    def issue(self, device_id: str) -> IssuedDeviceCredential:
        expires_at = self._now() + dt.timedelta(seconds=self._ttl_s)
        payload = {
            "device_id": device_id,
            "exp": int(expires_at.timestamp()),
            "nonce": secrets.token_urlsafe(12),
            "version": 1,
        }
        encoded = _encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        signature = _encode(hmac.digest(self._secret, f"v1.{encoded}".encode(), "sha256"))
        return IssuedDeviceCredential(
            credential=f"v1.{encoded}.{signature}",
            device_id=device_id,
            expires_at=expires_at,
        )

    def verify(self, credential: str) -> str:
        try:
            version, encoded, signature = credential.split(".", maxsplit=2)
            if version != "v1":
                raise ValueError
            expected = _encode(hmac.digest(self._secret, f"v1.{encoded}".encode(), "sha256"))
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            payload = json.loads(_decode(encoded))
            device_id = payload["device_id"]
            expires_at = int(payload["exp"])
            if payload["version"] != 1 or not isinstance(device_id, str) or not device_id:
                raise ValueError
            if expires_at <= int(self._now().timestamp()):
                raise ValueError
        except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise UnauthorizedError("invalid or expired device credential") from exc
        return device_id


class PairingRegistry:
    """Bounded in-memory set of hashed, short-lived, single-use codes."""

    def __init__(
        self,
        *,
        ttl_s: int,
        max_pending: int,
        signer: DeviceCredentialSigner,
        now: Callable[[], dt.datetime] = _utcnow,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_pending = max_pending
        self._signer = signer
        self._now = now
        self._pending: dict[str, dt.datetime] = {}

    @staticmethod
    def _digest(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()

    def _sweep(self) -> None:
        now = self._now()
        self._pending = {
            digest: expires_at for digest, expires_at in self._pending.items() if expires_at > now
        }

    def issue(self) -> IssuedPairingCode:
        self._sweep()
        if len(self._pending) >= self._max_pending:
            raise CapacityError("too many pending pairing codes", limit=self._max_pending)
        code = secrets.token_urlsafe(24)
        expires_at = self._now() + dt.timedelta(seconds=self._ttl_s)
        self._pending[self._digest(code)] = expires_at
        return IssuedPairingCode(code=code, expires_at=expires_at)

    def claim(self, *, code: str, device_id: str) -> IssuedDeviceCredential:
        self._sweep()
        digest = self._digest(code)
        expires_at = self._pending.pop(digest, None)
        if expires_at is None or expires_at <= self._now():
            raise UnauthorizedError("invalid or expired pairing code")
        return self._signer.issue(device_id)

    @property
    def pending(self) -> int:
        self._sweep()
        return len(self._pending)


__all__ = [
    "DeviceCredentialSigner",
    "IssuedDeviceCredential",
    "IssuedPairingCode",
    "PairingRegistry",
]
