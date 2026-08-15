"""Mint LiveKit access tokens.

Tokens are minted here rather than embedded in a client. That is LiveKit
adoption gate 5 from the S01 spike results, and docs/07-Privacy-and-Security.md
requires short-lived, least-privilege grants with secrets kept in protected
runtime configuration.

Every token is scoped to one room, carries the shortest useful lifetime, and
never grants the data channel: this project moves media, not arbitrary payloads
between participants.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from livekit import api

from media_gateway.config import Settings
from media_gateway.errors import UnavailableError

#: `publisher` is the glasses client: it publishes camera and microphone, and
#: subscribes so it can hear the assistant's return audio. `worker` is the
#: gateway itself: it subscribes to media and publishes synthesized speech.
#: `viewer` is the operator console and may only subscribe.
GrantRole = Literal["publisher", "viewer", "worker"]


class MintedToken:
    """A signed token and the facts a caller needs to use it."""

    __slots__ = ("expires_at", "identity", "room", "token")

    def __init__(self, *, token: str, identity: str, room: str, expires_at: dt.datetime) -> None:
        self.token = token
        self.identity = identity
        self.room = room
        self.expires_at = expires_at

    def __repr__(self) -> str:
        # Never render the token: repr lands in logs and tracebacks.
        return f"MintedToken(identity={self.identity!r}, room={self.room!r})"


def mint_access_token(
    settings: Settings,
    *,
    identity: str,
    room: str,
    role: GrantRole,
    ttl_s: int | None = None,
    now: dt.datetime | None = None,
) -> MintedToken:
    """Sign a room-scoped token, clamping the lifetime to the configured cap.

    Raises UnavailableError when LiveKit credentials are absent, which is the
    normal state under `VMA_MEDIA_SOURCE=scripted`.
    """
    try:
        key, secret = settings.require_livekit_credentials()
    except ValueError as exc:
        raise UnavailableError("livekit credentials are not configured") from exc

    # A caller may ask for less, never more.
    lifetime = settings.token_ttl_s if ttl_s is None else min(ttl_s, settings.token_ttl_s)
    lifetime = max(1, lifetime)

    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=role != "viewer",
        can_subscribe=True,
        # No data channel: this transport carries media only, and a data
        # channel would be an unaudited side path between participants.
        can_publish_data=False,
        # Neither role administers rooms or starts recordings.
        room_create=False,
        room_admin=False,
        room_list=False,
        room_record=False,
        can_update_own_metadata=False,
    )

    token = (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(dt.timedelta(seconds=lifetime))
        .with_grants(grants)
        .to_jwt()
    )

    issued_at = now or dt.datetime.now(dt.UTC)
    return MintedToken(
        token=token,
        identity=identity,
        room=room,
        expires_at=issued_at + dt.timedelta(seconds=lifetime),
    )


def publisher_identity(device_id: str) -> str:
    """Identity for a glasses client.

    Stable across rejoins on purpose: the S01 spike showed the track SID, not
    the identity, is what marks a new media epoch.
    """
    return device_id


def viewer_identity(session_id: str) -> str:
    """Stable identity for the one operator view of a session."""
    return f"viewer-{session_id}"


def worker_identity(session_id: str) -> str:
    """Identity the gateway itself joins a room under."""
    return f"gateway-{session_id}"


__all__ = [
    "GrantRole",
    "MintedToken",
    "mint_access_token",
    "publisher_identity",
    "viewer_identity",
    "worker_identity",
]
