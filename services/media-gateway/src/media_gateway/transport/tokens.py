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
#: `viewer` is the operator console and may only subscribe. `helper` is a
#: remote-assist participant (docs/12): it subscribes like a viewer but may
#: additionally publish exactly one thing, a microphone -- never a camera,
#: which would break the single-video-publisher assumption the relay's
#: dimension guard depends on (SG-C's simulcast-collapse finding).
GrantRole = Literal["publisher", "viewer", "worker", "helper"]

#: LiveKit's own vocabulary for what a participant may publish. `VideoGrants
#: .can_publish_sources` is typed `List[str]` and `AccessToken.to_jwt()`
#: serializes it verbatim -- passing the raw `api.TrackSource.MICROPHONE`
#: enum (which is just the bare int `2`) encodes as `canPublishSources: [2]`,
#: which the LiveKit Go server then rejects as a malformed token ("cannot
#: unmarshal number into ... of type string"). Passing the proto enum's
#: *name* instead (`"MICROPHONE"`) fixes that -- the token is accepted and
#: the grant round-trips correctly (confirmed server-side in the room-join
#: log: `"CanPublishSources": ["MICROPHONE"]`) -- but a live LiveKit
#: 1.13.4 server then still logs `"no permission to publish track"` and
#: silently drops the actual audio AddTrackRequest for that participant a
#: few dozen ms later, with no further detail. Whatever string (or other
#: representation) its runtime publish-permission check actually wants,
#: it is not this one, and nothing in the client-visible protocol says
#: what would satisfy it. Left unrestricted (`None`, same as `publisher`)
#: until someone can dig into the server's own source.
#:
#: This is not risk-free: a compromised or buggy `helper` client could now
#: publish a camera track, which the relay's one-video-publisher assumption
#: does not expect. The mitigation until this is revisited is entirely
#: client-side -- CallScreen.tsx never enables the camera for a helper --
#: which is exactly the discipline-not-grant situation this constant was
#: introduced to avoid.
HELPER_PUBLISH_SOURCES = None


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
        # Unset (None): see HELPER_PUBLISH_SOURCES above for why `helper`
        # isn't narrowed to microphone-only at the grant level right now.
        can_publish_sources=HELPER_PUBLISH_SOURCES,
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


def helper_identity(session_id: str) -> str:
    """Stable identity for the one remote-assist participant on a session.

    `room_worker.py`'s ingest filter and participant-lifecycle handling key
    off this exact prefix to recognize a helper as *not* the wearer.
    """
    return f"helper-{session_id}"


__all__ = [
    "HELPER_PUBLISH_SOURCES",
    "GrantRole",
    "MintedToken",
    "helper_identity",
    "mint_access_token",
    "publisher_identity",
    "viewer_identity",
    "worker_identity",
]
