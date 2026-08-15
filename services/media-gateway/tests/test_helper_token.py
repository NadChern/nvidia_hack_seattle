"""The `helper` grant is a security boundary: prove it in the minted JWT.

A test asserting the endpoint returns 200 would not catch a `helper` grant
that quietly ended up able to publish a camera or the data channel -- the
whole reason this role exists is `can_publish_sources` restricting it to a
microphone, confirmed against the installed `livekit-server-sdk` (see
`transport/tokens.py`'s `HELPER_PUBLISH_SOURCES`).
"""

from __future__ import annotations

import jwt
from livekit import api

from media_gateway.config import Settings
from media_gateway.transport.tokens import helper_identity, mint_access_token

SECRET = "a-livekit-secret-of-at-least-32-chars"


def _settings() -> Settings:
    return Settings(
        environment="ci",
        media_source="livekit",
        livekit_api_key="test-key",
        livekit_api_secret=SECRET,
    )


def _grants(token: str) -> dict[str, object]:
    decoded = jwt.decode(token, SECRET, algorithms=["HS256"])
    return decoded["video"]


def test_helper_grant_can_publish_only_a_microphone() -> None:
    minted = mint_access_token(
        _settings(),
        identity=helper_identity("sess_01"),
        room="vma-sess_01",
        role="helper",
    )

    grants = _grants(minted.token)
    assert grants["canPublish"] is True
    assert grants["canPublishSources"] == [api.TrackSource.MICROPHONE]
    assert grants["canSubscribe"] is True


def test_helper_grant_forbids_the_data_channel_and_room_administration() -> None:
    minted = mint_access_token(
        _settings(), identity=helper_identity("sess_01"), room="vma-sess_01", role="helper"
    )

    grants = _grants(minted.token)
    assert grants["canPublishData"] is False
    assert grants.get("roomAdmin", False) is False
    assert grants.get("roomRecord", False) is False


def test_helper_identity_is_stable_and_distinct_from_the_publisher() -> None:
    assert helper_identity("sess_01") == "helper-sess_01"
    assert helper_identity("sess_01") != "sess_01"


def test_viewer_and_publisher_grants_are_unaffected_by_the_helper_source_restriction() -> None:
    viewer = mint_access_token(
        _settings(), identity="viewer-sess_01", room="vma-sess_01", role="viewer"
    )
    publisher = mint_access_token(
        _settings(), identity="glasses-01", room="vma-sess_01", role="publisher"
    )

    viewer_grants = _grants(viewer.token)
    publisher_grants = _grants(publisher.token)

    assert viewer_grants["canPublish"] is False
    assert "canPublishSources" not in viewer_grants

    assert publisher_grants["canPublish"] is True
    # No source restriction: the glasses still publish both camera and mic.
    assert "canPublishSources" not in publisher_grants
