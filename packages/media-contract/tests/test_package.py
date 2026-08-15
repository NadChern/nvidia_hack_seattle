from visual_memory_media_contract import PROTOCOL_VERSION


def test_protocol_version_is_pinned() -> None:
    assert PROTOCOL_VERSION == "media-relay/1.0"
