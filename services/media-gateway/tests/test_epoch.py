"""Media epoch rules: the track SID is the boundary, not the identity."""

import datetime as dt

from media_gateway.config import Settings
from media_gateway.domain.epoch import EpochRegistry

T0 = dt.datetime(2026, 7, 30, 18, 0, 0, tzinfo=dt.UTC)
SESSION = "sess_01JAB"
DEVICE = "glasses-01"


def a_registry(**overrides: object) -> EpochRegistry:
    return EpochRegistry(Settings(media_source="scripted", **overrides))  # type: ignore[arg-type]


def begin(registry: EpochRegistry, track_sid: str, *, kind: str = "video"):
    return registry.begin(
        session_id=SESSION,
        stream_kind=kind,  # type: ignore[arg-type]
        track_sid=track_sid,
        participant_identity=DEVICE,
        at=T0,
    )


def test_a_rejoin_with_the_same_identity_is_a_new_epoch() -> None:
    """The spike proved rejoins keep the identity but change the track SID."""
    registry = a_registry()
    first, _ = begin(registry, "TR_VCaaa")
    second, displaced = begin(registry, "TR_VCbbb")

    assert first.participant_identity == second.participant_identity
    assert first.epoch_id != second.epoch_id
    assert displaced is first
    assert not first.active
    assert second.active


def test_sequence_restarts_at_zero_for_each_epoch() -> None:
    registry = a_registry()
    first, _ = begin(registry, "TR_VCaaa")
    for _ in range(5):
        first.take_sequence()

    second, _ = begin(registry, "TR_VCbbb")

    assert first.next_sequence == 5
    assert second.take_sequence() == 0


def test_epoch_id_is_the_track_sid() -> None:
    registry = a_registry()
    epoch, _ = begin(registry, "TR_VCaaa")

    assert epoch.epoch_id == epoch.track_sid == "TR_VCaaa"


def test_video_and_audio_epochs_are_independent() -> None:
    registry = a_registry()
    video, _ = begin(registry, "TR_VCaaa", kind="video")
    audio, displaced = begin(registry, "TR_ACbbb", kind="audio")

    assert displaced is None
    assert video.active
    assert audio.active
    assert registry.active_for(SESSION, "video") is video
    assert registry.active_for(SESSION, "audio") is audio


def test_only_video_epochs_carry_a_dimension_guard() -> None:
    registry = a_registry()
    video, _ = begin(registry, "TR_VCaaa", kind="video")
    audio, _ = begin(registry, "TR_ACbbb", kind="audio")

    assert video.guard is not None
    assert audio.guard is None


def test_ending_an_epoch_clears_the_active_slot() -> None:
    registry = a_registry()
    begin(registry, "TR_VCaaa")

    ended = registry.end_active(
        session_id=SESSION, stream_kind="video", reason="track_unsubscribed"
    )

    assert ended is not None
    assert ended.end_reason == "track_unsubscribed"
    assert registry.active_for(SESSION, "video") is None


def test_ending_an_epoch_twice_keeps_the_first_reason() -> None:
    registry = a_registry()
    epoch, _ = begin(registry, "TR_VCaaa")

    epoch.end("track_unsubscribed", at=T0)
    epoch.end("gateway_shutdown", at=T0 + dt.timedelta(seconds=5))

    assert epoch.end_reason == "track_unsubscribed"
    assert epoch.ended_at == T0


def test_ending_a_session_ends_every_stream() -> None:
    registry = a_registry()
    begin(registry, "TR_VCaaa", kind="video")
    begin(registry, "TR_ACbbb", kind="audio")

    ended = registry.end_session(SESSION, reason="session_ended")

    assert len(ended) == 2
    assert all(not epoch.active for epoch in ended)
    assert registry.active() == []


def test_ending_an_absent_epoch_is_not_an_error() -> None:
    registry = a_registry()

    assert (
        registry.end_active(session_id=SESSION, stream_kind="video", reason="track_unsubscribed")
        is None
    )


def test_forgetting_a_session_bounds_the_registry() -> None:
    registry = a_registry()
    begin(registry, "TR_VCaaa")
    registry.end_session(SESSION, reason="session_ended")

    registry.forget_session(SESSION)

    assert registry.get("TR_VCaaa") is None
    assert registry.active() == []
