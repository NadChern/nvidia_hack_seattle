"""The room worker must ingest only a session's own publisher.

Remote assist puts a second participant (a `helper-<session_id>` identity)
into a room that previously only ever held the wearer. Without the identity
check these tests assert, that helper's microphone would start a new audio
epoch and reach Speech/the Agent -- transcribing, and possibly replying to,
the helper instead of the wearer.

These are plain unit tests against `RoomWorker`'s event handlers, not the
opt-in LiveKit integration suite: the handlers are synchronous and only read
`.kind` / `.identity` off their arguments, so a minimal stand-in is enough and
no LiveKit server is needed.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from livekit import rtc

from media_gateway.config import Settings
from media_gateway.domain.session import Session
from media_gateway.transport.room_worker import RoomWorker
from media_gateway.transport.tokens import helper_identity

# `rtc.Room()` (constructed in `RoomWorker.__init__`) grabs the running event
# loop, so every test needs one -- these are async only for that reason, none
# of them actually await anything on the worker itself.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def session_started(self, **kwargs: object) -> None:
        self.calls.append(("session_started", kwargs))

    def epoch_started(self, **kwargs: object) -> None:
        self.calls.append(("epoch_started", kwargs))

    def video_frame(self, **kwargs: object) -> None:
        self.calls.append(("video_frame", kwargs))

    def audio_frame(self, **kwargs: object) -> None:
        self.calls.append(("audio_frame", kwargs))

    def epoch_ended(self, **kwargs: object) -> None:
        self.calls.append(("epoch_ended", kwargs))

    def session_ended(self, **kwargs: object) -> None:
        self.calls.append(("session_ended", kwargs))


def _settings() -> Settings:
    return Settings(environment="ci", media_source="scripted")


def _session() -> Session:
    now = dt.datetime.now(dt.UTC)
    return Session(
        session_id="sess_01",
        device_id="glasses-01",
        room="vma-sess_01",
        created_at=now,
        last_seen_at=now,
    )


def _publication(sid: str = "TR_video_1") -> SimpleNamespace:
    return SimpleNamespace(sid=sid, simulcasted=False, width=1280, height=720)


def _participant(identity: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity)


async def test_a_track_from_a_helper_never_starts_an_epoch() -> None:
    session = _session()
    sink = RecordingSink()
    worker = RoomWorker(settings=_settings(), session=session, sink=sink)

    track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
    worker._on_track_subscribed(  # noqa: SLF001 -- exercising the handler directly
        track, _publication("TR_audio_1"), _participant(helper_identity(session.session_id))
    )

    assert sink.calls == []
    assert worker._tasks == set()  # no consumer spawned for the ignored track


async def test_a_track_from_the_wearers_own_publisher_still_starts_an_epoch() -> None:
    session = _session()
    sink = RecordingSink()
    worker = RoomWorker(settings=_settings(), session=session, sink=sink)

    track = SimpleNamespace(kind=rtc.TrackKind.KIND_AUDIO)
    worker._on_track_subscribed(  # noqa: SLF001
        track, _publication("TR_audio_1"), _participant(session.device_id)
    )

    assert [name for name, _ in sink.calls] == ["epoch_started"]
    started = sink.calls[0][1]
    assert started["participant_identity"] == session.device_id
    for task in worker._tasks:  # noqa: SLF001 -- clean up the spawned consumer
        task.cancel()


async def test_a_helper_connecting_or_leaving_does_not_touch_publisher_present() -> None:
    session = _session()
    sink = RecordingSink()
    left: list[tuple[str, str]] = []
    worker = RoomWorker(
        settings=_settings(),
        session=session,
        sink=sink,
        on_participant_left=lambda session_id, identity: left.append((session_id, identity)),
    )
    session.publisher_present = True

    helper = _participant(helper_identity(session.session_id))
    worker._on_participant_connected(helper)  # noqa: SLF001
    assert session.publisher_present is True  # unchanged -- a helper is not the wearer

    worker._on_participant_disconnected(helper)  # noqa: SLF001
    assert session.publisher_present is True  # still unchanged
    assert sink.calls == []  # no epoch_ended fired for the wearer's stream
    assert left == [(session.session_id, helper_identity(session.session_id))]


async def test_the_wearer_disconnecting_still_clears_publisher_present() -> None:
    session = _session()
    sink = RecordingSink()
    worker = RoomWorker(settings=_settings(), session=session, sink=sink)
    session.publisher_present = True

    worker._on_participant_disconnected(_participant(session.device_id))  # noqa: SLF001

    assert session.publisher_present is False
    assert [name for name, _ in sink.calls] == ["epoch_ended", "epoch_ended"]
