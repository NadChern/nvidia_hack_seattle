"""RelayConsumer: epoch reset, device_id resolution, reconnect behavior.

Uses `visual_memory_media_contract.testing.replay_server` to serve hand-built
messages over a real WebSocket, so `MediaClient` is exercised end to end with
no gateway and no LiveKit -- the same harness the media-contract package
itself uses to test `MediaClient`.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from visual_memory_media_contract.client import MediaClient, ReconnectPolicy
from visual_memory_media_contract.framing import encode_message, payload_digest
from visual_memory_media_contract.protocol import (
    EpochEnded,
    EpochStarted,
    SessionEnded,
    SessionStarted,
    VideoFrame,
)
from visual_memory_media_contract.testing import flaky_replay_server, replay_server

from vision_worker.consume.relay import RelayConsumer

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)


def a_video_frame(*, session_id: str, epoch_id: str, sequence: int) -> bytes:
    payload = f"frame-{sequence}".encode()
    frame = VideoFrame(
        session_id=session_id,
        epoch_id=epoch_id,
        sequence=sequence,
        captured_at=T0,
        received_at=T0,
        relayed_at=T0,
        width=640,
        height=480,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
    )
    return encode_message(frame, payload)


class RecordingSink:
    """Records every callback invocation in order, for assertion."""

    def __init__(self) -> None:
        self.epoch_starts: list[tuple[str, str, str]] = []
        self.epoch_ends: list[tuple[str, str]] = []
        self.frames: list[tuple[str, str, str, int]] = []

    async def epoch_started(self, *, session_id: str, device_id: str, epoch_id: str) -> None:
        self.epoch_starts.append((session_id, device_id, epoch_id))

    async def epoch_ended(self, *, session_id: str, epoch_id: str) -> None:
        self.epoch_ends.append((session_id, epoch_id))

    async def video_frame(
        self, *, session_id: str, device_id: str, epoch_id: str, frame: VideoFrame
    ) -> None:
        self.frames.append((session_id, device_id, epoch_id, frame.sequence))


async def test_device_id_is_resolved_from_session_started() -> None:
    """VideoFrame and EpochStarted carry no device_id -- it must come from the
    session_started message that precedes them."""
    frames = [
        encode_message(SessionStarted(session_id="sess_1", device_id="glasses-01", started_at=T0)),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
        a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=1),
    ]

    sink = RecordingSink()
    async with replay_server(frames) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        await consumer.run()

    assert sink.epoch_starts == [("sess_1", "glasses-01", "TR_VCaaa")]
    assert sink.frames == [("sess_1", "glasses-01", "TR_VCaaa", 1)]


async def test_epoch_ended_is_forwarded() -> None:
    frames = [
        encode_message(SessionStarted(session_id="sess_1", device_id="glasses-01", started_at=T0)),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
        encode_message(
            EpochEnded(
                session_id="sess_1", epoch_id="TR_VCaaa", ended_at=T0, reason="track_unsubscribed"
            )
        ),
    ]

    sink = RecordingSink()
    async with replay_server(frames) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        await consumer.run()

    assert sink.epoch_ends == [("sess_1", "TR_VCaaa")]


async def test_a_device_id_falls_back_to_session_id_when_session_started_was_missed() -> None:
    """A consumer that connects mid-session (e.g. after a restart) must still
    attribute frames to something rather than silently dropping them."""
    frames = [
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
        a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=1),
    ]

    sink = RecordingSink()
    async with replay_server(frames) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        await consumer.run()

    assert sink.epoch_starts == [("sess_1", "sess_1", "TR_VCaaa")]


async def test_a_session_ending_forgets_its_device_id() -> None:
    frames = [
        encode_message(SessionStarted(session_id="sess_1", device_id="glasses-01", started_at=T0)),
        encode_message(SessionEnded(session_id="sess_1", ended_at=T0, reason="session_deleted")),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
    ]

    sink = RecordingSink()
    async with replay_server(frames) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        await consumer.run()

    # Falls back to session_id -- the mapping was evicted by session_ended.
    assert sink.epoch_starts == [("sess_1", "sess_1", "TR_VCaaa")]


async def test_a_reconnect_produces_a_second_epoch_started_and_a_second_reset() -> None:
    """`MediaClient` re-delivers `epoch_started` after a drop and rejoin. The
    sink must see it again -- that second call is what makes the reconnect a
    reset, matching the media-contract client's own contract.
    """
    frames = [
        encode_message(SessionStarted(session_id="sess_1", device_id="glasses-01", started_at=T0)),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
        a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=1),
    ]

    sink = RecordingSink()
    async with flaky_replay_server(frames, drop_after=2) as url:
        policy = ReconnectPolicy(initial_seconds=0.01, max_seconds=0.02)
        client = MediaClient(url, reconnect=True, policy=policy)
        consumer = RelayConsumer(client, sink)

        try:
            # The replay server resends the whole fixture (including a fresh
            # epoch_started) on every reconnect and never stops offering new
            # connections, so this only terminates by timeout -- which is the
            # point: it proves at least a second epoch_started arrives well
            # within a couple of reconnect cycles.
            await asyncio.wait_for(consumer.run(), timeout=2.0)
        except TimeoutError:
            pass
        finally:
            await client.aclose()

    assert len(sink.epoch_starts) >= 2
    assert all(call == ("sess_1", "glasses-01", "TR_VCaaa") for call in sink.epoch_starts)


# --- A slow sink must lose frames, not accumulate lag ------------------------


class SlowSink(RecordingSink):
    """A sink that cannot keep up, which is the whole point of the queue.

    Blocks on a gate for its first frame, standing in for a detector that takes
    longer per frame than the stream allows.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.first_frame_seen = asyncio.Event()

    async def video_frame(
        self, *, session_id: str, device_id: str, epoch_id: str, frame: VideoFrame
    ) -> None:
        if not self.first_frame_seen.is_set():
            self.first_frame_seen.set()
            await self.release.wait()
        await super().video_frame(
            session_id=session_id, device_id=device_id, epoch_id=epoch_id, frame=frame
        )


def _preamble() -> list[bytes]:
    return [
        encode_message(SessionStarted(session_id="sess_1", device_id="glasses-01", started_at=T0)),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                stream_kind="video",
                track_sid="TR_VCaaa",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
    ]


async def test_a_slow_sink_sees_the_newest_frame_not_a_backlog() -> None:
    """The bug this exists to prevent. Iterating the client and awaiting the
    sink per message consumes the receive buffer in arrival order, so a sink
    slower than the stream falls progressively further behind -- measured at
    over seven seconds and still climbing before the queue existed.

    A stale frame describes a moment that has already gone. Skipping to the
    newest is strictly better than rendering the past.
    """
    messages = [
        *_preamble(),
        *[a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=n) for n in range(1, 9)],
    ]

    sink = SlowSink()
    async with replay_server(messages) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        task = asyncio.ensure_future(consumer.run())

        # Let the reader drain every frame while the sink is stuck on the first.
        await asyncio.wait_for(sink.first_frame_seen.wait(), timeout=2)
        await asyncio.sleep(0.1)
        sink.release.set()
        await asyncio.wait_for(task, timeout=5)

    sequences = [sequence for _, _, _, sequence in sink.frames]
    assert sequences[-1] == 8, "the newest frame is never skipped"
    assert len(sequences) < 8, "the ones it superseded are dropped, not queued"
    assert consumer.frames_dropped == 8 - len(sequences)
    # The replay server delivers instantly, so the reader empties the socket
    # before the dispatcher takes anything -- which is exactly the point: the
    # first frame the sink ever sees is already the newest one available.


async def test_dropping_is_counted_so_it_cannot_go_unnoticed() -> None:
    """`observed_fps` cannot see this failure -- it measures the captured_at
    stamps of frames that *are* processed, and every one was, just late. This
    counter is the one that says the detector is behind the stream."""
    messages = [
        *_preamble(),
        *[a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=n) for n in range(1, 7)],
    ]

    sink = SlowSink()
    async with replay_server(messages) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        task = asyncio.ensure_future(consumer.run())
        await asyncio.wait_for(sink.first_frame_seen.wait(), timeout=2)
        await asyncio.sleep(0.1)
        sink.release.set()
        await asyncio.wait_for(task, timeout=5)

    assert consumer.frames_dropped > 0
    assert consumer.control_dropped == 0, "control messages are never dropped"


async def test_an_epoch_boundary_survives_a_slow_sink() -> None:
    """Frames are droppable; `epoch_started` is not. Losing one would leave
    tracker state unreset against a new epoch, which is the exact trap docs/06
    warns about -- and reordering one behind a frame would be as bad.
    """
    messages = [
        *_preamble(),
        *[a_video_frame(session_id="sess_1", epoch_id="TR_VCaaa", sequence=n) for n in range(1, 5)],
        encode_message(
            EpochEnded(
                session_id="sess_1",
                epoch_id="TR_VCaaa",
                ended_at=T0,
                reason="track_unsubscribed",
            )
        ),
        encode_message(
            EpochStarted(
                session_id="sess_1",
                epoch_id="TR_VCbbb",
                stream_kind="video",
                track_sid="TR_VCbbb",
                participant_identity="glasses-01",
                started_at=T0,
            )
        ),
        *[a_video_frame(session_id="sess_1", epoch_id="TR_VCbbb", sequence=n) for n in range(5, 9)],
    ]

    sink = SlowSink()
    async with replay_server(messages) as url:
        consumer = RelayConsumer(MediaClient(url, reconnect=False), sink)
        task = asyncio.ensure_future(consumer.run())
        await asyncio.wait_for(sink.first_frame_seen.wait(), timeout=2)
        await asyncio.sleep(0.1)
        sink.release.set()
        await asyncio.wait_for(task, timeout=5)

    assert sink.epoch_starts == [
        ("sess_1", "glasses-01", "TR_VCaaa"),
        ("sess_1", "glasses-01", "TR_VCbbb"),
    ], "both epoch boundaries must survive"
    assert sink.epoch_ends == [("sess_1", "TR_VCaaa")]

    # No frame of the new epoch may be delivered before the reset that starts
    # it -- the ordering guarantee docs/12 makes and this queue must preserve.
    epochs_in_order = [epoch for _, _, epoch, _ in sink.frames]
    assert epochs_in_order == sorted(epochs_in_order), (
        "frames must never be reordered across an epoch boundary"
    )
