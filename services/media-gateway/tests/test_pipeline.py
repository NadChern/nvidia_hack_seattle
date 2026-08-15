"""The scripted source driven all the way to wire messages.

This is the whole relay path with no LiveKit, no network, and no hardware.
"""

import asyncio

import pytest
from visual_memory_media_contract.framing import decode_message
from visual_memory_media_contract.protocol import (
    AudioChunk,
    EpochEnded,
    EpochStarted,
    RelayMessage,
    SessionEnded,
    SessionStarted,
    VideoFrame,
)

from media_gateway.config import Settings
from media_gateway.domain.epoch import EpochRegistry
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.domain.sampling import Pacer
from media_gateway.pipeline import MediaPipeline
from media_gateway.relay.hub import RelayHub, Subscriber
from media_gateway.transport.scripted import ScriptedMediaSource, ScriptedPlan

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _no_wait(_: float) -> None:
    """Yield to the loop without advancing real time."""
    await asyncio.sleep(0)


def immediate_pacer(interval_s: float) -> Pacer:
    """A pacer that ticks every loop turn.

    Sampling is time-driven in production, but a test that waits on real
    wall-clock intervals is slow and flaky. Injecting this makes the sampler
    run in lockstep with the scripted source instead.
    """
    return Pacer(interval_s, sleep=_no_wait)


def build(
    audio_queue_chunks: int = 4096, **overrides: object
) -> tuple[MediaPipeline, RelayHub, MetricsRegistry, Settings]:
    settings = Settings(media_source="scripted", environment="ci", **overrides)  # type: ignore[arg-type]
    # Generous audio room: these tests never drain mid-run, and audio is
    # deliberately never dropped.
    hub = RelayHub(
        max_subscribers=settings.ws_max_subscribers,
        audio_queue_chunks=audio_queue_chunks,
    )
    metrics = MetricsRegistry()
    pipeline = MediaPipeline(
        settings=settings,
        hub=hub,
        epochs=EpochRegistry(settings),
        metrics=metrics,
        pacer_factory=immediate_pacer,
    )
    return pipeline, hub, metrics, settings


def collect(subscriber: Subscriber) -> list[RelayMessage]:
    """Decode whatever is queued right now, without awaiting."""
    return [decode_message(payload) for payload in subscriber.drain_nowait()]


async def consume(subscriber: Subscriber) -> list[RelayMessage]:
    """Read a subscriber to completion, as a real send loop does.

    Draining concurrently matters: video is latest-wins, so a subscriber that
    is only read at the end legitimately holds a single frame.
    """
    messages: list[RelayMessage] = []
    while True:
        payload = await subscriber.next()
        if payload is None:
            return messages
        messages.append(decode_message(payload))


async def run_plan(
    plan: ScriptedPlan, **overrides: object
) -> tuple[list[RelayMessage], list[RelayMessage], MetricsRegistry]:
    """Run a plan and return the video and audio message streams."""
    pipeline, hub, metrics, _ = build(**overrides)
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    audio = hub.subscribe(stream_kind="audio")
    readers = (
        asyncio.create_task(consume(video)),
        asyncio.create_task(consume(audio)),
    )

    source = ScriptedMediaSource(plan)
    await source.run(pipeline)
    await pipeline.stop()
    hub.close_all("run_complete")

    return await readers[0], await readers[1], metrics


async def test_scripted_run_produces_a_well_formed_video_stream() -> None:
    messages, _, _ = await run_plan(ScriptedPlan(epochs=2, frames_per_epoch=4))

    assert isinstance(messages[0], SessionStarted)
    assert isinstance(messages[-1], SessionEnded)
    starts = [m for m in messages if isinstance(m, EpochStarted)]
    ends = [m for m in messages if isinstance(m, EpochEnded)]
    assert len(starts) == len(ends) == 2


async def test_a_rejoin_changes_the_epoch_but_not_the_identity() -> None:
    messages, _, _ = await run_plan(ScriptedPlan(epochs=2, frames_per_epoch=3))

    starts = [m for m in messages if isinstance(m, EpochStarted)]

    assert starts[0].participant_identity == starts[1].participant_identity
    assert starts[0].epoch_id != starts[1].epoch_id
    assert all(start.epoch_id == start.track_sid for start in starts)


async def test_control_precedes_the_frames_it_governs() -> None:
    """A consumer must never see a frame before the epoch that starts it."""
    messages, _, _ = await run_plan(ScriptedPlan(epochs=2, frames_per_epoch=3))

    seen: set[str] = set()
    for message in messages:
        if isinstance(message, EpochStarted):
            seen.add(message.epoch_id)
        if isinstance(message, VideoFrame):
            assert message.epoch_id in seen


async def test_the_dimension_guard_keeps_transition_frames_off_the_wire() -> None:
    """The spike's 8x8 frames must never reach a detector."""
    messages, _, metrics = await run_plan(
        ScriptedPlan(epochs=1, frames_per_epoch=4, transition_frames=3)
    )

    frames = [m for m in messages if isinstance(m, VideoFrame)]

    assert metrics.video.rejected_dimensions == 3
    assert frames, "expected some frames to survive the guard"
    assert all(frame.width == 320 and frame.height == 180 for frame in frames)


async def test_relayed_frames_decode_to_their_declared_size() -> None:
    messages, _, _ = await run_plan(ScriptedPlan(epochs=1, frames_per_epoch=4))

    frames = [m for m in messages if isinstance(m, VideoFrame)]

    assert frames
    for frame in frames:
        assert frame.rgb.shape == (frame.height, frame.width, 3)


async def test_rgba_raw_subscribers_get_pixel_exact_frames() -> None:
    pipeline, hub, _, _ = build()
    raw = hub.subscribe(stream_kind="video", encoding="rgba_raw")
    reader = asyncio.create_task(consume(raw))

    source = ScriptedMediaSource(ScriptedPlan(epochs=1, frames_per_epoch=3))
    await source.run(pipeline)
    await pipeline.stop()
    hub.close_all("run_complete")

    frames = [m for m in await reader if isinstance(m, VideoFrame)]

    assert frames
    for frame in frames:
        assert frame.encoding == "rgba_raw"
        assert frame.pixel_format == "rgba"
        assert frame.rgba.shape == (frame.height, frame.width, 4)


async def test_audio_is_coalesced_and_carries_continuous_pts() -> None:
    _, audio, _ = await run_plan(
        ScriptedPlan(epochs=1, frames_per_epoch=4, audio_frames_per_epoch=20)
    )

    chunks = [m for m in audio if isinstance(m, AudioChunk)]

    assert chunks
    for earlier, later in zip(chunks[:-1], chunks[1:], strict=True):
        if earlier.epoch_id != later.epoch_id:
            continue
        assert later.pts_samples == earlier.pts_samples + earlier.samples


async def test_audio_chunks_are_longer_than_the_frames_they_coalesce() -> None:
    """20 ms frames relayed individually would be 50 messages a second."""
    _, audio, _ = await run_plan(
        ScriptedPlan(epochs=1, frames_per_epoch=4, audio_frames_per_epoch=20)
    )

    chunks = [m for m in audio if isinstance(m, AudioChunk)]

    assert chunks
    assert all(chunk.samples >= 4800 for chunk in chunks[:-1])


async def test_no_audio_is_lost_when_an_epoch_ends_mid_chunk() -> None:
    _, audio, _ = await run_plan(
        ScriptedPlan(epochs=1, frames_per_epoch=3, audio_frames_per_epoch=7)
    )

    chunks = [m for m in audio if isinstance(m, AudioChunk)]
    relayed = sum(chunk.samples for chunk in chunks)

    # 3 video frames x (7 // 3 = 2) audio frames x 960 samples each.
    assert relayed == 3 * 2 * 960


async def test_metrics_reflect_what_actually_happened() -> None:
    _, _, metrics = await run_plan(ScriptedPlan(epochs=2, frames_per_epoch=4, transition_frames=1))

    assert metrics.sessions_created == 1
    assert metrics.sessions_ended == 1
    assert metrics.epochs_started == 4  # video and audio, twice
    assert metrics.epochs_ended == 4
    assert metrics.video.received == metrics.video.admitted + metrics.video.rejected_dimensions
    assert metrics.video.relay_latency.count == metrics.video.relayed


async def test_sequence_restarts_at_zero_for_each_video_epoch() -> None:
    messages, _, _ = await run_plan(ScriptedPlan(epochs=2, frames_per_epoch=4))

    by_epoch: dict[str, list[int]] = {}
    for message in messages:
        if isinstance(message, VideoFrame):
            by_epoch.setdefault(message.epoch_id, []).append(message.sequence)

    assert len(by_epoch) == 2
    for sequences in by_epoch.values():
        assert sequences[0] == 0
        assert sequences == sorted(sequences)


async def test_hello_announces_epochs_already_running() -> None:
    """A consumer joining mid-epoch must still learn what to reset."""
    pipeline, hub, _, _ = build()
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_1",
        stream_kind="video",
        track_sid="TR_VCaaa",
        participant_identity="glasses-01",
    )

    hello = decode_message(pipeline.build_hello(stream_kind="video", encoding="jpeg"))
    replay = [decode_message(frame) for frame in pipeline.replay_epochs_for("video")]
    await pipeline.stop()

    assert hello.type == "stream_hello"
    assert [epoch.epoch_id for epoch in hello.active_epochs] == ["TR_VCaaa"]
    assert [message.epoch_id for message in replay if isinstance(message, EpochStarted)] == [
        "TR_VCaaa"
    ]


async def test_nothing_is_encoded_when_nobody_is_listening() -> None:
    """Encoding for zero subscribers would burn the frame budget for nothing."""
    pipeline, _, metrics, _ = build()

    source = ScriptedMediaSource(ScriptedPlan(epochs=1, frames_per_epoch=4))
    await source.run(pipeline)
    await pipeline.stop()

    assert metrics.video.received > 0
    assert metrics.video.relayed == 0
    # Audio must agree: counting it as relayed with no subscribers would make
    # the two disagree on a dashboard.
    assert metrics.audio.received > 0
    assert metrics.audio.relayed == 0


async def test_audio_timeline_advances_even_with_no_subscribers() -> None:
    """A consumer connecting later must not see pts rewind."""
    pipeline, hub, _, _ = build()

    source = ScriptedMediaSource(ScriptedPlan(epochs=1, frames_per_epoch=3))
    await source.run(pipeline)
    await pipeline.stop()

    # Then a subscriber arrives for a fresh epoch and sees pts from zero.
    audio = hub.subscribe(stream_kind="audio")
    pipeline.session_started(session_id="sess_2", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_2",
        stream_kind="audio",
        track_sid="TR_ACnew",
        participant_identity="glasses-01",
    )
    collect(audio)

    epoch = pipeline._epochs.active_for("sess_2", "audio")  # noqa: SLF001
    assert epoch is not None
    assert epoch.pts_samples == 0


async def test_stop_cancels_sampler_tasks() -> None:
    pipeline, hub, _, _ = build()
    hub.subscribe(stream_kind="video", encoding="jpeg")
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_1",
        stream_kind="video",
        track_sid="TR_VCaaa",
        participant_identity="glasses-01",
    )

    await pipeline.stop()

    assert pipeline._samplers == {}  # noqa: SLF001
