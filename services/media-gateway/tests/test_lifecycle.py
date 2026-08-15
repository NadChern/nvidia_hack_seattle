"""Lifecycle signals: emission, ordering, and delivery to Memory.

The gateway reports that a track or a session went away. It never names an
object, because it has never run a detector -- Memory turns an epoch-scoped
signal into per-object transitions.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from visual_memory_media_contract.framing import decode_message
from visual_memory_media_contract.protocol import (
    EpochEnded,
    LifecycleEnvelope,
    LifecycleSignal,
    SessionEnded,
)

from media_gateway.config import Settings
from media_gateway.domain import lifecycle
from media_gateway.domain.epoch import EpochRegistry
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.pipeline import MediaPipeline
from media_gateway.relay.hub import RelayHub
from media_gateway.transport.memory_sink import MemorySink

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecordingSink:
    """Stands in for MemorySink without any HTTP."""

    def __init__(self) -> None:
        self.envelopes: list[LifecycleEnvelope] = []

    def emit(self, envelope: LifecycleEnvelope) -> None:
        self.envelopes.append(envelope)


def build() -> tuple[MediaPipeline, RelayHub, RecordingSink]:
    settings = Settings(environment="ci", media_source="scripted")
    hub = RelayHub(max_subscribers=4, audio_queue_chunks=64)
    sink = RecordingSink()
    pipeline = MediaPipeline(
        settings=settings,
        hub=hub,
        epochs=EpochRegistry(settings),
        metrics=MetricsRegistry(),
        lifecycle_sink=sink,
    )
    return pipeline, hub, sink


def messages(subscriber: object) -> list[object]:
    return [decode_message(payload) for payload in subscriber.drain_nowait()]  # type: ignore[attr-defined]


# --- The envelope ---------------------------------------------------------


async def test_a_track_ending_is_scoped_by_epoch_and_names_no_object() -> None:
    """The gateway cannot know objects, so it must not pretend to."""
    pipeline, _, sink = build()
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_1",
        stream_kind="video",
        track_sid="TR_VCaaa",
        participant_identity="glasses-01",
    )

    pipeline.epoch_ended(session_id="sess_1", stream_kind="video", reason="track_unsubscribed")
    await pipeline.stop()

    envelope = next(e for e in sink.envelopes if e.signal.action == "track_lost")
    assert envelope.scope.media_epoch_id == "TR_VCaaa"
    assert envelope.scope.object_id is None
    assert envelope.signal.reason == "track_unsubscribed"


async def test_a_session_ending_is_scoped_to_the_whole_session() -> None:
    """No epoch in the scope, so it reaches every in-transit object."""
    pipeline, _, sink = build()
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")

    pipeline.session_ended(session_id="sess_1", reason="session_deleted")
    await pipeline.stop()

    envelope = next(e for e in sink.envelopes if e.signal.action == "session_ended")
    assert envelope.scope.media_epoch_id is None
    assert envelope.scope.object_id is None
    assert envelope.device_id == "glasses-01"


async def test_the_idempotency_key_is_deterministic() -> None:
    """A gateway that restarts mid-teardown must not double-apply."""
    first = lifecycle.track_lost(
        session_id="sess_1",
        device_id="glasses-01",
        media_epoch_id="TR_VCaaa",
        reason="track_unsubscribed",
        occurred_at=dt.datetime.now(dt.UTC),
    )
    second = lifecycle.track_lost(
        session_id="sess_1",
        device_id="glasses-01",
        media_epoch_id="TR_VCaaa",
        reason="room_disconnected",
        occurred_at=dt.datetime.now(dt.UTC),
    )

    # Same scope and action, so the same key -- even though the reason and the
    # signal id differ. Memory applies it once.
    assert first.idempotency_key == second.idempotency_key
    assert first.signal_id != second.signal_id


# --- Ordering on the relay ------------------------------------------------


async def test_the_signal_precedes_the_terminal_message() -> None:
    """`epoch_ended` stays the last word on an epoch.

    A consumer that stops reading at the terminal message must not have missed
    the explanation for it.
    """
    pipeline, hub, _ = build()
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_1",
        stream_kind="video",
        track_sid="TR_VCaaa",
        participant_identity="glasses-01",
    )
    pipeline.epoch_ended(session_id="sess_1", stream_kind="video", reason="track_unsubscribed")
    await pipeline.stop()

    kinds = [type(m) for m in messages(video)]
    assert kinds.index(LifecycleSignal) < kinds.index(EpochEnded)


async def test_session_ended_is_the_last_message() -> None:
    pipeline, hub, _ = build()
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.session_ended(session_id="sess_1", reason="session_deleted")
    await pipeline.stop()

    assert isinstance(messages(video)[-1], SessionEnded)


async def test_the_relay_copy_and_the_posted_copy_are_the_same_envelope() -> None:
    """A consumer watching the relay and Memory must agree on what happened."""
    pipeline, hub, sink = build()
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    pipeline.session_started(session_id="sess_1", device_id="glasses-01")
    pipeline.epoch_started(
        session_id="sess_1",
        stream_kind="video",
        track_sid="TR_VCaaa",
        participant_identity="glasses-01",
    )
    pipeline.epoch_ended(session_id="sess_1", stream_kind="video", reason="track_unsubscribed")
    await pipeline.stop()

    relayed = next(m for m in messages(video) if isinstance(m, LifecycleSignal))
    posted = next(e for e in sink.envelopes if e.signal.action == "track_lost")

    # Serialized, not as objects: the relay copy has been through the wire and
    # carries millisecond timestamps, while the posted one is still in memory
    # with microseconds. Both reach their consumer as JSON through the same
    # serializer, so this compares what each side actually receives.
    assert relayed.envelope.model_dump(mode="json") == posted.model_dump(mode="json")


# --- The sink -------------------------------------------------------------


async def test_the_sink_is_disabled_without_a_url() -> None:
    """The gateway is fully useful with no Memory Service."""
    sink = MemorySink(Settings(environment="ci", media_source="scripted"))
    await sink.start()

    assert sink.enabled is False
    # Emitting is a no-op rather than an error.
    sink.emit(
        lifecycle.session_ended(
            session_id="sess_1",
            device_id="glasses-01",
            reason="gateway_shutdown",
            occurred_at=dt.datetime.now(dt.UTC),
        )
    )
    await sink.stop()
    assert sink.snapshot()["delivered"] == 0


async def test_emitting_never_blocks_when_memory_is_unreachable() -> None:
    """A dead Memory must not become a media outage.

    The sink points at a port nothing is listening on. Emission still returns
    immediately, and the failure is counted rather than raised.
    """
    settings = Settings(
        environment="ci",
        media_source="scripted",
        lifecycle_sink_url="http://127.0.0.1:9/v1/lifecycle",
        lifecycle_sink_timeout_s=0.2,
    )
    sink = MemorySink(settings)
    await sink.start()

    envelope = lifecycle.session_ended(
        session_id="sess_1",
        device_id="glasses-01",
        reason="gateway_shutdown",
        occurred_at=dt.datetime.now(dt.UTC),
    )
    async with asyncio.timeout(1.0):
        sink.emit(envelope)

    await sink.stop()
    assert sink.snapshot()["failed"] >= 1
    assert sink.snapshot()["delivered"] == 0
