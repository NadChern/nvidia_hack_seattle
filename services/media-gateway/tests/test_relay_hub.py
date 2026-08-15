"""Relay fan-out and the two deliberately different backpressure policies."""

import pytest

from media_gateway.errors import CapacityError
from media_gateway.relay.hub import AUDIO_BACKPRESSURE, CONTROL_BACKLOG, RelayHub, Subscriber


def a_hub(*, max_subscribers: int = 8, audio_queue_chunks: int = 3) -> RelayHub:
    return RelayHub(max_subscribers=max_subscribers, audio_queue_chunks=audio_queue_chunks)


def drain(subscriber: Subscriber) -> list[bytes]:
    return subscriber.drain_nowait()


def test_video_subscriber_keeps_only_the_newest_frame() -> None:
    """A slow vision consumer must lose stale frames, not the newest."""
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    for index in range(5):
        hub.publish_video({"jpeg": f"frame-{index}".encode()})

    assert drain(subscriber) == [b"frame-4"]
    assert subscriber.dropped == 4


def test_control_messages_survive_frame_eviction() -> None:
    """The invariant docs/12 guarantees: an epoch_started is never displaced.

    A single size-one queue cannot hold both, so eviction targets the pending
    frame specifically. Without this a consumer could process frames from a new
    epoch while still holding the previous epoch's tracker state.
    """
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    hub.publish_control(b"epoch-started", "video")
    for index in range(5):
        hub.publish_video({"jpeg": f"frame-{index}".encode()})
    hub.publish_control(b"epoch-ended", "video")

    assert drain(subscriber) == [b"epoch-started", b"frame-4", b"epoch-ended"]
    assert subscriber.close_reason is None


def test_many_control_messages_do_not_close_a_video_subscriber() -> None:
    """Regression: control used to exhaust a capacity-one video queue."""
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    for index in range(10):
        hub.publish_control(f"control-{index}".encode(), "video")

    assert subscriber.close_reason is None
    assert len(drain(subscriber)) == 10


def test_a_hopelessly_stalled_video_consumer_is_eventually_closed() -> None:
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    for index in range(CONTROL_BACKLOG + 5):
        hub.publish_control(f"control-{index}".encode(), "video")

    assert subscriber.close_reason == AUDIO_BACKPRESSURE


def test_video_fan_out_reaches_every_subscriber() -> None:
    hub = a_hub()
    first = hub.subscribe(stream_kind="video", encoding="jpeg")
    second = hub.subscribe(stream_kind="video", encoding="jpeg")

    assert hub.publish_video({"jpeg": b"frame"}) == 2
    assert drain(first) == [b"frame"]
    assert drain(second) == [b"frame"]


def test_only_requested_encodings_are_reported() -> None:
    """The pipeline encodes once per encoding actually wanted."""
    hub = a_hub()
    hub.subscribe(stream_kind="video", encoding="jpeg")

    assert hub.required_video_encodings() == {"jpeg"}

    hub.subscribe(stream_kind="video", encoding="rgba_raw")
    assert hub.required_video_encodings() == {"jpeg", "rgba_raw"}


def test_subscribers_receive_only_their_requested_encoding() -> None:
    hub = a_hub()
    jpeg = hub.subscribe(stream_kind="video", encoding="jpeg")
    raw = hub.subscribe(stream_kind="video", encoding="rgba_raw")

    hub.publish_video({"jpeg": b"as-jpeg", "rgba_raw": b"as-raw"})

    assert drain(jpeg) == [b"as-jpeg"]
    assert drain(raw) == [b"as-raw"]


def test_audio_is_buffered_not_dropped() -> None:
    hub = a_hub(audio_queue_chunks=3)
    subscriber = hub.subscribe(stream_kind="audio")

    for index in range(3):
        hub.publish_audio(f"chunk-{index}".encode())

    assert drain(subscriber) == [b"chunk-0", b"chunk-1", b"chunk-2"]
    assert subscriber.dropped == 0


def test_slow_audio_subscriber_is_closed_rather_than_silently_starved() -> None:
    """Losing audio corrupts transcription invisibly; closing is noticed."""
    hub = a_hub(audio_queue_chunks=2)
    subscriber = hub.subscribe(stream_kind="audio")

    hub.publish_audio(b"chunk-0")
    hub.publish_audio(b"chunk-1")
    delivered = hub.publish_audio(b"chunk-2")

    assert delivered == 0
    assert subscriber.close_reason == AUDIO_BACKPRESSURE
    assert subscriber.dropped == 0, "audio must never be counted as dropped"


def test_a_closed_subscriber_accepts_nothing_further() -> None:
    hub = a_hub(audio_queue_chunks=1)
    subscriber = hub.subscribe(stream_kind="audio")
    hub.publish_audio(b"chunk-0")
    hub.publish_audio(b"chunk-1")

    assert subscriber.close_reason == AUDIO_BACKPRESSURE
    assert hub.publish_audio(b"chunk-2") == 0


def test_streams_are_isolated() -> None:
    hub = a_hub()
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    audio = hub.subscribe(stream_kind="audio")

    hub.publish_video({"jpeg": b"frame"})
    hub.publish_audio(b"chunk")

    assert drain(video) == [b"frame"]
    assert drain(audio) == [b"chunk"]


def test_subscriber_limit_is_enforced() -> None:
    hub = a_hub(max_subscribers=1)
    hub.subscribe(stream_kind="video", encoding="jpeg")

    with pytest.raises(CapacityError, match="too many relay subscribers"):
        hub.subscribe(stream_kind="video", encoding="jpeg")


def test_unsubscribing_frees_a_slot() -> None:
    hub = a_hub(max_subscribers=1)
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    hub.unsubscribe(subscriber)

    assert len(hub) == 0
    assert hub.subscribe(stream_kind="video", encoding="jpeg") is not subscriber


def test_publishing_with_no_subscribers_is_harmless() -> None:
    hub = a_hub()

    assert hub.publish_video({"jpeg": b"frame"}) == 0
    assert hub.publish_audio(b"chunk") == 0


@pytest.mark.anyio
async def test_next_awaits_until_a_frame_arrives() -> None:
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="video", encoding="jpeg")

    hub.publish_video({"jpeg": b"frame"})

    assert await subscriber.next() == b"frame"


@pytest.mark.anyio
async def test_next_returns_none_once_closed() -> None:
    """The send loop must be woken rather than left waiting forever."""
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="audio")

    hub.close_all("gateway_shutdown")

    assert await subscriber.next() is None
    assert subscriber.close_reason == "gateway_shutdown"


@pytest.mark.anyio
async def test_closing_still_delivers_what_was_queued() -> None:
    """A clean shutdown must explain itself.

    Discarding the queue on close would drop the `epoch_ended` and
    `session_ended` that tell a consumer why the stream stopped, turning a
    graceful shutdown into an unexplained disconnect.
    """
    hub = a_hub()
    subscriber = hub.subscribe(stream_kind="audio")
    hub.publish_audio(b"chunk")
    hub.publish_control(b"session-ended", "audio")

    hub.close_all("gateway_shutdown")

    assert await subscriber.next() == b"chunk"
    assert await subscriber.next() == b"session-ended"
    assert await subscriber.next() is None


def test_close_all_marks_every_subscriber() -> None:
    hub = a_hub()
    video = hub.subscribe(stream_kind="video", encoding="jpeg")
    audio = hub.subscribe(stream_kind="audio")

    hub.close_all("gateway_shutdown")

    assert video.close_reason == "gateway_shutdown"
    assert audio.close_reason == "gateway_shutdown"
