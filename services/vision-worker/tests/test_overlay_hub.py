"""OverlayHub: fanning per-frame detections out without ever blocking.

The property under test throughout is that `publish` returns immediately and
drops rather than waits. A viewer is a browser tab on someone's laptop, and
`overlay/hub.py` explains why letting one apply backpressure would be worse than
it sounds: the pipeline would stall, and the gateway -- which keeps a single
latest-frame slot per subscriber -- would silently discard video the stability
machine never learns was missing.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    OverlayFrame,
    OverlayTrack,
)

from vision_worker.overlay.hub import OverlayHub

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 8, 7, 12, 0, tzinfo=dt.UTC)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def an_overlay(sequence: int, *, label: str = "keys") -> OverlayFrame:
    return OverlayFrame(
        session_id="sess_1",
        media_epoch_id="TR_VCaaa",
        sequence=sequence,
        captured_at=T0,
        relayed_at=T0,
        emitted_at=T0 + dt.timedelta(milliseconds=40),
        width=1280,
        height=720,
        tracks=(
            OverlayTrack(
                track_id="track-1",
                label=label,
                confidence=0.9,
                box=BoundingBox(x_min=0.4, y_min=0.5, x_max=0.5, y_max=0.6),
                motion_state="at_rest",
            ),
        ),
        pipeline_latency_ms=40.0,
    )


async def test_publishing_with_nobody_watching_is_harmless() -> None:
    """The normal state of a deployed service."""
    hub = OverlayHub(max_subscribers=4)
    hub.publish(an_overlay(1))
    assert hub.subscriber_count == 0
    assert hub.dropped == 0


async def test_a_subscriber_receives_what_was_published() -> None:
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe()
    assert subscriber is not None

    hub.publish(an_overlay(7))
    received = await subscriber.next()

    assert received is not None
    assert received.sequence == 7
    assert received.tracks[0].label == "keys"


async def test_an_unread_overlay_is_displaced_by_a_newer_one() -> None:
    """Latest-wins. A stale overlay describes a frame that has already gone; a
    viewer showing it would be drawing boxes over the wrong picture."""
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe()
    assert subscriber is not None

    hub.publish(an_overlay(1))
    hub.publish(an_overlay(2))
    hub.publish(an_overlay(3))

    received = await subscriber.next()
    assert received is not None
    assert received.sequence == 3, "the viewer should see the newest, not a backlog"
    assert subscriber.dropped == 2
    assert hub.dropped == 2


async def test_publish_never_waits_for_a_viewer_that_is_not_reading() -> None:
    """The whole point: a viewer that never calls `next` must not be able to
    slow frame handling down."""
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe()
    assert subscriber is not None

    for sequence in range(1000):
        hub.publish(an_overlay(sequence))

    # Control never left this coroutine, so nothing could have awaited.
    assert subscriber.sent == 1000
    assert subscriber.dropped == 999


async def test_a_session_scoped_viewer_only_gets_its_session() -> None:
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe(session_id="sess_2")
    assert subscriber is not None

    hub.publish(an_overlay(1))
    assert subscriber.sent == 0

    matching = an_overlay(2).model_copy(update={"session_id": "sess_2"})
    hub.publish(matching)
    assert await subscriber.next() == matching


async def test_every_viewer_gets_the_same_overlay() -> None:
    hub = OverlayHub(max_subscribers=4)
    first = hub.subscribe()
    second = hub.subscribe()
    assert first is not None and second is not None

    hub.publish(an_overlay(5))

    assert (await first.next()) is not None
    assert (await second.next()) is not None


async def test_the_viewer_limit_is_reported_rather_than_raised() -> None:
    """Refusing an extra debugging viewer is an ordinary outcome the endpoint
    turns into a close code, not an error anything unwinds from."""
    hub = OverlayHub(max_subscribers=1)
    assert hub.subscribe() is not None
    assert hub.subscribe() is None


async def test_unsubscribing_frees_a_slot_and_ends_the_send_loop() -> None:
    hub = OverlayHub(max_subscribers=1)
    subscriber = hub.subscribe()
    assert subscriber is not None

    hub.unsubscribe(subscriber)

    assert hub.subscriber_count == 0
    assert hub.subscribe() is not None, "the slot must come back"
    assert await subscriber.next() is None, "a closed subscriber ends its loop"


async def test_publishing_after_unsubscribe_reaches_nobody() -> None:
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe()
    assert subscriber is not None
    hub.unsubscribe(subscriber)

    hub.publish(an_overlay(1))

    assert subscriber.sent == 0


async def test_a_waiting_viewer_wakes_when_an_overlay_arrives() -> None:
    """`next` blocks until there is something to draw, rather than spinning."""
    hub = OverlayHub(max_subscribers=4)
    subscriber = hub.subscribe()
    assert subscriber is not None

    waiter = asyncio.ensure_future(subscriber.next())
    await asyncio.sleep(0)
    assert not waiter.done(), "nothing published yet"

    hub.publish(an_overlay(9))
    received = await asyncio.wait_for(waiter, timeout=1)

    assert received is not None
    assert received.sequence == 9


async def test_close_ends_every_viewer_for_shutdown() -> None:
    hub = OverlayHub(max_subscribers=4)
    first = hub.subscribe()
    second = hub.subscribe()
    assert first is not None and second is not None

    hub.close()

    assert await first.next() is None
    assert await second.next() is None
    assert hub.subscriber_count == 0


async def test_has_subscribers_is_what_the_pipeline_checks() -> None:
    hub = OverlayHub(max_subscribers=4)
    assert hub.has_subscribers is False
    subscriber = hub.subscribe()
    assert subscriber is not None
    assert hub.has_subscribers is True
