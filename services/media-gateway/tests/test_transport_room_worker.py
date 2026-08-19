"""The publisher's clock, mapped onto ours.

`captured_at` and `received_at` were the same value until this change, which
made the device->gateway hop unmeasurable -- the frame-budget probe reported a
flat 0.0 ms for it and we spent a session unable to tell a slow camera from a
bursty link. These cover the mapping that separates them.
"""

from __future__ import annotations

import datetime as dt


class _Event:
    """Minimal stand-in for `rtc.VideoFrameEvent`."""

    def __init__(self, timestamp_us: int | None) -> None:
        self.timestamp_us = timestamp_us


def test_sender_clock_preserves_spacing_not_absolute_time() -> None:
    """The publisher's clock is anchored to ours, so gaps survive the mapping.

    This is the whole diagnostic: evenly produced frames must stay evenly
    spaced in `captured_at` even when they arrive in clumps, which is what
    separates a slow camera from a bursty link.
    """
    from media_gateway.transport.room_worker import _SenderClock

    clock = _SenderClock("sess")
    # Sender is 50 seconds behind us and emits exactly every 100 ms.
    base_us = 1_000_000_000
    received = dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=dt.UTC)

    first = clock.stamp(_Event(base_us), received)
    # Second frame produced 100 ms later but delivered 900 ms later (a clump).
    second = clock.stamp(_Event(base_us + 100_000), received + dt.timedelta(milliseconds=900))

    assert first == received
    assert (second - first) == dt.timedelta(milliseconds=100)


def test_sender_clock_falls_back_when_transport_has_no_timestamp() -> None:
    """No timestamp must degrade to our clock, not to a crash or a zero."""
    from media_gateway.transport.room_worker import _SenderClock

    clock = _SenderClock("sess")
    received = dt.datetime(2026, 8, 18, 12, 0, 0, tzinfo=dt.UTC)
    assert clock.stamp(_Event(None), received) == received
    assert clock.stamp(_Event(0), received) == received
