"""EvidenceRing: a bounded, time-scoped buffer of already-sampled frames."""

from __future__ import annotations

import datetime as dt

from vision_worker.evidence.ring import BufferedFrame, EvidenceRing

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)


def a_frame(offset_seconds: float, *, width: int = 64, height: int = 48) -> BufferedFrame:
    return BufferedFrame(
        captured_at=T0 + dt.timedelta(seconds=offset_seconds),
        payload=f"frame-at-{offset_seconds}".encode(),
        width=width,
        height=height,
    )


def test_pushed_frames_are_retrievable_by_window() -> None:
    ring = EvidenceRing(dt.timedelta(seconds=60))
    ring.push(a_frame(0))
    ring.push(a_frame(1))
    ring.push(a_frame(2))

    window = ring.window(started_at=T0, ended_at=T0 + dt.timedelta(seconds=2))

    assert len(window) == 3


def test_a_window_excludes_frames_outside_its_range() -> None:
    ring = EvidenceRing(dt.timedelta(seconds=60))
    ring.push(a_frame(0))
    ring.push(a_frame(5))
    ring.push(a_frame(10))

    window = ring.window(
        started_at=T0 + dt.timedelta(seconds=4), ended_at=T0 + dt.timedelta(seconds=6)
    )

    assert len(window) == 1
    assert window[0].payload == b"frame-at-5"


def test_frames_older_than_max_duration_are_evicted() -> None:
    ring = EvidenceRing(dt.timedelta(seconds=5))
    ring.push(a_frame(0))
    ring.push(a_frame(1))

    # A frame far enough ahead that the first two fall outside the window.
    ring.push(a_frame(10))

    assert len(ring) == 1


def test_reset_drops_everything() -> None:
    ring = EvidenceRing(dt.timedelta(seconds=60))
    ring.push(a_frame(0))
    ring.push(a_frame(1))

    ring.reset()

    assert len(ring) == 0
    assert ring.window(started_at=T0, ended_at=T0 + dt.timedelta(minutes=1)) == ()


def test_max_duration_is_exposed_for_status_reporting() -> None:
    ring = EvidenceRing(dt.timedelta(seconds=8))

    assert ring.max_duration == dt.timedelta(seconds=8)
