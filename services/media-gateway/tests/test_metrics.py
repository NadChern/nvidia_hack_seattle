"""Counters and bounded latency sampling."""

import pytest

from media_gateway.domain.metrics import LatencyReservoir, MetricsRegistry


def test_empty_reservoir_reports_no_percentiles() -> None:
    reservoir = LatencyReservoir()

    snapshot = reservoir.snapshot()

    assert snapshot["count"] == 0
    assert snapshot["p50_ms"] is None
    assert snapshot["p95_ms"] is None


def test_percentiles_are_reported_in_milliseconds() -> None:
    reservoir = LatencyReservoir()
    for value in range(1, 101):
        reservoir.observe(value / 1000)

    snapshot = reservoir.snapshot()

    assert snapshot["p50_ms"] == pytest.approx(50, abs=1)
    assert snapshot["p95_ms"] == pytest.approx(96, abs=1)
    assert snapshot["max_ms"] == pytest.approx(100)


def test_reservoir_is_bounded_but_counts_everything() -> None:
    """Memory must not grow over a long session, but totals stay honest."""
    reservoir = LatencyReservoir(capacity=10)
    for value in range(1000):
        reservoir.observe(value / 1000)

    snapshot = reservoir.snapshot()

    assert snapshot["count"] == 1000
    assert snapshot["retained"] == 10
    assert snapshot["max_ms"] == pytest.approx(999)


def test_reservoir_rejects_a_useless_capacity() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        LatencyReservoir(capacity=0)


def test_registry_routes_by_stream_kind() -> None:
    registry = MetricsRegistry()

    registry.for_stream("video").received += 3
    registry.for_stream("audio").received += 5

    assert registry.video.received == 3
    assert registry.audio.received == 5


def test_snapshot_shape_is_stable() -> None:
    """The status dashboard reads this; the shape is a contract."""
    registry = MetricsRegistry()
    registry.video.dropped_before_sampling = 7
    registry.audio.subscribers_closed_for_backpressure = 1

    snapshot = registry.snapshot()

    assert set(snapshot) == {
        "sessions",
        "epochs",
        "tokens_issued",
        "lifecycle_signals_emitted",
        "video",
        "audio",
    }
    assert snapshot["video"]["dropped_before_sampling"] == 7
    assert snapshot["audio"]["subscribers_closed_for_backpressure"] == 1
    assert set(snapshot["sessions"]) == {"created", "ended", "expired"}
