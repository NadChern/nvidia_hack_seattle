"""Sanity checks on the golden scenarios themselves.

Behavioral correctness -- that each scenario actually produces its
documented stability outcome -- is verified in `services/vision-worker`,
which is the only place `domain.stability` exists; this package stays free
of it by design (see `protocol.py`'s module docstring). What belongs here is
schema sanity: every scenario is well-formed, deterministic, and internally
consistent.
"""

from __future__ import annotations

import pytest

from visual_memory_vision_contract import fixtures
from visual_memory_vision_contract.protocol import TrackSample


def test_every_scenario_is_non_empty() -> None:
    for name, build in fixtures.SCENARIOS.items():
        assert build(), f"scenario {name!r} produced no samples"


def test_every_scenario_uses_one_track_id_except_two_similar_objects() -> None:
    for name, build in fixtures.SCENARIOS.items():
        track_ids = {sample.track_id for sample in build()}
        if name == "two_similar_objects":
            assert track_ids == {"track-1", "track-2"}
        else:
            assert len(track_ids) == 1, f"scenario {name!r} unexpectedly spans {track_ids}"


def test_frame_index_is_non_decreasing_within_each_track() -> None:
    for name, build in fixtures.SCENARIOS.items():
        by_track: dict[str, list[TrackSample]] = {}
        for sample in build():
            by_track.setdefault(sample.track_id, []).append(sample)
        for track_id, samples in by_track.items():
            indices = [sample.frame_index for sample in samples]
            assert indices == sorted(indices), f"{name!r}/{track_id} frame_index is not ordered"


def test_captured_at_matches_frame_index_at_the_fixed_frame_interval() -> None:
    for name, build in fixtures.SCENARIOS.items():
        for sample in build():
            expected = fixtures.T0 + sample.frame_index * fixtures.FRAME_INTERVAL
            assert sample.captured_at == expected, f"scenario {name!r} has a mistimed sample"


def test_scenarios_are_deterministic_across_calls() -> None:
    for build in fixtures.SCENARIOS.values():
        assert build() == build()


def test_scenario_lookup_fails_loudly_on_a_typo() -> None:
    with pytest.raises(KeyError, match="unknown scenario"):
        fixtures.scenario("keys_palced_on_table")


def test_the_reconnect_scenario_returns_two_independent_sequences_sharing_a_track_id() -> None:
    before, after = fixtures.reconnect_reuses_a_track_id()

    assert before and after
    assert {sample.track_id for sample in before} == {"track-1"}
    assert {sample.track_id for sample in after} == {"track-1"}
    # Different physical objects wearing the same track_id -- the whole
    # point of the scenario.
    assert before[0].detection.label != after[0].detection.label
