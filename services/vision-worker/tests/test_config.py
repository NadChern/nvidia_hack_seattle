"""Settings invariants that keep the reasoning window coherent.

The placement-detection window (spike 5b, gate item 6) must be wide enough to
show a placement's before/after contrast, which couples it to two other knobs:
it is served from the evidence ring, and a wider window overlaps its neighbours.
These invariants keep an env override from silently breaking either coupling.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vision_worker.config import Settings


def _settings(**kwargs: object) -> Settings:
    return Settings(environment="ci", **kwargs)  # type: ignore[arg-type]


def test_default_window_fits_the_ring_and_is_covered_by_the_cooldown() -> None:
    settings = _settings()
    # Wide enough to straddle the placement transition, but the ring must be
    # able to supply it and the cooldown must cover a full overlapping span.
    assert settings.reason_window_seconds <= settings.evidence_ring_seconds
    assert settings.event_cooldown_seconds >= settings.reason_window_seconds


def test_a_window_wider_than_the_ring_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reason_window_seconds cannot exceed"):
        _settings(reason_window_seconds=40.0, evidence_ring_seconds=30.0)


def test_a_cooldown_shorter_than_the_window_is_rejected() -> None:
    # One placement would then be rewritten once per interval for the span it
    # lingers in overlapping windows.
    with pytest.raises(ValidationError, match="event_cooldown_seconds cannot be below"):
        _settings(reason_window_seconds=20.0, event_cooldown_seconds=10.0)
