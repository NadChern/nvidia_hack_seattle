"""FixtureDetector: the no-GPU, no-model path the `ci` and `dev-macos`
profiles depend on being real."""

from __future__ import annotations

import numpy as np
import pytest
from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D

from vision_worker.detect.base import Detector
from vision_worker.detect.fixture import FixtureDetector

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


A_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def a_detection(label: str = "keys") -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        box=BoundingBox(x_min=0.4, y_min=0.4, x_max=0.5, y_max=0.5),
        centroid=Point2D(x=0.45, y=0.45),
    )


async def test_replays_the_script_in_order() -> None:
    first = (a_detection("keys"),)
    second = (a_detection("wallet"),)
    detector = FixtureDetector([first, second], loop=False)
    await detector.initialize()

    assert await detector.detect(A_FRAME, labels=()) == first
    assert await detector.detect(A_FRAME, labels=()) == second


async def test_loops_by_default() -> None:
    only = (a_detection("keys"),)
    detector = FixtureDetector([only])

    await detector.detect(A_FRAME, labels=())
    looped = await detector.detect(A_FRAME, labels=())

    assert looped == only


async def test_returns_nothing_once_exhausted_when_loop_is_false() -> None:
    detector = FixtureDetector([(a_detection(),)], loop=False)

    await detector.detect(A_FRAME, labels=())
    exhausted = await detector.detect(A_FRAME, labels=())

    assert exhausted == ()


async def test_filters_by_requested_labels() -> None:
    frame = (a_detection("keys"), a_detection("wallet"))
    detector = FixtureDetector([frame])

    result = await detector.detect(A_FRAME, labels=("wallet",))

    assert [d.label for d in result] == ["wallet"]


async def test_empty_labels_means_prompt_free_and_returns_everything() -> None:
    frame = (a_detection("keys"), a_detection("wallet"))
    detector = FixtureDetector([frame])

    result = await detector.detect(A_FRAME, labels=())

    assert result == frame


async def test_readiness_payload_tracks_call_count() -> None:
    detector = FixtureDetector([(a_detection(),)])

    await detector.detect(A_FRAME, labels=())
    await detector.detect(A_FRAME, labels=())

    assert detector.readiness_payload()["calls"] == 2


def test_an_empty_script_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FixtureDetector([])


def test_satisfies_the_detector_protocol() -> None:
    detector: Detector = FixtureDetector([(a_detection(),)])
    assert detector is not None
