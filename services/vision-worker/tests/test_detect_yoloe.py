"""YoloeDetector: the real detector, exercised against the real checkpoint.

Opt-in and marked `models` (see pyproject.toml's `addopts`) because this
needs `uv sync --extra models` and a downloaded checkpoint neither `ci` nor
`dev-macos` has -- this suite is what proves the port works, not something
the standards CI loop can run. Both the text and prompt-free slots point at
the same checkpoint file: this test exercises our own wiring (readiness,
postprocessing, the embedding cache, lifecycle), not YOLOE's prompt-free
weights specifically, and pointing both at one already-cached file means
running this test never triggers a second download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.models]

_CHECKPOINT = Path(__file__).resolve().parent.parent / "yoloe-26l-seg.pt"

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
if not _CHECKPOINT.exists():
    pytest.skip(
        f"{_CHECKPOINT.name} is not downloaded; see README.md for how to fetch it",
        allow_module_level=True,
    )

from vision_worker.detect.yoloe import YoloeDetector  # noqa: E402

A_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def detector():
    detector = YoloeDetector(
        text_model=_CHECKPOINT.name,
        prompt_free_model=_CHECKPOINT.name,
        score_threshold=0.01,
    )
    await detector.initialize()
    try:
        yield detector
    finally:
        await detector.aclose()


async def test_initialize_is_idempotent(detector: YoloeDetector) -> None:
    assert detector.is_ready
    await detector.initialize()  # must not reload
    assert detector.is_ready


async def test_readiness_payload_reports_the_loaded_checkpoints(
    detector: YoloeDetector,
) -> None:
    payload = detector.readiness_payload()
    assert payload["ready"] is True
    assert payload["text_model"] == _CHECKPOINT.name
    assert payload["prompt_free_model"] == _CHECKPOINT.name


async def test_detect_on_a_blank_frame_finds_nothing(detector: YoloeDetector) -> None:
    detections = await detector.detect(A_FRAME, labels=("keys",))
    assert detections == ()


async def test_detect_with_no_labels_uses_the_prompt_free_model(
    detector: YoloeDetector,
) -> None:
    detections = await detector.detect(A_FRAME, labels=())
    assert detections == ()


async def test_the_class_embedding_cache_is_reused_across_calls(
    detector: YoloeDetector,
) -> None:
    await detector.detect(A_FRAME, labels=("keys",))
    size_after_first = detector.readiness_payload()["embedding_cache_size"]

    await detector.detect(A_FRAME, labels=("keys",))
    size_after_second = detector.readiness_payload()["embedding_cache_size"]

    assert size_after_second == size_after_first


async def test_request_count_and_latency_are_tracked(detector: YoloeDetector) -> None:
    await detector.detect(A_FRAME, labels=("keys",))
    await detector.detect(A_FRAME, labels=("keys",))

    payload = detector.readiness_payload()
    assert payload["request_count"] == 2
    assert payload["average_latency_ms"] >= 0.0


async def test_aclose_drops_readiness(detector: YoloeDetector) -> None:
    await detector.aclose()
    assert detector.is_ready is False
