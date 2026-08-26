"""The reasoner pipeline: window in, matched observation out.

No model and no memory service run here. The reasoner is scripted
(`FixtureReasoner`), the identity gate is a stub gallery that returns a chosen
score, and `on_confirmed` just records. What is under test is the pipeline's
own logic: it schedules a window on a cadence, it writes only events that match
a registered object (the write gate), it drops repeats within the cooldown, and
it never promotes a `nothing_happened`.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Sequence
from dataclasses import replace

import pytest
from PIL import Image
from visual_memory_media_contract.framing import payload_digest
from visual_memory_media_contract.protocol import VideoFrame
from visual_memory_vision_contract.protocol import BoundingBox, CandidateEvent, VerifierResult

from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.identity.fixture import FixtureEmbedder
from vision_worker.identity.gallery import GalleryScore
from vision_worker.pipeline import Pipeline
from vision_worker.reason.base import WindowEvent
from vision_worker.reason.fixture import FixtureReasoner

pytestmark = pytest.mark.anyio

T0 = dt.datetime(2026, 8, 15, 12, 0, 0, tzinfo=dt.UTC)
_W, _H = 64, 48


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _jpeg(color: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    image = Image.new("RGB", (_W, _H), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _frame(offset_s: float, *, sequence: int) -> VideoFrame:
    payload = _jpeg()
    at = T0 + dt.timedelta(seconds=offset_s)
    return VideoFrame(
        session_id="sess_1",
        epoch_id="TR_a",
        sequence=sequence,
        captured_at=at,
        received_at=at,
        relayed_at=at,
        width=_W,
        height=_H,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
    ).attach_payload(payload)


def _event(
    action: str, *, label: str = "keys", location: str | None = "on the table"
) -> WindowEvent:
    return WindowEvent(
        label=label,
        box=BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8),
        action=action,
        location_description=location,
        confidence=0.85,
    )


class StubGallery:
    """Stands in for `GalleryCache`: fixed labels and a chosen match score."""

    def __init__(
        self,
        *,
        labels: set[str],
        score: GalleryScore | None,
        override_threshold: float | None = None,
    ) -> None:
        self._labels = frozenset(labels)
        self._score = score
        self._override_threshold = override_threshold
        self.refreshes = 0

    @property
    def labels(self) -> frozenset[str]:
        return self._labels

    async def refresh(self, *, force: bool = False) -> bool:
        self.refreshes += 1
        return False

    def match(
        self,
        queries: Sequence[object],
        *,
        label: str,
        summary_weight: float,
        floor: float = 0.0,
        confusion_margin: float = 0.0,
    ) -> GalleryScore | None:
        # A lone-object stand-in: its bar is just the floor, so the pipeline's
        # per-object gate reduces to the old global-cosine comparison -- unless a
        # test pins a raised bar to model a confusable same-label sibling.
        if self._score is None:
            return None
        return replace(
            self._score,
            threshold=floor if self._override_threshold is None else self._override_threshold,
        )


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[CandidateEvent, VerifierResult]] = []

    async def __call__(
        self, candidate: CandidateEvent, result: VerifierResult, frames: Sequence[BufferedFrame]
    ) -> None:
        self.calls.append((candidate, result))


def _match(object_id: str = "obj_keys", score: float = 0.9) -> GalleryScore:
    return GalleryScore(
        object_id=object_id, score=score, margin=None, runner_up_object_id=None, crop_references=()
    )


def _pipeline(
    reasoner: FixtureReasoner, gallery: StubGallery, recorder: Recorder, **kwargs: object
) -> Pipeline:
    return Pipeline(
        reasoner=reasoner,
        embedder=FixtureEmbedder(),
        gallery=gallery,  # type: ignore[arg-type]
        evidence_ring=EvidenceRing(max_duration=dt.timedelta(seconds=60)),
        on_confirmed=recorder,
        pipeline_version="test-1",
        reason_window_s=3.0,
        reason_interval_s=2.0,
        identity_min_cosine=0.75,
        event_cooldown_s=20.0,
        **kwargs,  # type: ignore[arg-type]
    )


async def _feed(pipeline: Pipeline, frames: Sequence[VideoFrame]) -> None:
    # Drain after each frame: one window is analyzed at a time, and a live
    # stream schedules the next window only after the previous finished, which
    # is exactly what draining between frames reproduces.
    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_a")
    for frame in frames:
        await pipeline.video_frame(
            session_id="sess_1", device_id="glasses-01", epoch_id="TR_a", frame=frame
        )
        await pipeline.drain()


async def test_a_matched_placed_event_is_written_with_its_location() -> None:
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert len(recorder.calls) == 1
    candidate, result = recorder.calls[0]
    assert candidate.action == "placed"
    assert candidate.identity is not None and candidate.identity.object_id == "obj_keys"
    assert result.description == "on the table"
    assert pipeline.metrics.observations_written == 1


async def test_second_localization_abstention_uses_the_first_event_crop() -> None:
    reasoner = FixtureReasoner(default=(_event("placed"),), localize_box=None)
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert len(recorder.calls) == 1
    assert pipeline.metrics.identity_matched == 1
    assert pipeline.metrics.observations_written == 1


async def test_an_unmatched_event_is_skipped_not_written() -> None:
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = StubGallery(labels={"keys"}, score=_match(score=0.4))  # below min_cosine
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert recorder.calls == []
    assert pipeline.metrics.identity_skipped == 1
    assert pipeline.metrics.observations_written == 0


async def test_a_query_below_its_raised_per_object_bar_is_skipped() -> None:
    # 0.82 clears the 0.75 floor -- the old global gate would have written this
    # observation. But a confusable same-label sibling raised this object's bar
    # to 0.90, so the per-object gate refuses rather than guess between the two.
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = StubGallery(labels={"keys"}, score=_match(score=0.82), override_threshold=0.90)
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert recorder.calls == []
    assert pipeline.metrics.identity_skipped == 1
    assert pipeline.metrics.observations_written == 0


async def test_no_gallery_match_at_all_is_skipped() -> None:
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = StubGallery(labels={"keys"}, score=None)
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert recorder.calls == []
    assert pipeline.metrics.identity_skipped == 1


async def test_nothing_happened_is_never_promoted() -> None:
    reasoner = FixtureReasoner(default=(_event("nothing_happened"),))
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert recorder.calls == []
    assert pipeline.metrics.events_detected == 0


async def test_a_repeat_within_cooldown_is_deduped() -> None:
    # Two windows, both "placed" for the same object, inside the cooldown.
    reasoner = FixtureReasoner(script=[(_event("placed"),), (_event("placed"),)])
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    # Frames far enough apart to schedule two windows (interval 2s) but inside
    # the 20s cooldown.
    await _feed(pipeline, [_frame(0, sequence=1), _frame(3, sequence=2)])

    assert len(recorder.calls) == 1
    assert pipeline.metrics.events_deduped == 1


async def test_motion_events_are_suppressed_by_default() -> None:
    reasoner = FixtureReasoner(script=[(_event("picked_up"),), (_event("placed"),)])
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1), _frame(3, sequence=2)])

    actions = [candidate.action for candidate, _ in recorder.calls]
    assert actions == ["placed"]
    assert pipeline.metrics.events_detected == 2
    assert pipeline.metrics.motion_events_suppressed == 1
    assert pipeline.metrics.identity_matched == 1
    assert pipeline.recent_events[0].outcome == "suppressed_by_policy"


async def test_motion_events_can_be_promoted_with_the_config_toggle() -> None:
    reasoner = FixtureReasoner(script=[(_event("picked_up"),), (_event("placed"),)])
    gallery = StubGallery(labels={"keys"}, score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder, promote_motion_events=True)

    await _feed(pipeline, [_frame(0, sequence=1), _frame(3, sequence=2)])

    actions = [candidate.action for candidate, _ in recorder.calls]
    assert actions == ["picked_up", "placed"]
    assert pipeline.metrics.motion_events_suppressed == 0


class LateGallery:
    """Empty until the first refresh -- a registration made before this process
    started, still in memory, that the cache has not loaded yet."""

    def __init__(self, *, score: GalleryScore) -> None:
        self._loaded = False
        self._score = score
        self.refreshes = 0

    @property
    def labels(self) -> frozenset[str]:
        return frozenset({"keys"}) if self._loaded else frozenset()

    async def refresh(self, *, force: bool = False) -> bool:
        self.refreshes += 1
        self._loaded = True  # memory had it all along; the cache just warmed up
        return True

    def match(
        self,
        queries: Sequence[object],
        *,
        label: str,
        summary_weight: float,
        floor: float = 0.0,
        confusion_margin: float = 0.0,
    ) -> GalleryScore | None:
        return replace(self._score, threshold=floor)


async def test_a_prior_registration_reappears_after_a_cold_start() -> None:
    # The gallery cache starts empty (fresh process), but the object is in
    # memory. Scheduling must NOT be gated on a non-empty gallery, or the cache
    # would never refresh and the registration would stay invisible forever.
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = LateGallery(score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)  # type: ignore[arg-type]

    await _feed(pipeline, [_frame(0, sequence=1)])

    assert gallery.refreshes >= 1
    assert len(recorder.calls) == 1  # the prior registration matched once loaded


async def test_an_empty_gallery_asks_the_reasoner_nothing() -> None:
    reasoner = FixtureReasoner(default=(_event("placed"),))
    gallery = StubGallery(labels=set(), score=_match())
    recorder = Recorder()
    pipeline = _pipeline(reasoner, gallery, recorder)

    await _feed(pipeline, [_frame(0, sequence=1), _frame(3, sequence=2)])

    assert reasoner.calls == []  # never invoked
    assert recorder.calls == []
