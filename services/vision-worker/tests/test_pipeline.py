"""Pipeline: the whole no-model path wired together end to end.

Every stage built so far -- FixtureDetector, GreedyIoUTracker,
ImageMotionPose, the stability machine, the evidence ring, and
RuleBasedVerifier -- driven by hand-built relay `VideoFrame`s, exactly the
way `consume.relay.RelayConsumer` would call this class from a real
connection. This is what the plan's critical-path claim rests on: "stop
after this and the demo works end to end on recorded clips, with no model
required."
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io

import numpy as np
import pytest
from PIL import Image
from visual_memory_media_contract.framing import payload_digest
from visual_memory_media_contract.protocol import VideoFrame
from visual_memory_vision_contract.protocol import (
    BoundingBox,
    CandidateEvent,
    Detection,
    DetectorRef,
    OverlayFrame,
    Point2D,
    VerifierResult,
)

from vision_worker.depth.base import DepthEstimator
from vision_worker.depth.fixture import FixtureDepthEstimator
from vision_worker.detect.fixture import FixtureDetector
from vision_worker.domain.stability import StabilityConfig, TrackRegistry
from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.pipeline import OBSERVED_NOT_PROMOTED, OverlaySink, Pipeline
from vision_worker.pose.image_motion import ImageMotionPose
from vision_worker.track.greedy_iou import GreedyIoUTracker
from vision_worker.verify.rules import RuleBasedVerifier, RuleBasedVerifierConfig

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
FRAME_INTERVAL = dt.timedelta(seconds=1 / 24)

_DETECTOR_REF = DetectorRef(name="fixture", checkpoint="n/a", revision="v1")
_TRACKER_REF = DetectorRef(name="greedy-iou", checkpoint="n/a", revision="v1")

_WIDTH, _HEIGHT = 48, 32
#: Byte-identical content on every frame, so ImageMotionPose's background
#: estimate stays ~(0, 0) throughout -- a stationary-camera simulation. The
#: fixture detector never looks at pixels, so this needn't visually match
#: the scripted box positions; it only needs to be a real, decodable JPEG so
#: the evidence ring and clip encoder see real bytes.
_BACKGROUND = np.random.default_rng(seed=7).integers(
    0, 256, size=(_HEIGHT, _WIDTH, 3), dtype=np.uint8
)


def _jpeg_payload() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(_BACKGROUND, mode="RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


_PAYLOAD = _jpeg_payload()


def a_detection(x: float) -> Detection:
    half_width = 0.08
    return Detection(
        label="keys",
        confidence=0.9,
        box=BoundingBox(x_min=x - half_width, y_min=0.42, x_max=x + half_width, y_max=0.58),
        centroid=Point2D(x=x, y=0.5),
    )


def a_video_frame(*, sequence: int, epoch_id: str = "TR_VCaaa") -> VideoFrame:
    captured_at = T0 + sequence * FRAME_INTERVAL
    frame = VideoFrame(
        session_id="sess_1",
        epoch_id=epoch_id,
        sequence=sequence,
        captured_at=captured_at,
        received_at=captured_at,
        relayed_at=captured_at,
        width=_WIDTH,
        height=_HEIGHT,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(_PAYLOAD),
        sha256=payload_digest(_PAYLOAD),
    )
    return frame.attach_payload(_PAYLOAD)


class RecordingSink:
    def __init__(self) -> None:
        self.confirmed: list[tuple[CandidateEvent, VerifierResult, tuple[BufferedFrame, ...]]] = []

    async def __call__(
        self, candidate: CandidateEvent, result: VerifierResult, frames: object
    ) -> None:
        self.confirmed.append((candidate, result, tuple(frames)))  # type: ignore[arg-type]


def a_pipeline(
    *,
    script: list[list[Detection]],
    sink: RecordingSink,
    stability_config: StabilityConfig,
    depth_estimator: DepthEstimator | None = None,
    depth_model_ref: DetectorRef | None = None,
    source_fps: float = 24.0,
    verification_queue_depth: int = 8,
    overlay_sink: OverlaySink | None = None,
    overlay_depth_interval_s: float = 1.0,
    max_detections_per_frame: int = 20,
) -> Pipeline:
    return Pipeline(
        detector=FixtureDetector(script, loop=False),
        detector_ref=_DETECTOR_REF,
        tracker=GreedyIoUTracker(iou_threshold=0.15),
        tracker_ref=_TRACKER_REF,
        pose_source=ImageMotionPose(),
        track_registry=TrackRegistry(stability_config),
        evidence_ring=EvidenceRing(dt.timedelta(seconds=30)),
        verifier=RuleBasedVerifier(RuleBasedVerifierConfig(min_confidence=0.5, min_frame_count=1)),
        detection_labels=(),
        state_machine_version="vision-stability-v1",
        pipeline_version="vision-pipeline-v1",
        on_confirmed=sink,
        # Matches FRAME_INTERVAL above, so the observed-rate check agrees with
        # the timeline these tests actually feed it.
        source_fps=source_fps,
        depth_estimator=depth_estimator,
        depth_model_ref=depth_model_ref,
        verification_queue_depth=verification_queue_depth,
        overlay_sink=overlay_sink,
        overlay_depth_interval_s=overlay_depth_interval_s,
        max_detections_per_frame=max_detections_per_frame,
    )


async def drive(pipeline: Pipeline, frame_count: int, *, epoch_id: str = "TR_VCaaa") -> None:
    """Feed `frame_count` consecutive frames of one epoch, then wait for every
    candidate they proposed to be verified.

    Verification runs off the frame loop (`verify/pending.py`), so the last
    frame returning does not mean the pipeline has finished thinking. Draining
    here keeps every test below reading as a plain sequence of causes and
    effects, and `test_verification_does_not_block_the_frame_loop` covers the
    asynchrony itself directly rather than leaving it implicit everywhere.
    """
    for sequence in range(frame_count):
        await pipeline.video_frame(
            session_id="sess_1",
            device_id="glasses-01",
            epoch_id=epoch_id,
            frame=a_video_frame(sequence=sequence, epoch_id=epoch_id),
        )
    await pipeline.drain()


async def test_keys_carried_in_and_set_down_confirms_a_placed_candidate() -> None:
    """Clip 1's shape: carried into frame, then settles -- a `placed`
    candidate should reach `on_confirmed` with real evidence attached."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    # Moving: 0.10 -> 0.16 -> 0.22 (well above the 0.02 residual threshold,
    # with the tracker's lowered iou_threshold still matching the overlap).
    # Then still at 0.22 for enough frames to cross dwell_frames.
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
    ]
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    # Exactly one candidate reaches Memory. The track's first sample also
    # produced an "observed", but a first sighting is not evidence that
    # anything happened and never becomes an observation -- see
    # pipeline._NON_PROMOTING_ACTIONS.
    actions = [candidate.action for candidate, _, _ in sink.confirmed]
    assert actions == ["placed"]

    candidate, result, frames = sink.confirmed[0]
    assert candidate.label == "keys"
    assert candidate.session_id == "sess_1"
    assert candidate.device_id == "glasses-01"
    assert candidate.media_epoch_id == "TR_VCaaa"
    assert result.outcome == "confirmed"
    assert len(frames) >= 1
    assert pipeline.metrics.candidates_confirmed == 1
    assert pipeline.metrics.sightings_not_promoted == 1
    assert pipeline.metrics.frames_processed == len(script)


async def test_a_track_that_only_ever_moves_never_confirms_a_placed_candidate() -> None:
    """The failure this whole design exists to prevent, at the pipeline
    level: an object that keeps moving must never produce a `placed`
    candidate just because a sighting happened.
    """
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10 + 0.06 * i)] for i in range(8)]
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert sink.confirmed == []


async def test_an_object_merely_seen_produces_no_observation_at_all() -> None:
    """Clips 4 and 5 from the plan: "object visible, never touched" and
    "walking past an object" must produce **zero** observations, not a
    harmless one. The state machine still reports the sighting -- the
    activity log shows it -- but nothing is verified, encoded, or uploaded.
    """
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    # Stationary from the very first frame and never long enough to cross
    # passive_confirmation_frames: indistinguishable from an object that was
    # simply always there.
    script = [[a_detection(0.5)] for _ in range(6)]
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert sink.confirmed == []
    assert pipeline.metrics.candidates_proposed == 0
    assert pipeline.metrics.sightings_not_promoted == 1

    # Visible to a human watching, though, and honestly labelled: no verifier
    # was consulted, so the outcome is not one of the verifier's three.
    [event] = pipeline.recent_events
    assert event.action == "observed"
    assert event.outcome == "not_promoted"
    assert event.reason_code == OBSERVED_NOT_PROMOTED


async def test_frames_for_a_different_epoch_are_ignored() -> None:
    """A frame that arrives for an epoch this pipeline was never told about
    (e.g. right after a reset raced a stale in-flight message) must not be
    processed against the current epoch's state."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    sink = RecordingSink()
    pipeline = a_pipeline(script=[[a_detection(0.2)]], sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    stale = a_video_frame(sequence=0, epoch_id="TR_VCstale")

    await pipeline.video_frame(
        session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCstale", frame=stale
    )

    assert pipeline.metrics.frames_processed == 0


async def test_epoch_reset_treats_a_reused_track_id_as_a_new_sighting() -> None:
    """The pipeline-level version of docs/06's reconnect trap: `track-1`
    before and after an epoch reset must be treated as unrelated objects.

    Each epoch's first frame independently produces a first sighting -- not a
    "placed", which would mean the second epoch inherited the first's
    stability state instead of starting from a clean slate.
    """
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    sink = RecordingSink()
    pipeline = a_pipeline(
        script=[[a_detection(0.2)], [a_detection(0.2)]], sink=sink, stability_config=config
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, 1)

    # A reconnect: fresh epoch, same physical scene by coincidence -- the
    # tracker mints "track-1" again from a clean slate.
    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCbbb")
    await drive(pipeline, 1, epoch_id="TR_VCbbb")

    assert sink.confirmed == []
    # Two independent first sightings, one per epoch. If state had leaked
    # across the reset, the second frame would have continued the first
    # track's settling instead of announcing itself.
    assert [event.action for event in pipeline.recent_events] == ["observed", "observed"]
    assert pipeline.metrics.sightings_not_promoted == 2


async def test_recent_events_records_every_verifier_outcome() -> None:
    """`/v1/events` (task #53) is backed by this -- a human watching the
    pipeline needs to see rejections and unverified outcomes too, not only
    what reached Memory."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
    ]
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    events = pipeline.recent_events
    assert [e.action for e in events] == ["observed", "placed"]
    assert [e.outcome for e in events] == ["not_promoted", "confirmed"]
    assert all(e.track_id == "track-1" and e.label == "keys" for e in events)
    assert all(e.confidence > 0.0 for e in events)


async def test_a_configured_depth_estimator_annotates_the_confirmed_candidate() -> None:
    """When a depth estimator is wired in, `depth_m` reaches the candidate
    Memory eventually sees -- task #39's low-cadence wiring, exercised with
    no GPU via `FixtureDepthEstimator`."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    # Moves, then holds long enough to cross dwell_frames=3 -- only a `placed`
    # candidate reaches the sink, so the script has to actually produce one.
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
    ]
    sink = RecordingSink()
    depth_ref = DetectorRef(name="fixture", checkpoint="n/a", revision="v1")
    pipeline = a_pipeline(
        script=script,
        sink=sink,
        stability_config=config,
        depth_estimator=FixtureDepthEstimator(range_m=2.0),
        depth_model_ref=depth_ref,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert sink.confirmed, "expected a placed candidate to reach the sink"
    for candidate, _, _ in sink.confirmed:
        assert candidate.object_candidate.depth_m == 2.0
        assert candidate.depth_model == depth_ref


async def test_with_no_depth_estimator_configured_candidates_carry_no_depth() -> None:
    """The default shape -- `depth_estimator=None` -- must stay exactly what
    it always was: no `depth_m`, no `depth_model`, nothing silently implied."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    # Moves, then holds long enough to cross dwell_frames=3 -- only a `placed`
    # candidate reaches the sink, so the script has to actually produce one.
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
    ]
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert sink.confirmed, "expected a placed candidate to reach the sink"
    for candidate, _, _ in sink.confirmed:
        assert candidate.object_candidate.depth_m is None
        assert candidate.depth_model is None


# --- The frame rate every threshold is derived from -------------------------


async def test_observed_fps_measures_what_the_relay_actually_delivers() -> None:
    """Nothing else in the system fails loudly when `VMA_SOURCE_FPS`
    disagrees with the gateway's `VMA_SAMPLE_FPS` -- the stability thresholds
    just quietly come out scaled. This measurement is what catches it, and
    `/v1/status` is where a human sees it."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    sink = RecordingSink()
    pipeline = a_pipeline(script=[[]], sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    assert pipeline.observed_fps is None, "must not guess a rate before it has one"

    await drive(pipeline, 40)

    measured = pipeline.observed_fps
    assert measured is not None
    # a_video_frame stamps frames FRAME_INTERVAL apart, i.e. 24fps.
    assert measured == pytest.approx(24.0, rel=0.01)


async def test_a_rate_that_disagrees_with_the_configured_one_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Configured for the gateway's default 2fps, fed a 24fps stream: every
    stability threshold is 12x shorter than intended. It says so."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    sink = RecordingSink()
    pipeline = a_pipeline(script=[[]], sink=sink, stability_config=config, source_fps=2.0)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    with caplog.at_level("WARNING", logger="vision_worker.pipeline"):
        await drive(pipeline, 40)

    warnings = [r for r in caplog.records if "frame rate disagrees" in r.message]
    assert len(warnings) == 1, "one warning, not one per frame"


async def test_the_epoch_reset_forgets_the_previous_epoch_s_rate() -> None:
    """A reconnect may come back at a different rate; averaging across the
    boundary would report neither."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    sink = RecordingSink()
    pipeline = a_pipeline(script=[[]], sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, 40)
    assert pipeline.observed_fps is not None

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCbbb")

    assert pipeline.observed_fps is None


# --- A resting object that vanishes -----------------------------------------


class ScriptedVerifier:
    """Confirms whatever it is asked, resolving a question into `answer`."""

    def __init__(self, answer: str = "picked_up") -> None:
        self._answer = answer
        self.seen: list[tuple[str, int, int]] = []

    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: object = (),
        samples: object = (),
        decoded: object = (),
    ) -> VerifierResult:
        del decoded
        self.seen.append((candidate.action, len(frames), len(samples)))  # type: ignore[arg-type]
        return VerifierResult(
            candidate_id=candidate.candidate_id,
            outcome="confirmed",
            reason_code="scripted",
            latency_ms=1.0,
            verifier=_DETECTOR_REF,
            occurred_at=T0,
            resolved_action=self._answer if candidate.action == "vanished" else None,  # type: ignore[arg-type]
        )


async def test_keys_that_settle_then_disappear_raise_a_question() -> None:
    """Clip 2's exact shape: the keys are set down, then stop being detected
    while a hand is over them. Before this, that produced silence -- and
    silence means "still on the desk", which is the wrong answer this whole
    service exists to prevent.
    """
    config = StabilityConfig(
        dwell_frames=3, passive_confirmation_frames=999, reacquire_within_frames=3
    )
    # Carried in, set down, then gone: the detector finds nothing at all.
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [],
        [],
        [],
        [],
        [],
    ]
    sink = RecordingSink()
    verifier = ScriptedVerifier(answer="picked_up")
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)
    pipeline._verifier = verifier  # type: ignore[assignment]  # noqa: SLF001

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    actions = [candidate.action for candidate, _, _ in sink.confirmed]
    assert "vanished" in actions, "the disappearance must be raised, not assumed away"
    assert pipeline.metrics.vanishings_questioned >= 1

    # And the verifier's answer -- not the question -- is what carries forward.
    vanished = next(r for c, r, _ in sink.confirmed if c.action == "vanished")
    assert vanished.resolved_action == "picked_up"


async def test_the_vanish_window_reaches_back_to_before_it_disappeared() -> None:
    """The interesting moment is the approach -- a hand arriving -- not the
    empty desk afterwards. A window containing only empty frames would give a
    verifier nothing to reason about."""
    config = StabilityConfig(
        dwell_frames=3, passive_confirmation_frames=999, reacquire_within_frames=3
    )
    script = [
        [a_detection(0.10)],
        [a_detection(0.16)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [a_detection(0.22)],
        [],
        [],
        [],
        [],
        [],
    ]
    sink = RecordingSink()
    verifier = ScriptedVerifier()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)
    pipeline._verifier = verifier  # type: ignore[assignment]  # noqa: SLF001

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    vanish_calls = [call for call in verifier.seen if call[0] == "vanished"]
    assert vanish_calls, "no vanish candidate reached the verifier"
    _, frame_count, sample_count = vanish_calls[0]
    assert frame_count > 1, "the window must span more than the moment of loss"
    assert sample_count > 1, "the verifier needs the track's history, not one sample"


async def test_an_object_that_never_settled_raises_no_question_when_it_leaves() -> None:
    """Only a *resting* object's disappearance changes anything. A track that
    was still moving is already in transit as far as memory is concerned."""
    config = StabilityConfig(
        dwell_frames=3, passive_confirmation_frames=999, reacquire_within_frames=3
    )
    script = [[a_detection(0.10 + 0.06 * i)] for i in range(6)] + [[]] * 5
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)
    pipeline._verifier = ScriptedVerifier()  # type: ignore[assignment]  # noqa: SLF001

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert [c.action for c, _, _ in sink.confirmed] == []
    assert pipeline.metrics.vanishings_questioned == 0


# --- Verification runs off the frame loop ------------------------------------


class BlockingVerifier:
    """Confirms, but only once released -- a stand-in for the VLM verifier,
    which really does take tens of seconds against a live 8fps stream."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = 0

    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: object = (),
        samples: object = (),
        decoded: object = (),
    ) -> VerifierResult:
        del frames, samples, decoded
        self.started += 1
        await self.release.wait()
        return VerifierResult(
            candidate_id=candidate.candidate_id,
            outcome="confirmed",
            reason_code="blocking",
            latency_ms=1.0,
            verifier=_DETECTOR_REF,
            occurred_at=T0,
        )


async def test_a_slow_verifier_does_not_stall_frame_handling() -> None:
    """The failure this design exists to prevent. `media_gateway.relay.hub`
    keeps one latest-frame slot per subscriber, so a pipeline that blocks in
    `video_frame` does not slow the stream -- it makes the gateway discard
    frames, and the stability machine then reads a gap it cannot see.

    So the measure is not that verification is fast. It is that frames keep
    being processed while a verification is stuck.
    """
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 10
    sink = RecordingSink()
    verifier = BlockingVerifier()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)
    pipeline._verifier = verifier  # type: ignore[assignment]  # noqa: SLF001

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    for sequence in range(len(script)):
        await pipeline.video_frame(
            session_id="sess_1",
            device_id="glasses-01",
            epoch_id="TR_VCaaa",
            frame=a_video_frame(sequence=sequence),
        )
        # Stands in for the relay's own suspension point: `RelayConsumer` awaits
        # the next websocket message between frames, which is where the worker
        # task gets to run. The fixture detector never awaits anything real, so
        # without this the loop would never yield and the asynchrony under test
        # would be invisible.
        await asyncio.sleep(0)

    assert pipeline.metrics.frames_processed == len(script), (
        "every frame must be handled while a verification is still in flight"
    )
    assert verifier.started == 1, "verification should have begun and still be stuck"
    assert sink.confirmed == [], "nothing can be confirmed while the verifier is blocked"

    verifier.release.set()
    await pipeline.aclose()
    assert len(sink.confirmed) >= 1, "the answer lands once the verifier returns"


async def test_candidates_are_dropped_rather_than_stalling_the_stream() -> None:
    """A drop is a real event lost, so it is counted and surfaced at
    `/v1/status` -- but the alternative is blocking, which corrupts the state
    machine that produced the candidate."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    # Five objects, far enough apart to stay distinct tracks, all settling
    # together -- five candidates proposed within a frame or two of each other,
    # against a queue that can hold one.
    # Each moves for two frames and then settles -- a static object never
    # produces a `placed` candidate at all, only a first sighting.
    bases = [0.08, 0.25, 0.42, 0.59, 0.76]
    script = [[a_detection(base + 0.06 * min(i, 2)) for base in bases] for i in range(10)]
    sink = RecordingSink()
    verifier = BlockingVerifier()
    pipeline = a_pipeline(
        script=script, sink=sink, stability_config=config, verification_queue_depth=1
    )
    pipeline._verifier = verifier  # type: ignore[assignment]  # noqa: SLF001

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    for sequence in range(len(script)):
        await pipeline.video_frame(
            session_id="sess_1",
            device_id="glasses-01",
            epoch_id="TR_VCaaa",
            frame=a_video_frame(sequence=sequence),
        )
        await asyncio.sleep(0)

    assert pipeline.metrics.frames_processed == len(script), (
        "frames keep flowing even as candidates are being discarded"
    )
    assert pipeline.metrics.candidates_proposed >= 5
    assert pipeline.verifications_dropped > 0, "a depth-1 queue behind a stuck verifier must drop"

    verifier.release.set()
    await pipeline.aclose()


# --- The overlay stream ------------------------------------------------------


async def test_no_overlay_sink_means_no_overlay_work_at_all() -> None:
    """The default, and the normal state of a deployed service: nothing is
    watching, so nothing is assembled."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 4
    sink = RecordingSink()
    pipeline = a_pipeline(script=script, sink=sink, stability_config=config)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert pipeline.metrics.frames_processed == len(script)


async def test_every_tracked_object_is_published_including_ones_never_promoted() -> None:
    """A viewer wants to see what the detector sees. A console that only drew
    objects worth remembering would show an empty frame for the commonest case
    and look broken -- so overlays are collected above the promotion rules, not
    after them."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 4
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        overlay_sink=overlays.append,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert len(overlays) == len(script), "one overlay per processed frame"
    # The very first frame is a first sighting -- `observed`, which the pipeline
    # deliberately never promotes. It must still be drawn.
    assert overlays[0].tracks, "the first sighting must reach a viewer"
    assert pipeline.metrics.sightings_not_promoted >= 1


async def test_detection_output_is_bounded_before_overlay_and_depth_work() -> None:
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    frame = [
        a_detection(0.1 + index * 0.01).model_copy(update={"confidence": 0.5 + index * 0.01})
        for index in range(6)
    ]
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=[frame],
        sink=RecordingSink(),
        stability_config=config,
        overlay_sink=overlays.append,
        max_detections_per_frame=3,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, 1)

    assert len(overlays[0].tracks) == 3
    assert min(track.confidence for track in overlays[0].tracks) >= 0.53


async def test_an_overlay_carries_what_a_viewer_needs_to_draw_it() -> None:
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 4
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        overlay_sink=overlays.append,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    last = overlays[-1]
    assert last.session_id == "sess_1"
    assert last.media_epoch_id == "TR_VCaaa"
    assert last.sequence == len(script) - 1
    assert last.width > 0 and last.height > 0
    assert last.pipeline_latency_ms >= 0.0

    [track] = last.tracks
    assert track.label == "keys"
    assert 0.0 <= track.box.x_min <= 1.0, "normalized, so a viewer scales freely"
    # By the last frame the object has stopped moving; a viewer showing the
    # state change is the clearest evidence a state machine is running.
    assert track.motion_state in {"settling", "at_rest"}


async def test_a_broken_viewer_cannot_stop_the_pipeline() -> None:
    """The overlay stream is a debugging and demo surface. A bug in it, or in
    serializing a frame, must never cost the service its actual job."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 4
    sink = RecordingSink()

    def exploding(_: OverlayFrame) -> None:
        raise RuntimeError("the viewer is on fire")

    pipeline = a_pipeline(script=script, sink=sink, stability_config=config, overlay_sink=exploding)

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert pipeline.metrics.frames_processed == len(script)
    assert [c.action for c, _, _ in sink.confirmed] == ["placed"], (
        "candidates must still be proposed, verified and confirmed"
    )


# --- Depth on the overlay, sampled at a cadence ------------------------------


class CountingDepth:
    """A depth adapter that reports a constant and counts how often it ran."""

    def __init__(self, range_m: float = 1.25) -> None:
        self.range_m = range_m
        self.calls = 0

    async def initialize(self) -> None: ...

    async def aclose(self) -> None: ...

    async def estimate(self, frame_rgb: object, detections: object) -> list[Detection]:
        del frame_rgb
        self.calls += 1
        return [d.model_copy(update={"depth_m": self.range_m}) for d in detections]  # type: ignore[union-attr]


async def test_depth_is_not_measured_when_nobody_is_watching() -> None:
    """This sampling exists only to put a number on a box. A deployment with no
    viewer attached -- which is most of them, most of the time -- must not pay
    for a second model."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 6
    depth = CountingDepth()
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        depth_estimator=depth,  # type: ignore[arg-type]
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    # The candidate path still runs depth on settling; what must not happen is
    # a per-frame overlay pass.
    assert depth.calls <= 2, "no overlay depth sampling without a viewer"


async def test_depth_reaches_the_overlay_and_carries_its_age() -> None:
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 6
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        depth_estimator=CountingDepth(range_m=1.25),  # type: ignore[arg-type]
        overlay_sink=overlays.append,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    measured = [t for frame in overlays for t in frame.tracks if t.depth_m is not None]
    assert measured, "a viewer should see a depth reading"
    assert measured[0].depth_m == 1.25
    assert all(t.depth_age_s is not None and t.depth_age_s >= 0 for t in measured)


async def test_depth_is_sampled_at_a_cadence_not_every_frame() -> None:
    """A second heavy model per frame costs far more than the measurement is
    worth for a quantity that changes slowly -- and on a machine already at its
    frame budget it would just cause more frames to be dropped."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 22
    depth = CountingDepth()
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        depth_estimator=depth,  # type: ignore[arg-type]
        overlay_sink=[].append,
        # FRAME_INTERVAL is 1/24s, so 24 frames span one second.
        overlay_depth_interval_s=1.0,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert depth.calls < len(script) // 4, (
        f"depth ran {depth.calls}x for {len(script)} frames -- per-frame, not a cadence"
    )


async def test_a_stale_reading_is_carried_forward_with_a_growing_age() -> None:
    """Between samples a track keeps its last reading. The age is what stops
    that being a lie: a number shown as live when it is seconds old is worse
    than showing none."""
    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 20
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        depth_estimator=CountingDepth(),  # type: ignore[arg-type]
        overlay_sink=overlays.append,
        overlay_depth_interval_s=1.0,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    ages = [t.depth_age_s for f in overlays for t in f.tracks if t.depth_age_s is not None]
    assert ages, "readings should be carried forward between samples"
    assert max(ages) > min(ages), "and their age must grow while they are not refreshed"


async def test_a_depth_failure_leaves_the_pipeline_running() -> None:
    """`depth/moge.py` is explicit that a pipeline annotating nothing is
    degraded, not broken."""

    class ExplodingDepth(CountingDepth):
        async def estimate(self, frame_rgb: object, detections: object) -> list[Detection]:
            self.calls += 1
            raise RuntimeError("no depth today")

    config = StabilityConfig(dwell_frames=3, passive_confirmation_frames=999)
    script = [[a_detection(0.10)], [a_detection(0.16)]] + [[a_detection(0.22)]] * 6
    overlays: list[OverlayFrame] = []
    pipeline = a_pipeline(
        script=script,
        sink=RecordingSink(),
        stability_config=config,
        depth_estimator=ExplodingDepth(),  # type: ignore[arg-type]
        overlay_sink=overlays.append,
    )

    await pipeline.epoch_started(session_id="sess_1", device_id="glasses-01", epoch_id="TR_VCaaa")
    await drive(pipeline, len(script))

    assert pipeline.metrics.frames_processed == len(script)
    assert all(t.depth_m is None for f in overlays for t in f.tracks)
