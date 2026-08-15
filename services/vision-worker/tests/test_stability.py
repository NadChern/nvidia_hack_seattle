"""The interaction/rest state machine -- the product.

No GPU, no relay, no clock -- every scenario is a hand-built sequence of
`TrackSample`s fed through `TrackRegistry`, matching the discipline
`services/application-memory/tests/test_reducer.py` uses for the same reason.
These are the image-space-only path (no `world_point`), which is also the
no-glasses path anyone can run on a laptop -- see the plan's "Working without
the glasses" section.

Each test name states a clip scenario from the plan so a fixture built later
in `packages/vision-contract/fixtures/` (task #34) has a direct table-driven
counterpart to replay against.
"""

from __future__ import annotations

import datetime as dt

from visual_memory_vision_contract.protocol import BoundingBox, Detection, Point2D, TrackSample

from vision_worker.domain.stability import StabilityConfig, TrackRegistry

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)
FRAME_INTERVAL = dt.timedelta(seconds=1 / 24)

#: A config with short thresholds so a test does not need hundreds of frames
#: to reach a conclusion. Real values are the module defaults, tuned for a
#: 24fps stream; these are proportionally smaller, not differently shaped.
FAST = StabilityConfig(
    dwell_frames=3,
    passive_confirmation_frames=6,
    world_motion_threshold_m=0.05,
    image_residual_threshold=0.02,
    reacquire_within_frames=4,
    carried_emit_interval_frames=3,
)


def still(track_id: str, frame_index: int, *, x: float = 0.5, y: float = 0.5) -> TrackSample:
    """A sample with zero background motion and an unmoving object -- the
    image-space "at rest" signal: the object's screen position matches the
    background's (both zero), so the residual is zero."""
    at = T0 + frame_index * FRAME_INTERVAL
    detection = Detection(
        label="keys",
        confidence=0.9,
        box=BoundingBox(x_min=x - 0.02, y_min=y - 0.02, x_max=x + 0.02, y_max=y + 0.02),
        centroid=Point2D(x=x, y=y),
    )
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        captured_at=at,
        detection=detection,
        background_motion=Point2D(x=0.0, y=0.0),
    )


def carried(track_id: str, frame_index: int, *, x: float, y: float = 0.5) -> TrackSample:
    """A sample whose object moves while the background does not -- the
    image-space "moving" signal: a held object stays roughly fixed relative
    to the camera while diverging from a stationary background."""
    at = T0 + frame_index * FRAME_INTERVAL
    detection = Detection(
        label="keys",
        confidence=0.9,
        box=BoundingBox(x_min=x - 0.02, y_min=y - 0.02, x_max=x + 0.02, y_max=y + 0.02),
        centroid=Point2D(x=x, y=y),
    )
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        captured_at=at,
        detection=detection,
        background_motion=Point2D(x=0.0, y=0.0),
    )


def actions(registry: TrackRegistry, track_id: str, samples: list[TrackSample]) -> list[str]:
    out: list[str] = []
    for sample in samples:
        result = registry.observe(track_id, sample)
        if result.action is not None:
            out.append(result.action)
    return out


# --- The demo scenario ------------------------------------------------------


def test_keys_picked_up_and_carried_out_never_answers_with_the_last_sighting() -> None:
    """Clip 2, the scenario this whole design exists for.

    Keys are placed (motion then settle), then picked up and carried away.
    They must never re-settle -- the walk to the front door and out the door
    is exactly the failure a plain last-seen rule would get wrong.
    """
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []

    # Carried into frame and set down.
    for i in range(3):
        samples.append(carried("track-42", frame, x=0.2 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.35))
        frame += 1
    # Picked back up and carried out of frame, never settling again.
    for i in range(6):
        samples.append(carried("track-42", frame, x=0.35 + i * 0.05))
        frame += 1

    seen = actions(registry, "track-42", samples)

    assert "placed" in seen
    assert "picked_up" in seen
    assert seen.index("placed") < seen.index("picked_up")
    # The critical assertion: nothing after the pickup settles into a new
    # "placed". A system that emits one here would answer with the front
    # hall instead of admitting the keys are unaccounted for.
    assert "placed" not in seen[seen.index("picked_up") :]


def test_keys_carried_to_another_room_and_set_down_reports_the_new_location() -> None:
    """Clip 3: carried, then a new placed."""
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []
    for i in range(3):
        samples.append(carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.25))
        frame += 1
    for i in range(4):
        samples.append(carried("track-42", frame, x=0.25 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.45))
        frame += 1

    seen = actions(registry, "track-42", samples)

    assert seen.count("placed") == 2
    assert seen.count("picked_up") == 1


# --- The cases a naive pipeline gets wrong ----------------------------------


def test_an_object_visible_but_never_touched_produces_no_placement() -> None:
    """Clip 4. A track that is stable from its very first sample -- no
    observed motion beforehand -- must not promote on a brief sighting: it
    looks exactly like an object that has always been there. Only "observed"
    may fire."""
    registry = TrackRegistry(FAST)
    samples = [still("track-42", i, x=0.5) for i in range(FAST.passive_confirmation_frames - 1)]

    seen = actions(registry, "track-42", samples)

    assert seen == ["observed"]
    assert "placed" not in seen


def test_walking_past_an_object_produces_no_candidate_at_all() -> None:
    """Clip 5: a single fleeting sighting. Even "observed" only fires once,
    and nothing else follows for a track that vanishes immediately after."""
    registry = TrackRegistry(FAST)
    samples = [still("track-42", 0, x=0.5)]

    seen = actions(registry, "track-42", samples)

    assert seen == ["observed"]


def test_a_sighting_while_moving_never_overwrites_a_confirmed_placement() -> None:
    """The core invariant, stated directly: once at_rest, only a genuine
    motion signal may leave it -- a single ambiguous or ambiguous-then-stable
    frame must not silently re-confirm a different spot."""
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []
    for i in range(3):
        samples.append(carried("track-42", frame, x=0.2 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.35))
        frame += 1

    first_pass = actions(registry, "track-42", samples)
    assert first_pass == ["observed", "placed"]

    # Continuing to observe it at rest must never re-emit "placed".
    more_stillness = [still("track-42", frame + i, x=0.35) for i in range(5)]
    assert actions(registry, "track-42", more_stillness) == []


# --- Occlusion and reconnection ---------------------------------------------


def test_a_brief_occlusion_does_not_reset_a_confirmed_placement() -> None:
    """Clip 6: the track drops for a few frames (a hand passes in front of
    it) well within `reacquire_within_frames`, then returns unmoved. No new
    "observed" or "placed" should fire for the reappearance."""
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []
    for i in range(3):
        samples.append(carried("track-42", frame, x=0.2 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.35))
        frame += 1

    before_gap = actions(registry, "track-42", samples)
    assert before_gap == ["observed", "placed"]

    # Occluded for fewer frames than reacquire_within_frames.
    for _ in range(FAST.reacquire_within_frames - 1):
        registry.observe("track-42", None)

    reappearance = registry.observe("track-42", still("track-42", frame + 10, x=0.35))
    assert reappearance.action is None


def test_a_reconnect_reuses_the_track_id_but_the_epoch_reset_prevents_a_merge() -> None:
    """Clip 8. `docs/06-Data-Contract.md`:110 -- a tracker's numbering
    restarts after a dropout, so `track-1` before and after a reconnect are
    different physical objects. `TrackRegistry.reset()` on `epoch_started` is
    what prevents the second object from inheriting the first's state."""
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []
    for i in range(3):
        samples.append(carried("track-1", frame, x=0.2 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-1", frame, x=0.35))
        frame += 1

    before_reconnect = actions(registry, "track-1", samples)
    assert before_reconnect == ["observed", "placed"]

    # The gateway reports a new media epoch; the consumer must reset.
    registry.reset()

    # A different physical object reuses the same track_id numbering.
    first_sample_after_reconnect = registry.observe("track-1", still("track-1", 0, x=0.6))

    # Must be treated as a brand new track -- "observed", not silently
    # continuing the prior object's "at_rest" state.
    assert first_sample_after_reconnect.action == "observed"


def test_an_occlusion_longer_than_the_grace_period_is_treated_as_a_new_sighting() -> None:
    registry = TrackRegistry(FAST)
    registry.observe("track-42", still("track-42", 0, x=0.5))

    for _ in range(FAST.reacquire_within_frames + 1):
        registry.observe("track-42", None)

    reappearance = registry.observe("track-42", still("track-42", 100, x=0.5))
    assert reappearance.action == "observed"


# --- The periodic "carried" ping --------------------------------------------


def test_carried_is_re_emitted_periodically_while_motion_continues() -> None:
    """The plan's "carried while it persists": a ping, not a candidate on every frame."""
    registry = TrackRegistry(FAST)
    frame = 0
    samples = []
    for i in range(3):
        samples.append(carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        samples.append(still("track-42", frame, x=0.25))
        frame += 1
    # A long motion phase, long enough to cross the carried-ping interval
    # more than once.
    for i in range(FAST.carried_emit_interval_frames * 2 + 2):
        samples.append(carried("track-42", frame, x=0.25 + i * 0.05))
        frame += 1

    seen = actions(registry, "track-42", samples)

    assert seen.count("carried") >= 2


# --- state_started_at: what an EvidenceWindow gets built from --------------


def test_state_started_at_spans_the_whole_approach_not_just_the_final_stillness() -> None:
    """A `placed` candidate's window should start when the object first began
    moving toward where it settled, not only once it went still.

    This is deliberate, not an artifact: a clip showing the approach and the
    hand setting the object down is better evidence than one showing only
    idle stillness -- the object arriving is the interesting event. The
    first sample of a brand-new track is always "settling" (never "moving"),
    so the window's start is pinned to the *second* sample, which is where
    the first `entering_motion` transition actually occurs.
    """
    registry = TrackRegistry(FAST)
    frame = 0
    for i in range(3):
        registry.observe("track-42", carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    approach_started_at_frame = 1

    result = None
    for _ in range(FAST.dwell_frames + 1):
        result = registry.observe("track-42", still("track-42", frame, x=0.25))
        frame += 1

    assert result is not None and result.action == "placed"
    expected_start = T0 + approach_started_at_frame * FRAME_INTERVAL
    assert result.state.state_started_at == expected_start


def test_picking_up_resets_state_started_at_to_the_pickup_frame() -> None:
    registry = TrackRegistry(FAST)
    frame = 0
    for i in range(3):
        registry.observe("track-42", carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        registry.observe("track-42", still("track-42", frame, x=0.25))
        frame += 1

    pickup_frame = frame
    result = registry.observe("track-42", carried("track-42", frame, x=0.4))

    assert result.action == "picked_up"
    assert result.state.state_started_at == T0 + pickup_frame * FRAME_INTERVAL


# --- Thresholds are durations, not frame counts ----------------------------


def test_from_durations_converts_at_the_configured_rate() -> None:
    """The frame counts the state machine counts are meaningless without the
    rate they were derived from -- the relay's, which is the gateway's
    sampled rate and not the glasses' capture rate."""
    at_24 = StabilityConfig.from_durations(source_fps=24.0)
    assert at_24.dwell_frames == 12
    assert at_24.passive_confirmation_frames == 90
    assert at_24.reacquire_within_frames == 45
    assert at_24.carried_emit_interval_frames == 60

    # A deliberately slow source. The same durations, an entirely different
    # set of frame counts -- which is the whole point of not hard-coding them.
    at_2 = StabilityConfig.from_durations(source_fps=2.0)
    assert at_2.dwell_frames == 1
    assert at_2.passive_confirmation_frames == 8
    assert at_2.reacquire_within_frames == 4
    assert at_2.carried_emit_interval_frames == 5


def test_from_durations_never_rounds_a_threshold_down_to_zero() -> None:
    """A zero-frame threshold would confirm a placement from no evidence at
    all. Callers that care about the rounding compare and warn -- see
    `main.build_stability_config` -- but the floor is enforced here."""
    config = StabilityConfig.from_durations(source_fps=0.5, dwell_seconds=0.1)

    assert config.dwell_frames == 1


def test_from_durations_passes_the_motion_thresholds_through_unscaled() -> None:
    """Metres and normalized image displacement are per-frame quantities in
    space, not in time -- a frame rate must not touch them."""
    config = StabilityConfig.from_durations(
        source_fps=24.0, world_motion_threshold_m=0.11, image_residual_threshold=0.07
    )

    assert config.world_motion_threshold_m == 0.11
    assert config.image_residual_threshold == 0.07


# --- Retirement reclaims what it stops tracking ----------------------------


def test_a_track_absent_past_the_reacquire_window_is_dropped_not_just_reset() -> None:
    """Resetting a retired track's state but keeping its entry would leave
    every id a tracker ever minted accumulating for the life of the epoch,
    and the caller's per-frame sweep over absent tracks growing with it."""
    registry = TrackRegistry(FAST)
    registry.observe("track-42", still("track-42", 0))
    assert registry.active_track_ids == {"track-42"}

    for _ in range(FAST.reacquire_within_frames):
        result = registry.observe("track-42", None)
        assert not result.retired
        assert registry.active_track_ids == {"track-42"}

    result = registry.observe("track-42", None)

    assert result.retired
    assert registry.active_track_ids == frozenset()


def test_a_retired_track_id_seen_again_starts_from_a_clean_slate() -> None:
    """Dropping the entry must not change what a later sighting means: the
    same id reappearing is a new sighting either way, since whatever produced
    it before the gap might not be the same physical object."""
    registry = TrackRegistry(FAST)
    registry.observe("track-42", still("track-42", 0))
    for _ in range(FAST.reacquire_within_frames + 1):
        registry.observe("track-42", None)

    result = registry.observe("track-42", still("track-42", 50))

    assert result.action == "observed"
    assert result.state.motion_state == "settling"
    assert registry.active_track_ids == {"track-42"}


# --- Vanishing: the silence that was clip 2's failure -----------------------


def test_a_resting_object_that_disappears_asks_rather_than_assuming() -> None:
    """The keys sat on the desk; a hand covered them; the desk was empty.
    Nothing here ever saw them move, and the old rule -- gone while at rest
    means still there -- kept a placement that had stopped being true.
    """
    registry = TrackRegistry(FAST)
    frame = 0
    for i in range(3):
        registry.observe("track-42", carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        result = registry.observe("track-42", still("track-42", frame, x=0.25))
        frame += 1
    assert result.action == "placed" and result.state.motion_state == "at_rest"

    emitted = []
    for _ in range(FAST.reacquire_within_frames + 1):
        step = registry.observe("track-42", None)
        if step.action is not None:
            emitted.append(step.action)

    assert emitted == ["vanished"]
    assert registry.active_track_ids == frozenset()


def test_an_object_that_was_moving_when_it_left_asks_nothing() -> None:
    """Already accounted for: a track that was in motion when it disappeared
    left the object in transit, which memory already knows. Only a *resting*
    object's disappearance changes anything."""
    registry = TrackRegistry(FAST)
    frame = 0
    for i in range(6):
        registry.observe("track-42", carried("track-42", frame, x=0.1 + i * 0.08))
        frame += 1

    emitted = [
        step.action
        for _ in range(FAST.reacquire_within_frames + 1)
        if (step := registry.observe("track-42", None)).action is not None
    ]

    assert emitted == []


def test_a_brief_occlusion_does_not_ask() -> None:
    """Within the re-acquisition window the track is merely absent, not gone.
    Asking on every blink would spend a model call on nothing."""
    registry = TrackRegistry(FAST)
    frame = 0
    for i in range(3):
        registry.observe("track-42", carried("track-42", frame, x=0.1 + i * 0.05))
        frame += 1
    for _ in range(FAST.dwell_frames + 1):
        registry.observe("track-42", still("track-42", frame, x=0.25))
        frame += 1

    emitted = [
        step.action
        for _ in range(FAST.reacquire_within_frames)
        if (step := registry.observe("track-42", None)).action is not None
    ]

    assert emitted == []
