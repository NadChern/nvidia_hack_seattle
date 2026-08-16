"""The interaction/rest state machine -- the product.

Decides, from a per-frame stream of `TrackSample`s for one tracked object,
whether the object is at rest somewhere or moving through the world, and
emits a `CandidateAction` only on a transition that means something:
`observed` (a track's first sighting), `placed` (settled into confirmed
rest), `picked_up` (left confirmed rest into motion), `carried` (continues
moving). Most frames emit nothing -- most samples change nothing worth a
candidate.

Pure: no torch, no fastapi, no I/O, no wall clock. Every timestamp this module
touches comes from the sample itself. `tests/test_domain_isolation.py` asserts
that mechanically, the same discipline `services/application-memory/src/
application_memory/domain/reducer.py` uses for the same reason.

**The distinction that decides a location was never "is a hand touching it" --
it is "is this object at rest in the world, or moving through it."** Keys
carried from the kitchen to the front door and pocketed must never resolve to
"the front hall" just because the last sighting was there. A sighting only
updates the confirmed location once the object has held its position for
`dwell_seconds`; a sighting while moving marks the object as moving and never
overwrites a confirmed placement.

**Why a brand-new track never promotes on its first stable sighting.** An
object that has always been sitting somewhere, glanced at once, looks exactly
like an object that was just placed -- both are stationary the moment Vision
first sees them. Distinguishing "just placed" from "was already there" from
motion alone is not something a single frame can answer honestly, so this
module does not pretend to: a track that settles immediately from its very
first sample needs `passive_confirmation_seconds` of sustained stillness
before it promotes, while a track that visibly moved and then settled needs
only the much shorter `dwell_seconds`, because motion-then-settle is strong
evidence that a placement genuinely just happened. Clip 4 ("object visible,
never touched") is calibrated to be shorter than `passive_confirmation_seconds`
so a brief glance never promotes; clip 1 ("keys placed on a table") is staged
to show the motion-then-settle transition so it promotes quickly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

from visual_memory_vision_contract.protocol import (
    CandidateAction,
    MotionState,
    Point2D,
    Point3D,
    TrackSample,
)


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    """Thresholds the state machine uses to decide "at rest" from "moving".

    Configuration, not constants -- reported at `/v1/status`, matching how the
    Memory Service reports its `PromotionPolicy`, so an evaluation run can
    cite the threshold set it used.

    **Durations, not frame counts.** Earlier this machine counted frames, which
    silently means a different real duration at every source rate: thresholds
    built for 8fps mean a 7x-longer dwell when the relay actually delivers
    ~1fps, and the relay's rate is device- and network-bound, not something the
    service controls. So the thresholds are wall-clock seconds, compared
    against the `captured_at` timestamps the samples already carry -- correct at
    1fps or 24fps, and immune to the rate changing mid-track. The module stays
    pure: the "clock" is always a sample's (or the current frame's) timestamp,
    never `datetime.now()`.
    """

    #: Seconds of held position required to confirm "placed" after an observed
    #: motion phase -- short, because motion-then-settle is strong evidence.
    dwell_seconds: float = 0.5
    #: Seconds of held position required to confirm "placed" from a track's
    #: very first sighting, with no observed motion beforehand. Deliberately
    #: much larger than `dwell_seconds` -- see the module docstring.
    passive_confirmation_seconds: float = 3.75
    #: World-space metres of displacement considered "moving", used when both
    #: the current and reference sample carry a `world_point` (depth and pose
    #: both available). Accounts for depth-estimate noise.
    world_motion_threshold_m: float = 0.05
    #: Normalized image-space residual -- |object screen motion - background
    #: motion| -- considered "moving" on the image-space-only path. The
    #: inversion this exploits: a held object stays roughly fixed in the
    #: frame while the background sweeps past; a resting object's screen
    #: motion tracks the background's, since both come from ego-motion alone.
    image_residual_threshold: float = 0.02
    #: Seconds a track may go undetected (occlusion, a missed frame) before its
    #: state resets to "absent" rather than being treated as continuous. Sized
    #: for a brief hand occlusion.
    reacquire_within_seconds: float = 1.875
    #: While a track continues moving, "carried" is re-emitted at most this
    #: often -- a periodic ping rather than a candidate on every frame.
    carried_emit_interval_seconds: float = 2.5


def _elapsed(now: dt.datetime, since: dt.datetime) -> float:
    return (now - since).total_seconds()


@dataclass(frozen=True, slots=True)
class TrackState:
    """Per-track state carried between steps.

    Owned by the caller: a live consumer keeps one of these per active
    `track_id` (see `TrackRegistry`), and a fixture-driven test keeps one per
    scenario. Never inspects a clock or a database -- every timestamp it
    carries came from a `TrackSample`.
    """

    motion_state: MotionState = "absent"
    #: `captured_at` of the first sample of the current uninterrupted stable
    #: run -- the anchor the dwell / passive-confirmation durations are measured
    #: from. `None` while moving or absent; reset whenever motion resumes.
    settling_started_at: dt.datetime | None = None
    #: `captured_at` of the most recent frame in which this track was detected.
    #: The reacquire window is measured as (current frame time - this), so an
    #: occlusion is timed in seconds regardless of frame rate.
    last_seen_at: dt.datetime | None = None
    #: `captured_at` of the last `carried`/`picked_up` emission, so the periodic
    #: `carried` ping is spaced by a real duration, not a frame count.
    last_emission_at: dt.datetime | None = None
    #: True once this settling phase followed an observed motion phase, which
    #: is what makes the fast `dwell_seconds` threshold apply instead of the
    #: slow `passive_confirmation_seconds` one.
    settled_from_motion: bool = False
    #: `captured_at` of the sample that began the current excursion out of
    #: "absent"/"at_rest" -- reset whenever the track first starts moving (or
    #: is first sighted), and preserved across "moving" -> "settling" ->
    #: "at_rest". A `placed` candidate's window therefore spans from the
    #: start of the approach through to confirmation, not just the final
    #: still frames: a clip of the object arriving and being set down is
    #: better evidence than one of it merely sitting still. A `picked_up`
    #: candidate's window starts at the pickup itself, since nothing
    #: preceded it worth including.
    state_started_at: dt.datetime | None = None
    #: A fixed point set when settling began, used by the world-space
    #: stability check (total drift since rest began). The image-space check
    #: instead compares consecutive frames via `last_centroid` -- per-frame
    #: background-relative velocity is the more meaningful signal there,
    #: since `background_motion` is itself a per-frame quantity.
    reference_world_point: Point3D | None = None
    last_world_point: Point3D | None = None
    last_centroid: Point2D | None = None
    #: The action last emitted for this track, so a continued "moving" state
    #: does not re-emit "picked_up" on every frame -- only the transition does.
    last_emitted_action: CandidateAction | None = None


@dataclass(frozen=True, slots=True)
class StabilityStep:
    """One sample's outcome: the next `TrackState`, and an action to emit, if
    any. `action` is `None` on most frames."""

    state: TrackState
    action: CandidateAction | None = None
    #: True on the one step where a track has been absent long enough that its
    #: state was reset to a clean slate. The state machine itself needs no
    #: more than that reset, but a caller holding per-track bookkeeping needs
    #: to know it can stop holding it -- without this, every `track_id` a
    #: tracker ever mints accumulates for the life of the epoch, and the
    #: per-frame sweep over absent tracks grows with it. See `TrackRegistry`.
    retired: bool = False


def _world_distance(a: Point3D, b: Point3D) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _image_residual(state: TrackState, sample: TrackSample) -> float | None:
    """|object screen motion - background motion| between the last sample and
    this one. `None` when there is nothing to compare against -- the first
    sample of a track, or a frame with no background-motion signal (no
    `ImageMotionPose` this frame)."""
    if state.last_centroid is None or sample.background_motion is None:
        return None
    object_dx = sample.detection.centroid.x - state.last_centroid.x
    object_dy = sample.detection.centroid.y - state.last_centroid.y
    residual_x = object_dx - sample.background_motion.x
    residual_y = object_dy - sample.background_motion.y
    return (residual_x**2 + residual_y**2) ** 0.5


def _is_stable(state: TrackState, sample: TrackSample, config: StabilityConfig) -> bool | None:
    """Whether this sample continues to hold position relative to the world.

    `None` means inconclusive -- neither a world-point comparison nor an
    image-space residual could be computed -- and the caller must not advance
    or reset the dwell counter on an inconclusive frame; a track that cannot
    be assessed should not be silently assumed to be moving or resting.
    """
    if sample.world_point is not None and state.reference_world_point is not None:
        distance = _world_distance(sample.world_point, state.reference_world_point)
        return distance <= config.world_motion_threshold_m
    residual = _image_residual(state, sample)
    if residual is not None:
        return residual <= config.image_residual_threshold
    return None


def initial_state() -> TrackState:
    return TrackState()


def step(
    state: TrackState,
    sample: TrackSample | None,
    config: StabilityConfig | None = None,
    *,
    now: dt.datetime | None = None,
) -> StabilityStep:
    """Advance one track by one frame.

    `sample=None` means the track was not detected this frame -- a missed
    detection or brief occlusion, not necessarily gone for good. Nothing is
    ever emitted on an absent frame: a track disappearing while moving leaves
    the object moving (Memory keeps it `in_transit`) and a track disappearing
    while at rest leaves the confirmed placement standing -- both are already
    true from the last emitted action, so there is nothing new to tell Memory.

    `now` is the current frame's `captured_at`. It is what times an absent
    frame, which carries no sample of its own; for a present frame it defaults
    to the sample's own timestamp, so a caller with one stream of frames can
    pass it unconditionally (the pipeline does) or omit it (tests with only
    present frames do).
    """
    config = config or StabilityConfig()

    if sample is None:
        return _step_absent(state, config, now)

    if state.motion_state == "absent":
        return _step_first_sighting(sample)

    seen_state = replace(
        state,
        last_world_point=sample.world_point,
        last_centroid=sample.detection.centroid,
        last_seen_at=sample.captured_at,
    )

    stable = _is_stable(state, sample, config)
    if stable is None:
        # Inconclusive frame: hold position and wait for one that can
        # actually tell us something, rather than guessing.
        return StabilityStep(state=seen_state)
    if stable:
        return _step_stable(seen_state, sample, config)
    return _step_moving(seen_state, sample, config)


def _step_first_sighting(sample: TrackSample) -> StabilityStep:
    """A track's very first sample -- either genuinely the first time this
    `track_id` has ever been seen, or a reappearance after a gap long enough
    that `_step_absent` reset it to a clean slate. Either way this is treated
    as a brand-new object: establish reference points and emit `observed`,
    never `placed`, no matter how stable it already looks. See the module
    docstring for why "placed" requires more than one stable frame.
    """
    next_state = TrackState(
        motion_state="settling",
        settling_started_at=sample.captured_at,
        last_seen_at=sample.captured_at,
        settled_from_motion=False,
        reference_world_point=sample.world_point,
        last_world_point=sample.world_point,
        last_centroid=sample.detection.centroid,
        last_emitted_action="observed",
        state_started_at=sample.captured_at,
    )
    return StabilityStep(state=next_state, action="observed")


def _step_stable(
    state: TrackState, sample: TrackSample, config: StabilityConfig
) -> StabilityStep:
    if state.motion_state == "at_rest":
        # Already confirmed and still not moving -- nothing new to say.
        return StabilityStep(state=state)

    # The stable run is anchored at its first frame; a run continuing from
    # "moving" (where the anchor was cleared) starts it here.
    settling_started_at = state.settling_started_at or sample.captured_at
    held_for = _elapsed(sample.captured_at, settling_started_at)
    threshold = (
        config.dwell_seconds if state.settled_from_motion else config.passive_confirmation_seconds
    )
    if held_for >= threshold:
        next_state = replace(
            state,
            motion_state="at_rest",
            settling_started_at=settling_started_at,
            last_emitted_action="placed",
            last_emission_at=sample.captured_at,
        )
        return StabilityStep(state=next_state, action="placed")

    return StabilityStep(
        state=replace(state, motion_state="settling", settling_started_at=settling_started_at)
    )


def _step_moving(state: TrackState, sample: TrackSample, config: StabilityConfig) -> StabilityStep:
    was_at_rest = state.motion_state == "at_rest"
    entering_motion = state.motion_state != "moving"
    reset_emission = was_at_rest or entering_motion

    next_state = replace(
        state,
        motion_state="moving",
        settling_started_at=None,
        settled_from_motion=True,
        reference_world_point=state.last_world_point,
        # A fresh motion phase starts its own window and emission clock;
        # continuing an existing one keeps the timestamps it already had.
        state_started_at=(sample.captured_at if reset_emission else state.state_started_at),
        last_emission_at=(sample.captured_at if reset_emission else state.last_emission_at),
    )

    if was_at_rest:
        # Left a confirmed rest state -- the one transition that must always
        # be reported, since it invalidates the placement Memory is holding.
        return StabilityStep(
            state=replace(next_state, last_emitted_action="picked_up"), action="picked_up"
        )

    if (
        not entering_motion
        and state.last_emission_at is not None
        and _elapsed(sample.captured_at, state.last_emission_at) >= config.carried_emit_interval_seconds
    ):
        return StabilityStep(
            state=replace(next_state, last_emitted_action="carried", last_emission_at=sample.captured_at),
            action="carried",
        )

    return StabilityStep(state=next_state)


def _step_absent(
    state: TrackState, config: StabilityConfig, now: dt.datetime | None
) -> StabilityStep:
    if state.last_seen_at is None or now is None:
        # No timing reference yet (a track that has only ever been absent, or a
        # caller that did not supply the frame time): hold, do not retire.
        return StabilityStep(state=state)
    if _elapsed(now, state.last_seen_at) > config.reacquire_within_seconds:
        # An object that was **resting** and is now gone is the one absence
        # worth reporting. Everything else is already accounted for: a track
        # that was moving when it disappeared left the object in transit, and
        # a track that was never confirmed anywhere had nothing to invalidate.
        #
        # This one is different, and it is the failure clip 2 exposed. The
        # keys sat on the desk; a hand covered them; the next clean frame
        # showed an empty desk. Nothing here ever saw them move, so the old
        # rule -- gone while at rest means still there -- kept a placement
        # that had already stopped being true, and answered with it
        # confidently. Silence was the bug.
        #
        # `vanished` does not claim a pickup. It claims only that this
        # pipeline cannot account for the object any more, and asks. A
        # verifier that can look at the frames resolves it; one that cannot
        # returns `unverified`, and nothing is written -- which is still
        # better than silently asserting the object never left.
        if state.motion_state == "at_rest":
            return StabilityStep(state=TrackState(), action="vanished", retired=True)
        # Gone long enough that a reappearance is a new sighting, not a
        # continuation -- the same discipline as the epoch reset, applied at
        # the scale of one occlusion instead of one reconnect. A completely
        # clean TrackState, with no preserved emission history: as far as
        # this track_id's future is concerned, whatever produced candidates
        # before the gap might not even be the same physical object, so the
        # next sighting must go through _step_first_sighting and announce
        # "observed" again rather than staying silent.
        return StabilityStep(state=TrackState(), retired=True)
    # Still within the reacquire window: nothing to advance -- absence is timed
    # from `last_seen_at`, which is already on the state.
    return StabilityStep(state=state)


class TrackRegistry:
    """Owns per-track state across one media epoch.

    `reset()` on every `epoch_started`. `track_id` is only ever meaningful
    within one `(session_id, media_epoch_id)` -- carrying it across a
    reconnect is the exact trap `docs/06-Data-Contract.md` warns about: a
    tracker's numbering restarts after a dropout, so `track-42` before and
    after are different physical objects.
    """

    def __init__(self, config: StabilityConfig | None = None) -> None:
        self._config = config or StabilityConfig()
        self._tracks: dict[str, TrackState] = {}

    @property
    def config(self) -> StabilityConfig:
        return self._config

    @property
    def active_track_ids(self) -> frozenset[str]:
        """Ids still worth stepping. A caller sweeping absent tracks each
        frame reads this rather than keeping its own set, so retirement
        reclaims work in both places at once -- see `Pipeline.video_frame`."""
        return frozenset(self._tracks)

    def reset(self) -> None:
        """Drop all track state. Call on every `epoch_started`."""
        self._tracks.clear()

    def observe(
        self, track_id: str, sample: TrackSample | None, *, now: dt.datetime | None = None
    ) -> StabilityStep:
        state = self._tracks.get(track_id, TrackState())
        result = step(state, sample, self._config, now=now)
        if result.retired:
            # Absent past `reacquire_within_seconds`: its state is a clean
            # TrackState now, so holding the entry buys nothing and a
            # never-shrinking dict costs per-frame work forever. A later
            # sighting of the same id starts from the same clean slate via
            # the `.get(..., TrackState())` default above.
            self._tracks.pop(track_id, None)
        else:
            self._tracks[track_id] = result.state
        return result

    def drop(self, track_id: str) -> None:
        """Stop tracking an id entirely -- e.g. the tracker itself reports it
        permanently lost, distinct from the transient absence `step` already
        tolerates within `reacquire_within_seconds`."""
        self._tracks.pop(track_id, None)


__all__ = [
    "StabilityConfig",
    "StabilityStep",
    "TrackRegistry",
    "TrackState",
    "initial_state",
    "step",
]
