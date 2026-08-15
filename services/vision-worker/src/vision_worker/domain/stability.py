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
`dwell_frames`; a sighting while moving marks the object as moving and never
overwrites a confirmed placement.

**Why a brand-new track never promotes on its first stable sighting.** An
object that has always been sitting somewhere, glanced at once, looks exactly
like an object that was just placed -- both are stationary the moment Vision
first sees them. Distinguishing "just placed" from "was already there" from
motion alone is not something a single frame can answer honestly, so this
module does not pretend to: a track that settles immediately from its very
first sample needs `passive_confirmation_frames` of sustained stillness
before it promotes, while a track that visibly moved and then settled needs
only the much shorter `dwell_frames`, because motion-then-settle is strong
evidence that a placement genuinely just happened. Clip 4 ("object visible,
never touched") is calibrated to be shorter than `passive_confirmation_frames`
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

    **Every frame count here is meaningless without a frame rate.** The
    defaults below are the plan's values at 24fps, the measured 6DoF-stable
    ceiling on the X3 Pro -- but the frame rate this service actually sees is
    the Media Gateway's `VMA_SAMPLE_FPS`, which relays a *sampled* stream at a
    fraction of the capture rate. At 8fps these same numbers would mean an
    11-second passive confirmation instead of 3.75. Nothing in a fixture can
    catch that, since a fixture supplies its own timestamps.

    So a service builds this with `from_durations()` from real durations and
    the rate it is actually being fed, and the raw frame-count constructor
    stays for tests and fixtures that control their own timeline. See
    `vision_worker.main.build_stability_config`.
    """

    #: Frames of held position required to confirm "placed" after an observed
    #: motion phase -- fast, because motion-then-settle is strong evidence.
    #: 12 frames at 24fps is 0.5s.
    dwell_frames: int = 12
    #: Frames of held position required to confirm "placed" from a track's
    #: very first sighting, with no observed motion beforehand. Deliberately
    #: much larger than `dwell_frames` -- see the module docstring. 90 frames
    #: at 24fps is 3.75s.
    passive_confirmation_frames: int = 90
    #: World-space metres of displacement per frame considered "moving", used
    #: when both the current and reference sample carry a `world_point`
    #: (depth and pose both available). Accounts for depth-estimate noise.
    world_motion_threshold_m: float = 0.05
    #: Normalized image-space residual -- |object screen motion - background
    #: motion| -- considered "moving" on the image-space-only path. The
    #: inversion this exploits: a held object stays roughly fixed in the
    #: frame while the background sweeps past; a resting object's screen
    #: motion tracks the background's, since both come from ego-motion alone.
    image_residual_threshold: float = 0.02
    #: Frames a track may go undetected (occlusion, a missed frame) before its
    #: state resets to "absent" rather than being treated as continuous. 45
    #: frames at 24fps is 1.875s, sized for a brief hand occlusion.
    reacquire_within_frames: int = 45
    #: While a track continues moving, "carried" is re-emitted at most this
    #: often -- a periodic ping rather than a candidate on every frame. 60
    #: frames at 24fps is 2.5s.
    carried_emit_interval_frames: int = 60

    @classmethod
    def from_durations(
        cls,
        *,
        source_fps: float,
        dwell_seconds: float = 0.5,
        passive_confirmation_seconds: float = 3.75,
        reacquire_within_seconds: float = 1.875,
        carried_emit_interval_seconds: float = 2.5,
        world_motion_threshold_m: float = 0.05,
        image_residual_threshold: float = 0.02,
    ) -> StabilityConfig:
        """Convert real durations into frame counts at `source_fps`.

        The durations are what a human actually reasons about ("half a second
        of stillness confirms a placement"); the frame counts are what a
        per-frame state machine can count. Keeping the conversion here rather
        than in `config.py` keeps it pure and testable with no settings
        object, and means the arithmetic that decides what a threshold *means*
        lives next to the code that applies it.

        Rounds to at least one frame: a duration shorter than one frame
        interval cannot be represented, and silently producing a zero-frame
        threshold would confirm a placement from a single sighting. A caller
        that cares whether a duration survived the rounding compares the
        result against its own inputs -- see `main.build_stability_config`,
        which warns when it did not.
        """
        return cls(
            dwell_frames=_frames_for(dwell_seconds, source_fps),
            passive_confirmation_frames=_frames_for(passive_confirmation_seconds, source_fps),
            world_motion_threshold_m=world_motion_threshold_m,
            image_residual_threshold=image_residual_threshold,
            reacquire_within_frames=_frames_for(reacquire_within_seconds, source_fps),
            carried_emit_interval_frames=_frames_for(carried_emit_interval_seconds, source_fps),
        )


def _frames_for(seconds: float, fps: float) -> int:
    return max(1, round(seconds * fps))


@dataclass(frozen=True, slots=True)
class TrackState:
    """Per-track state carried between steps.

    Owned by the caller: a live consumer keeps one of these per active
    `track_id` (see `TrackRegistry`), and a fixture-driven test keeps one per
    scenario. Never inspects a clock or a database -- every timestamp it
    carries came from a `TrackSample`.
    """

    motion_state: MotionState = "absent"
    stable_frames: int = 0
    frames_since_seen: int = 0
    frames_since_last_emission: int = 0
    #: True once this settling phase followed an observed motion phase, which
    #: is what makes the fast `dwell_frames` threshold apply instead of the
    #: slow `passive_confirmation_frames` one.
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
    state: TrackState, sample: TrackSample | None, config: StabilityConfig | None = None
) -> StabilityStep:
    """Advance one track by one frame.

    `sample=None` means the track was not detected this frame -- a missed
    detection or brief occlusion, not necessarily gone for good. Nothing is
    ever emitted on an absent frame: a track disappearing while moving leaves
    the object moving (Memory keeps it `in_transit`) and a track disappearing
    while at rest leaves the confirmed placement standing -- both are already
    true from the last emitted action, so there is nothing new to tell Memory.
    """
    config = config or StabilityConfig()

    if sample is None:
        return _step_absent(state, config)

    if state.motion_state == "absent":
        return _step_first_sighting(sample)

    seen_state = replace(
        state,
        last_world_point=sample.world_point,
        last_centroid=sample.detection.centroid,
        frames_since_seen=0,
    )

    stable = _is_stable(state, sample, config)
    if stable is None:
        # Inconclusive frame: hold position and wait for one that can
        # actually tell us something, rather than guessing.
        return StabilityStep(state=seen_state)
    if stable:
        return _step_stable(seen_state, config)
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
        stable_frames=1,
        settled_from_motion=False,
        reference_world_point=sample.world_point,
        last_world_point=sample.world_point,
        last_centroid=sample.detection.centroid,
        last_emitted_action="observed",
        state_started_at=sample.captured_at,
    )
    return StabilityStep(state=next_state, action="observed")


def _step_stable(state: TrackState, config: StabilityConfig) -> StabilityStep:
    if state.motion_state == "at_rest":
        # Already confirmed and still not moving -- nothing new to say.
        return StabilityStep(state=state)

    stable_frames = state.stable_frames + 1
    threshold = (
        config.dwell_frames if state.settled_from_motion else config.passive_confirmation_frames
    )
    if stable_frames >= threshold:
        next_state = replace(
            state,
            motion_state="at_rest",
            stable_frames=stable_frames,
            last_emitted_action="placed",
            frames_since_last_emission=0,
        )
        return StabilityStep(state=next_state, action="placed")

    return StabilityStep(state=replace(state, motion_state="settling", stable_frames=stable_frames))


def _step_moving(state: TrackState, sample: TrackSample, config: StabilityConfig) -> StabilityStep:
    was_at_rest = state.motion_state == "at_rest"
    entering_motion = state.motion_state != "moving"

    next_state = replace(
        state,
        motion_state="moving",
        stable_frames=0,
        settled_from_motion=True,
        reference_world_point=state.last_world_point,
        frames_since_last_emission=(
            0 if (was_at_rest or entering_motion) else state.frames_since_last_emission + 1
        ),
        # A fresh motion phase starts its own window; continuing an existing
        # one keeps the timestamp it already had.
        state_started_at=(
            sample.captured_at if (was_at_rest or entering_motion) else state.state_started_at
        ),
    )

    if was_at_rest:
        # Left a confirmed rest state -- the one transition that must always
        # be reported, since it invalidates the placement Memory is holding.
        return StabilityStep(
            state=replace(next_state, last_emitted_action="picked_up"), action="picked_up"
        )

    if (
        not entering_motion
        and next_state.frames_since_last_emission >= config.carried_emit_interval_frames
    ):
        return StabilityStep(
            state=replace(next_state, last_emitted_action="carried", frames_since_last_emission=0),
            action="carried",
        )

    return StabilityStep(state=next_state)


def _step_absent(state: TrackState, config: StabilityConfig) -> StabilityStep:
    frames_since_seen = state.frames_since_seen + 1
    if frames_since_seen > config.reacquire_within_frames:
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
    return StabilityStep(state=replace(state, frames_since_seen=frames_since_seen))


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

    def observe(self, track_id: str, sample: TrackSample | None) -> StabilityStep:
        state = self._tracks.get(track_id, TrackState())
        result = step(state, sample, self._config)
        if result.retired:
            # Absent past `reacquire_within_frames`: its state is a clean
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
        tolerates within `reacquire_within_frames`."""
        self._tracks.pop(track_id, None)


__all__ = [
    "StabilityConfig",
    "StabilityStep",
    "TrackRegistry",
    "TrackState",
    "initial_state",
    "step",
]
