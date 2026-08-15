"""The candidate-event and verifier contract.

`docs/06-Data-Contract.md` § Candidate verification boundary is the normative
definition. These models are its executable form.

This is deliberately **not** the canonical Memory Service payload -- docs/06 is
explicit that "the candidate-event record is an internal Vision Service
contract, not the canonical Memory Service payload." A `CandidateEvent` only
becomes a `visual_memory_memory_contract.protocol.Observation` after a
`VerifierResult` of `confirmed`; `rejected` and `unverified` candidates are
retained as diagnostics and never cross into trusted memory.

Person 2 (spatial verification and evaluation, per docs/05) owns
`verify/rules.py` inside `services/vision-worker` and replaces it with a real
adapter. This package is what makes that swap mechanical: implement
`verify(CandidateEvent, EvidenceWindow) -> VerifierResult` against the models
here, assert against `fixtures.py`, and nothing else in the pipeline changes.

Models are frozen and ignore unknown fields, matching every other contract
package in this repository, so a producer pinned to an older minor version
keeps working when a consumer starts accepting an additional field.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, TypeAlias, TypeGuard, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PlainSerializer

#: Additive optional fields are a minor bump; removals, renames, and semantic
#: changes are a major bump. 1.1 added `CandidateAction.vanished` plus
#: `VerifierResult.resolved_action` and `.description` -- additive, and a
#: producer pinned to 1.0 keeps working because it simply never emits them.
#: 1.2 added `OverlayTrack` and `OverlayFrame`, which are new shapes rather
#: than changes to existing ones, so nothing pinned to 1.1 is affected. 1.3
#: added `OverlayTrack.depth_age_s`, optional and defaulting to None.
#: Versioned independently of the memory-contract schema -- this is a different
#: shape with a different producer and consumer.
SCHEMA_VERSION = "1.3"


def _utc_iso(value: dt.datetime) -> str:
    """UTC ISO-8601 with a Z suffix and millisecond precision.

    Matches `visual_memory_memory_contract.protocol._utc_iso` and
    `visual_memory_media_contract`'s equivalent exactly, so a timestamp that
    crosses from a candidate into an `Observation` round-trips byte-for-byte.
    Duplicated rather than imported for the same reason the rest of this
    module is self-contained -- see the module docstring.
    """
    return value.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


UtcTimestamp: TypeAlias = Annotated[AwareDatetime, PlainSerializer(_utc_iso, return_type=str)]

#: What a candidate claims, or asks.
#:
#: The first four mirror `visual_memory_memory_contract.protocol.EventAction`
#: exactly, and a confirmed candidate carrying one becomes an
#: `Observation.event.action` unchanged.
#:
#: `vanished` is the exception, and is **not** memory vocabulary: it means "an
#: object that was resting here stopped being detected, and this pipeline does
#: not know why." It is a question rather than a claim. Left unanswered it can
#: never become an observation -- `emit/memory.py` refuses it -- so a verifier
#: must resolve it into one of the other four via
#: `VerifierResult.resolved_action`, or the candidate is retained as a
#: diagnostic and nothing is written.
#:
#: Why it exists: an object can leave a table without this service ever seeing
#: it move. A hand covers it, and the next clean frame shows an empty table.
#: Before this, that produced silence, and silence means "still there" -- the
#: confident wrong answer this whole contract is built to prevent.
#: The four a memory observation may actually carry, mirroring
#: `visual_memory_memory_contract.protocol.EventAction`.
MemoryAction: TypeAlias = Literal["observed", "picked_up", "carried", "placed"]
CandidateAction: TypeAlias = MemoryAction | Literal["vanished"]

MEMORY_ACTIONS: frozenset[str] = frozenset(get_args(MemoryAction))


def is_memory_action(action: CandidateAction) -> TypeGuard[MemoryAction]:
    """Whether this action can become an observation.

    A `TypeGuard` rather than a bare membership test so the narrowing is
    visible to a type checker too -- `emit/memory.py` refuses a `vanished`
    that no verifier resolved, and that refusal should be provable rather
    than merely asserted at runtime.
    """
    return action in MEMORY_ACTIONS


#: What a verifier returns. Exactly these three, per docs/06: "confirmed",
#: "rejected", or "unverified" -- never a partial or a free-text result.
VerifierOutcome: TypeAlias = Literal["confirmed", "rejected", "unverified"]

#: The stability state machine's states (see `domain/stability.py`). Carried
#: in the contract so a fixture can assert the state a track was in when a
#: candidate fired, independent of the motion signal that produced it.
MotionState: TypeAlias = Literal["absent", "moving", "settling", "at_rest"]

Confidence: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Point2D(_Frozen):
    """Normalized image coordinates, top-left origin -- matches the prior
    project's `VisionTypes.cs` convention, which the client-side back-
    projection code this package's geometry layer ports was built against."""

    x: float
    y: float


class Point3D(_Frozen):
    """Camera-space or world-space metres, depending on context. Which one is
    always stated by the field that carries a `Point3D` -- see `Detection.box3d`
    (camera space) versus `TrackSample.world_point` (world space)."""

    x: float
    y: float
    z: float


class BoundingBox(_Frozen):
    """Normalized, top-left origin. Matches the prior project's `VisionTypes.cs`."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Box3D(_Frozen):
    """An oriented 3D box fit to a detection's depth points.

    Eight corners in **camera space, metres, MoGe's Y-down convention** --
    front face CCW as 0,1,2,3, then the matching back face 4,5,6,7, so the 12
    edges are the two faces (0-1-2-3-0, 4-5-6-7-4) plus the four connectors
    (0-4, 1-5, 2-6, 3-7). A world-frame consumer must flip Y before use; see
    `domain/geometry.py`, which ports the prior project's `ComputeWorldPoint`.
    """

    corners: tuple[Point3D, Point3D, Point3D, Point3D, Point3D, Point3D, Point3D, Point3D]


class DetectorRef(_Frozen):
    """What produced something, precisely enough to reproduce it.

    Mirrors `visual_memory_memory_contract.protocol.DetectorRef`. Duplicated
    rather than imported; see the module docstring.
    """

    name: str
    checkpoint: str
    revision: str


class Detection(_Frozen):
    """One model's output for one object in one frame.

    `depth_m` and `box3d` are populated only when a depth adapter ran on this
    frame -- per the plan, depth runs at low cadence and on settling, not on
    every frame, so most `Detection`s carry `depth_m=None`. That is the normal
    case, not a degraded one: a `Detection` with no depth still drives the
    image-space stability path.
    """

    label: str
    confidence: Confidence
    box: BoundingBox
    centroid: Point2D
    #: Metres from the camera to the detected surface, median-sampled over the
    #: box. `None` when no depth adapter ran on this frame, or it produced no
    #: valid sample.
    depth_m: float | None = None
    box3d: Box3D | None = None


class TrackSample(_Frozen):
    """One frame's sample of a tracked object -- the unit the stability state
    machine consumes, and the shape synthetic fixtures in `fixtures/` are
    written in.

    `world_point` is populated only once both a depth adapter and a pose
    source have run; `background_motion` is the cheaper `ImageMotionPose`
    signal used when it has not. A live pipeline may carry both on the same
    sample; the stability machine prefers `world_point` when present -- see
    `domain/stability.py`.
    """

    track_id: str
    frame_index: int
    captured_at: UtcTimestamp
    detection: Detection
    #: World-space position in metres, once depth and pose are both available.
    world_point: Point3D | None = None
    #: Image-space displacement of background features between this frame and
    #: the last, the `ImageMotionPose` fallback signal. `None` on the first
    #: frame of a track, or when the pose source is `DevicePose`.
    background_motion: Point2D | None = None


class HandCandidate(_Frozen):
    """A hand detection associated with a candidate.

    docs/06 lists "object, hand, room, and surface candidates" as fields a
    candidate-event record identifies. This implementation deliberately does
    not detect hands -- see docs/02's amended "Hand/object interaction state
    machine" section, and the plan's Context: the distinction that decides a
    location is whether the object is at rest, not whether a hand touches it.

    The field stays optional and always `None` here rather than being removed,
    so a future owner who adds hand detection has a contract-compatible slot
    to fill instead of a breaking change. This mirrors how `Location`'s fields
    in memory-contract stay null rather than guessed when unobserved.
    """

    box: BoundingBox
    confidence: Confidence


class EvidenceWindow(_Frozen):
    """The bounded temporal span a verifier and Memory both need.

    Distinct from `visual_memory_memory_contract.protocol.Evidence`: this is
    Vision-internal and pre-upload -- a span plus how many sampled frames fall
    inside it, not yet a stored `evidence_id`. `evidence_ids` is populated only
    after a candidate is confirmed and `put_evidence()` has run; it stays empty
    while a candidate is pending verification.
    """

    window_started_at: UtcTimestamp
    window_ended_at: UtcTimestamp
    frame_count: int = Field(ge=1)
    evidence_ids: tuple[str, ...] = ()


class CandidateEvent(_Frozen):
    """A placement or pickup Vision proposes, before verification.

    Never becomes a canonical `Observation` on its own -- per docs/06 rule 8,
    "a candidate event is not a trusted observation until its required
    verification succeeds." Only a `confirmed` `VerifierResult` for this
    `candidate_id` authorizes `emit/memory.py` to construct one.
    """

    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    session_id: str
    device_id: str
    #: The LiveKit track SID this was derived from. Always populated -- unlike
    #: `Observation.media_epoch_id`, a `CandidateEvent` is never a manual
    #: correction or a backfill; it exists only because a frame produced it.
    media_epoch_id: str
    #: Unique only within `(session_id, media_epoch_id)` -- the epoch rule.
    #: Never joined across a reconnect; see `domain/stability.py`'s epoch reset.
    track_id: str
    label: str
    action: CandidateAction
    window: EvidenceWindow
    object_candidate: Detection
    hand_candidate: HandCandidate | None = None
    room_candidate: str | None = None
    surface_candidate: str | None = None
    #: World-space position in metres, when depth and pose were both
    #: available at confirmation time. `None` on the image-space-only path.
    world_point: Point3D | None = None
    detector: DetectorRef
    tracker: DetectorRef
    #: `None` when no depth adapter ran for this candidate -- the image-space
    #: stability path never invokes one.
    depth_model: DetectorRef | None = None
    state_machine_version: str
    pipeline_version: str


class VerifierResult(_Frozen):
    """What a verifier returns for one candidate. Exactly one of three outcomes.

    `unverified` covers "evidence is missing, the verifier timed out or
    failed, JSON is invalid after the allowed repair, or the result is
    inconclusive" per docs/06 -- all of these collapse to the same outcome
    because `emit/memory.py` treats them identically: retained as a
    diagnostic, never promoted.
    """

    candidate_id: str
    outcome: VerifierOutcome
    #: A short machine-readable reason, e.g. `"below_confidence_threshold"`,
    #: `"evidence_missing"`, `"moved_during_settling"`. Free text is not
    #: accepted here; docs/06 requires reason codes, not prose, on candidate
    #: and verifier records.
    reason_code: str
    latency_ms: float = Field(ge=0.0)
    verifier: DetectorRef
    prompt_version: str | None = None
    occurred_at: UtcTimestamp
    #: What the verifier concluded actually happened, when that differs from
    #: what the candidate claimed. A `vanished` candidate is a question and
    #: **must** be resolved through this field before it can become an
    #: observation; for the other actions it is normally `None`, meaning "the
    #: claim stands as made."
    #:
    #: Never `vanished` itself: resolving a question into the same question
    #: answers nothing. A verifier that cannot tell returns `unverified`.
    resolved_action: CandidateAction | None = None
    #: The verifier's own words about where the object is -- "on a white desk,
    #: next to a tablet". Prose, deliberately, and deliberately separate from
    #: `reason_code`, which stays machine-readable.
    #:
    #: This is the honest form of an answer a human asked for in the first
    #: place. A room name and a surface derived from geometry would be more
    #: structured and less useful: nobody recognises "surface at (1.2, 0.7)".
    description: str | None = None


class OverlayTrack(_Frozen):
    """One tracked object as it looked in a single frame.

    Deliberately not a `Detection`: this describes what a *viewer* needs to draw
    a box and label, not what a verifier needs to reason about. `motion_state`
    is here and nowhere else in this module's wire shapes, because seeing a box
    change colour as an object settles is the clearest possible evidence that a
    state machine is running rather than a detector firing.
    """

    track_id: str
    label: str
    confidence: Confidence
    #: Normalized 0..1, so a viewer scales to whatever size it renders at.
    box: BoundingBox
    motion_state: MotionState
    #: Metric range along the view ray -- how far away the object is. Populated
    #: only when a depth adapter is configured; `None` is the honest
    #: image-space-only shape, not a missing measurement.
    depth_m: float | None = None
    #: How old that reading is, in seconds.
    #:
    #: Depth is sampled at a low cadence rather than every frame, because a
    #: second heavy model per frame costs more than the measurement is worth
    #: for a quantity that changes slowly. So a viewer is told the age and can
    #: say so: a number presented as live when it is four seconds stale is a
    #: worse failure than no number at all.
    depth_age_s: float | None = None


class OverlayFrame(_Frozen):
    """Everything a viewer needs to draw one frame's detections.

    Carries no pixels. A browser publishing its own camera already has the
    frames, so sending coordinates costs kilobytes where an annotated video
    track would cost megabits and a second encode.

    This is telemetry, not a contract between services: it is droppable by
    design, and a consumer that misses one simply draws the next.
    """

    schema_version: str = SCHEMA_VERSION
    session_id: str
    media_epoch_id: str
    #: The source frame's sequence number, so a viewer can tell a repeated
    #: overlay from a new one and measure how many it dropped.
    sequence: int
    captured_at: UtcTimestamp
    #: When the relay handed this frame to the pipeline.
    relayed_at: UtcTimestamp
    emitted_at: UtcTimestamp
    width: int
    height: int
    tracks: tuple[OverlayTrack, ...] = ()
    #: `emitted_at - relayed_at`. Both are stamped by the same process, so this
    #: is a real measurement of how long detection took rather than a figure
    #: skewed by a viewer's clock. Showing it is the point: a demo has to prove
    #: the work is happening live, and a number that moves is that proof.
    pipeline_latency_ms: float = Field(ge=0.0)


__all__ = [
    "MEMORY_ACTIONS",
    "MemoryAction",
    "SCHEMA_VERSION",
    "BoundingBox",
    "Box3D",
    "CandidateAction",
    "CandidateEvent",
    "Confidence",
    "Detection",
    "DetectorRef",
    "EvidenceWindow",
    "HandCandidate",
    "MotionState",
    "OverlayFrame",
    "OverlayTrack",
    "Point2D",
    "Point3D",
    "TrackSample",
    "UtcTimestamp",
    "VerifierOutcome",
    "VerifierResult",
    "is_memory_action",
]
