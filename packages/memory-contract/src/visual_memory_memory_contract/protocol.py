"""The canonical observation, lifecycle, and answer models.

`docs/06-Data-Contract.md` is the normative definition. These models are the
executable form of it and the single source of truth for field names; when the
two disagree, one of them is a bug and `tests/test_docs_contract.py` says so.

This is the contract the Vision Service *produces* and the Memory Service
consumes. It is deliberately not the media relay contract -- that carries
decoded frames and lives in `packages/media-contract`. A relay message is never
an observation: it has no object, no location, and no confidence, because the
gateway observes none of those.

Models are frozen and ignore unknown fields, so a producer pinned to an older
minor version keeps working when Memory starts accepting an additional one.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

#: Additive optional fields are a minor bump; removals, renames, and semantic
#: changes are a major bump. 1.1 added `media_epoch_id`.
SCHEMA_VERSION = "1.2"

#: Lifecycle envelopes version independently of observations: they are a
#: different shape with a different producer.
LIFECYCLE_SCHEMA_VERSION = "1.0"


def _utc_iso(value: dt.datetime) -> str:
    """UTC ISO-8601 with a Z suffix and millisecond precision.

    `docs/01-Recommended-Architecture.md` requires UTC ISO-8601 everywhere.
    Fixed precision keeps golden fixtures byte-stable.
    """
    return value.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


UtcTimestamp: TypeAlias = Annotated[AwareDatetime, PlainSerializer(_utc_iso, return_type=str)]

#: Perception events. An unknown action is rejected rather than ignored.
EventAction: TypeAlias = Literal["observed", "picked_up", "carried", "placed"]

#: Emitted by the Media Gateway or by Memory itself, never by perception.
LifecycleAction: TypeAlias = Literal["track_lost", "session_ended"]

CurrentStatus: TypeAlias = Literal["confirmed_at_location", "in_transit", "unknown"]

LocationRelation: TypeAlias = Literal[
    "on", "in", "under", "beside", "in_front_of", "behind", "unknown"
]

AnswerStatus: TypeAlias = Literal["confirmed", "last_confirmed_only", "unknown", "ambiguous_object"]

#: Why a lifecycle signal fired. The union of the gateway's epoch-end and
#: session-end reasons; `packages/media-contract` holds the same values for the
#: relay's in-band copy.
LifecycleReason: TypeAlias = Literal[
    "track_unsubscribed",
    "participant_disconnected",
    "room_disconnected",
    "session_ended",
    "session_deleted",
    "session_ttl_expired",
    "gateway_shutdown",
]

Confidence: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class ObjectRef(BaseModel):
    """Which object an observation is about.

    `object_id` may be null at ingestion when identity is unresolved; the label
    and the session-scoped `track_id` remain required. Only observations
    resolved to one stable object may be promoted to trusted state.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    object_id: str | None = None
    label: str
    #: Unique only within `(session_id, media_epoch_id)`. Never join tracker ids
    #: across epochs -- a reconnect restarts the numbering, so `track-42` before
    #: and after a dropout are different physical objects.
    track_id: str


class EventDetail(_Frozen):
    action: EventAction
    source: str
    occurred_at: UtcTimestamp
    window_started_at: UtcTimestamp | None = None
    window_ended_at: UtcTimestamp | None = None


class Location(_Frozen):
    """Where the object was observed.

    Unknown fields must be null or the explicit `unknown` enum. They must never
    be filled with a guessed label to satisfy the schema -- a guessed room is
    indistinguishable from an observed one once it is stored.
    """

    room: str | None = None
    surface: str | None = None
    relation: LocationRelation | None = None
    description: str | None = None


class ObservationConfidence(_Frozen):
    event: Confidence
    identity: Confidence
    room: Confidence | None = None
    surface: Confidence | None = None
    relation: Confidence | None = None


class Evidence(_Frozen):
    """A stored frame supporting an observation.

    `sha256` covers the bytes. The server assigns the storage path; clients must
    not submit local file paths.
    """

    evidence_id: str
    captured_at: UtcTimestamp
    media_type: str = "image/jpeg"
    sha256: str
    frame_index: int | None = None


class DetectorRef(_Frozen):
    name: str
    checkpoint: str
    revision: str


class Provenance(_Frozen):
    """What produced this observation, precisely enough to reproduce it."""

    detector: DetectorRef | None = None
    geometry: DetectorRef | None = None
    verifier: DetectorRef | None = None
    prompt_version: str | None = None
    pipeline_version: str


class Observation(_Frozen):
    """The canonical ingestion payload.

    The server stamps `ingested_at`, assigns stored evidence paths, and records
    validation results. Nothing here is trusted state until the reducer
    promotes it.
    """

    schema_version: str = SCHEMA_VERSION
    observation_id: str
    #: A repeat returns the original result rather than re-applying.
    idempotency_key: str
    session_id: str
    device_id: str
    #: The LiveKit track SID this was derived from. Null only for an
    #: observation with no media provenance -- a manual correction, a backfill,
    #: or a fixture. Never null for one derived from a video frame.
    media_epoch_id: str | None = None
    object: ObjectRef
    event: EventDetail
    location: Location | None = None
    confidence: ObservationConfidence
    evidence: tuple[Evidence, ...] = ()
    provenance: Provenance

    @model_validator(mode="after")
    def _placement_needs_a_location(self) -> Observation:
        """A `placed` event with no location cannot become a placement.

        Accepting one would create trusted state that answers "where?" with
        null, which is worse than refusing the observation.
        """
        if self.event.action == "placed" and self.location is None:
            raise ValueError("a 'placed' observation requires a location")
        return self


class LifecycleScope(_Frozen):
    """The blast radius of a lifecycle signal.

    A transport component knows nothing about objects, so it scopes by media
    epoch: the transition applies to every object whose in-transit state
    originated there. Memory performs the fan-out, because Memory is the only
    component that knows which objects those are.
    """

    media_epoch_id: str | None = None
    object_id: str | None = None
    track_id: str | None = None


class LifecycleDetail(_Frozen):
    action: LifecycleAction
    source: str = "media_gateway"
    occurred_at: UtcTimestamp
    reason: LifecycleReason


class LifecycleProvenance(_Frozen):
    component: str
    version: str
    protocol_version: str | None = None


class LifecycleEnvelope(_Frozen):
    """What an emitter posts to Memory when a track or session goes away.

    Deliberately not an observation: no object, location, confidence, or
    evidence, because the emitter observes none of those. `docs/06` § Lifecycle
    signals explains why the observation envelope cannot carry this.

    `packages/media-contract` defines the same shape for the gateway's in-band
    relay copy. The two are kept in step by both packages validating against the
    same JSON block in `docs/06` rather than by importing one another, which
    would drag media dependencies into every consumer of this package.
    """

    schema_version: str = LIFECYCLE_SCHEMA_VERSION
    signal_id: str
    #: Deterministic -- `{device_id}/{session_id}/{media_epoch_id}/{action}` --
    #: so an emitter that restarts mid-teardown cannot double-apply.
    idempotency_key: str
    session_id: str
    device_id: str
    signal: LifecycleDetail
    scope: LifecycleScope
    provenance: LifecycleProvenance


class _PlacementCore(_Frozen):
    """What both the stored and the answered form of a placement carry."""

    occurred_at: UtcTimestamp
    room: str | None = None
    surface: str | None = None
    relation: LocationRelation | None = None
    evidence_id: str | None = None


class AnsweredPlacement(_PlacementCore):
    """A placement as reported in an answer.

    Carries no `event_id`, matching the query response example in `docs/06`.
    The distinction from `Placement` is deliberate rather than an oversight: an
    internal event identifier is meaningful to the reducer and meaningless to
    the conversational layer, which needs the location, the time, and something
    it can show the user.
    """

    #: Where to fetch the evidence. Additive in schema 1.2 -- an id alone made
    #: every consumer hard-code the route, and a UI that has to build URLs from
    #: ids is a UI that breaks when the route moves.
    #:
    #: Populated **only when the bytes are actually retrievable**. A URL that
    #: 404s is worse than no URL: it looks like evidence right up until someone
    #: clicks it. This is also why the field lives here and not on `Placement`:
    #: trusted state is durable, a URL is not, and a stored URL would outlive
    #: both the file it points at and the route that served it.
    evidence_url: str | None = None
    #: `image/jpeg`, `video/mp4`, and so on. Lets a client choose between an
    #: <img> and a <video> without sniffing the response.
    evidence_media_type: str | None = None


class Placement(_PlacementCore):
    """A confirmed placement in trusted state, preserved even once invalidated.

    Adds the originating `event_id` so a state transition can be traced back to
    the observation that caused it.
    """

    event_id: str


class LastSeen(_Frozen):
    occurred_at: UtcTimestamp
    room: str | None = None
    evidence_id: str | None = None


class ObjectState(_Frozen):
    """Trusted current state: what the assistant is allowed to claim."""

    object_id: str
    current_status: CurrentStatus
    current_location: Location | None = None
    current_event_id: str | None = None
    state_reason: str | None = None
    invalidated_at: UtcTimestamp | None = None
    last_confirmed_placement: Placement | None = None
    last_seen: LastSeen | None = None
    updated_at: UtcTimestamp


class QueryRequest(_Frozen):
    """Ask about one object, by id or by label."""

    object_id: str | None = None
    label: str | None = None
    session_id: str | None = None

    @model_validator(mode="after")
    def _needs_an_object(self) -> QueryRequest:
        if self.object_id is None and self.label is None:
            raise ValueError("a query needs either object_id or label")
        return self


class QueryResponse(_Frozen):
    """The answer contract.

    The conversational layer may shorten `spoken_answer`, but it must preserve
    `answer_status`, the uncertainty, and any invalidation. Dropping the second
    half of "I last confirmed them there, but they were picked up afterward"
    turns a truthful answer into a false one.
    """

    object_id: str | None = None
    answer_status: AnswerStatus
    current_status: CurrentStatus | None = None
    current_location: Location | None = None
    last_confirmed_placement: AnsweredPlacement | None = None
    last_confirmed_placement_confidence: Confidence | None = None
    spoken_answer: str
    #: Populated only for `ambiguous_object`, so a caller can disambiguate
    #: rather than guess.
    candidates: tuple[str, ...] = ()


__all__ = [
    "LIFECYCLE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "AnswerStatus",
    "AnsweredPlacement",
    "Confidence",
    "CurrentStatus",
    "DetectorRef",
    "EventAction",
    "EventDetail",
    "Evidence",
    "LastSeen",
    "LifecycleAction",
    "LifecycleDetail",
    "LifecycleEnvelope",
    "LifecycleProvenance",
    "LifecycleReason",
    "LifecycleScope",
    "Location",
    "LocationRelation",
    "ObjectRef",
    "ObjectState",
    "Observation",
    "ObservationConfidence",
    "Placement",
    "Provenance",
    "QueryRequest",
    "QueryResponse",
    "UtcTimestamp",
]
