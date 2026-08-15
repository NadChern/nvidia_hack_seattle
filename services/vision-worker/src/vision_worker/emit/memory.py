"""Turns a confirmed candidate into a canonical Observation and posts it to
Memory.

Only ever called for `confirmed` `VerifierResult`s -- `Pipeline` enforces
that boundary and this module trusts it. Rejected and unverified candidates
never reach here and never become a `placed`, `picked_up`, or `carried`
observation, per docs/06 rule 8.

Neither do first sightings: `Pipeline` drops `observed` before a verifier is
ever consulted, so nothing here uploads evidence for an object that was
merely seen. See `pipeline._NON_PROMOTING_ACTIONS`.

**Known limitation until depth and geometry land (tasks #39/#40):** a
`placed` candidate here carries no room or surface -- `CandidateEvent.
room_candidate`/`surface_candidate` stay null with no depth adapter wired.
The resulting `Location` is still valid: memory-contract's validator only
requires a `Location` object to exist for a `placed` action, not that its
fields are populated, and `Location`'s own contract says unknown fields must
stay null rather than be guessed. The resulting answer is an honest "I
confirmed it was placed, but not exactly where" rather than a wrong or a
withheld one.

Order matters: evidence is uploaded *before* `record()`, so the observation
can cite a real stored evidence id rather than a promise of one.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Sequence

from visual_memory_media_contract.framing import payload_digest
from visual_memory_memory_contract.client import MemoryClient, MemoryError_
from visual_memory_memory_contract.ids import new_observation_id, observation_idempotency_key
from visual_memory_memory_contract.protocol import (
    DetectorRef as MemoryDetectorRef,
)
from visual_memory_memory_contract.protocol import (
    EventDetail,
    Evidence,
    Location,
    ObjectRef,
    Observation,
    ObservationConfidence,
    Provenance,
)
from visual_memory_vision_contract.protocol import (
    CandidateEvent,
    DetectorRef,
    MemoryAction,
    VerifierResult,
    is_memory_action,
)

from vision_worker.evidence.clip import (
    CLIP_MEDIA_TYPE,
    ClipEncodeError,
    encode_clip,
    select_still_frame,
)
from vision_worker.evidence.ring import BufferedFrame

logger = logging.getLogger(__name__)

#: Source string every observation this service produces carries, matching
#: the value memory-contract's own fixtures use for the same field.
_SOURCE = "vision_pipeline"


class MemoryEmitter:
    """Uploads evidence and records one observation per confirmed candidate.

    `MemoryClient` is synchronous (it says so itself: "Memory is a
    request/response service, not a stream"), so every call into it here
    runs via `asyncio.to_thread` -- this service's event loop must keep
    consuming relay frames while an HTTP round-trip to Memory is in flight.
    """

    def __init__(
        self,
        client: MemoryClient,
        *,
        clip_fps: float = 24.0,
        min_identity_cosine: float = 0.8334,
        memory_min_identity_confidence: float = 0.7,
    ) -> None:
        self._client = client
        self._clip_fps = clip_fps
        self._min_identity_cosine = min_identity_cosine
        self._memory_min_identity_confidence = memory_min_identity_confidence

    async def emit(
        self,
        candidate: CandidateEvent,
        result: VerifierResult,
        frames: Sequence[BufferedFrame],
    ) -> None:
        if result.outcome != "confirmed":
            # Defensive: Pipeline is the actual boundary enforcing docs/06
            # rule 8, but a caller that ever wires this up incorrectly must
            # not silently create trusted state from a candidate nothing
            # confirmed.
            raise ValueError(
                f"refusing to emit candidate {candidate.candidate_id!r}: "
                f"verifier outcome was {result.outcome!r}, not confirmed"
            )

        action = result.resolved_action or candidate.action
        if not is_memory_action(action):
            # A `vanished` candidate is a question, and this one came back
            # confirmed without an answer. Recording it would assert an event
            # the contract has no word for; refusing costs one diagnostic.
            raise ValueError(
                f"refusing to emit candidate {candidate.candidate_id!r}: "
                f"action {action!r} is not a memory action, and the verifier "
                f"resolved it to nothing"
            )

        evidence = await asyncio.to_thread(self._upload_evidence, candidate, frames)
        observation = self._build_observation(candidate, result, evidence, action)

        try:
            await asyncio.to_thread(self._client.record, observation)
        except MemoryError_:
            logger.exception(
                "memory refused a confirmed observation",
                extra={"candidate_id": candidate.candidate_id, "action": candidate.action},
            )
            raise

    async def emit_last_seen(
        self,
        candidate: CandidateEvent,
        frames: Sequence[BufferedFrame],
    ) -> None:
        """Write one identity-gated `observed` event when a track retires.

        This intentionally bypasses the ordinary first-sighting drop. It is
        bounded to one write per disappearance and requires a registered id.
        """
        if candidate.action != "observed" or candidate.identity is None:
            raise ValueError("last-seen emission requires an identity-annotated observed candidate")
        if candidate.identity.object_id is None:
            raise ValueError("last-seen emission requires a resolved registered object")
        evidence = await asyncio.to_thread(self._upload_still_evidence, candidate, frames)
        result = VerifierResult(
            candidate_id=candidate.candidate_id,
            outcome="confirmed",
            reason_code="registered_track_ended",
            latency_ms=0.0,
            verifier=DetectorRef(
                name="identity-track-end", checkpoint="n/a", revision="last-seen-v1"
            ),
            occurred_at=candidate.window.window_ended_at,
        )
        observation = self._build_observation(candidate, result, evidence, "observed")
        await asyncio.to_thread(self._client.record, observation)

    def _upload_still_evidence(
        self, candidate: CandidateEvent, frames: Sequence[BufferedFrame]
    ) -> tuple[Evidence, ...]:
        if not frames:
            return ()
        still = select_still_frame(frames)
        still_sha256 = payload_digest(still.payload)
        still_id = self._client.put_evidence(
            still.payload, session_id=candidate.session_id, sha256=still_sha256
        )
        return (
            Evidence(
                evidence_id=still_id,
                captured_at=still.captured_at,
                media_type="image/jpeg",
                sha256=still_sha256,
            ),
        )

    def _upload_evidence(
        self, candidate: CandidateEvent, frames: Sequence[BufferedFrame]
    ) -> tuple[Evidence, ...]:
        if not frames:
            # A confirmed candidate with no evidence bytes at all should not
            # happen -- the rule-based verifier already refuses "unverified"
            # on empty frames -- but a smarter verifier could in principle
            # confirm from other signals. Recording with no evidence is
            # honest; inventing a placeholder id would not be.
            return ()

        uploaded = list(self._upload_still_evidence(candidate, frames))

        # The clip is best-effort. A still frame already stands as valid
        # evidence, so an encode or upload failure here degrades to today's
        # single-frame behavior rather than losing the observation entirely.
        try:
            clip_bytes = encode_clip(frames, fps=self._clip_fps)
        except ClipEncodeError:
            logger.warning(
                "evidence clip encode failed; falling back to the still frame alone",
                extra={"candidate_id": candidate.candidate_id},
            )
            return tuple(uploaded)

        clip_sha256 = payload_digest(clip_bytes)
        try:
            clip_id = self._client.put_evidence(
                clip_bytes,
                session_id=candidate.session_id,
                sha256=clip_sha256,
                media_type=CLIP_MEDIA_TYPE,
            )
        except MemoryError_:
            logger.warning(
                "evidence clip upload failed; the still frame still stands",
                extra={"candidate_id": candidate.candidate_id},
            )
            return tuple(uploaded)

        uploaded.append(
            Evidence(
                evidence_id=clip_id,
                captured_at=frames[-1].captured_at,
                media_type=CLIP_MEDIA_TYPE,
                sha256=clip_sha256,
            )
        )
        return tuple(uploaded)

    def _build_observation(
        self,
        candidate: CandidateEvent,
        result: VerifierResult,
        evidence: tuple[Evidence, ...],
        action: MemoryAction,
    ) -> Observation:
        location: Location | None = None
        if action == "placed":
            # `result.description` is the verifier's own words -- "on a white
            # desk, next to a tablet". It goes in `surface` because that is
            # what it describes: a place a person would recognise, which is
            # the thing they asked for. `room` stays null until something
            # actually knows which room this is; guessing it would be the
            # kind of confident invention this service exists to avoid.
            location = Location(
                room=candidate.room_candidate,
                surface=result.description or candidate.surface_candidate,
            )

        occurred_at: dt.datetime = candidate.window.window_ended_at
        return Observation(
            observation_id=new_observation_id(),
            idempotency_key=observation_idempotency_key(
                device_id=candidate.device_id,
                session_id=candidate.session_id,
                track_id=candidate.track_id,
                action=action,
                occurred_at=occurred_at.isoformat(),
            ),
            session_id=candidate.session_id,
            device_id=candidate.device_id,
            media_epoch_id=candidate.media_epoch_id,
            object=ObjectRef(
                object_id=(candidate.identity.object_id if candidate.identity else None),
                label=candidate.label,
                track_id=candidate.track_id,
            ),
            event=EventDetail(
                action=action,
                source=_SOURCE,
                occurred_at=occurred_at,
                window_started_at=candidate.window.window_started_at,
                window_ended_at=candidate.window.window_ended_at,
            ),
            location=location,
            confidence=ObservationConfidence(
                event=candidate.object_candidate.confidence,
                identity=self._identity_confidence(candidate),
            ),
            evidence=evidence,
            provenance=Provenance(
                detector=MemoryDetectorRef(
                    name=candidate.detector.name,
                    checkpoint=candidate.detector.checkpoint,
                    revision=candidate.detector.revision,
                ),
                verifier=MemoryDetectorRef(
                    name=result.verifier.name,
                    checkpoint=result.verifier.checkpoint,
                    revision=result.verifier.revision,
                ),
                prompt_version=result.prompt_version,
                pipeline_version=candidate.pipeline_version,
            ),
        )

    def _identity_confidence(self, candidate: CandidateEvent) -> float:
        identity = candidate.identity
        if identity is None or identity.object_id is None or identity.best_score is None:
            # Identity did not resolve. Keep the exact pre-feature behavior so
            # unregistered objects continue promoting at the same rate.
            return candidate.object_candidate.confidence
        threshold = self._memory_min_identity_confidence
        denominator = max(1e-9, 1.0 - self._min_identity_cosine)
        normalized = max(
            0.0,
            min(1.0, (identity.best_score - self._min_identity_cosine) / denominator),
        )
        return threshold + (1.0 - threshold) * normalized


__all__ = ["MemoryEmitter"]
