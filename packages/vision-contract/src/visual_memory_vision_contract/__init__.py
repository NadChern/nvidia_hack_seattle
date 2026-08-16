"""Candidate-event and verifier contract produced by the Vision Service.

`docs/06-Data-Contract.md` § Candidate verification boundary is the normative
definition; these models are its executable form. Person 2's verifier
implementation depends on this package to consume `CandidateEvent`s and
produce `VerifierResult`s. The Memory Service never sees this package -- a
confirmed candidate crosses into `visual_memory_memory_contract.protocol.
Observation`, constructed by `emit/memory.py` inside the Vision Service.
"""

from visual_memory_vision_contract.protocol import (
    SCHEMA_VERSION,
    BoundingBox,
    Box3D,
    CandidateAction,
    CandidateEvent,
    Confidence,
    Detection,
    DetectorRef,
    EvidenceWindow,
    HandCandidate,
    IdentityMatch,
    MotionState,
    OverlayFrame,
    OverlayTrack,
    Point2D,
    Point3D,
    TrackSample,
    VerifierOutcome,
    VerifierResult,
)

__version__ = "0.1.0"

__all__ = [
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
    "IdentityMatch",
    "MotionState",
    "OverlayFrame",
    "OverlayTrack",
    "Point2D",
    "Point3D",
    "TrackSample",
    "VerifierOutcome",
    "VerifierResult",
    "__version__",
]
