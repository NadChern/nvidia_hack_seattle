"""Canonical observation, lifecycle, and answer contract.

`docs/06-Data-Contract.md` is the normative definition; these models are its
executable form. Vision and Speech depend on this package to produce
observations and to read answers; the Memory Service depends on it to consume
them. Both sides assert against the same fixtures, which is what makes an
interface complete per `docs/05-Team-Split.md`.
"""

from visual_memory_memory_contract.protocol import (
    LIFECYCLE_SCHEMA_VERSION,
    OBJECT_REGISTRY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AnswerStatus,
    CurrentStatus,
    EnrolledObject,
    EventAction,
    LifecycleEnvelope,
    ObjectGallery,
    ObjectState,
    ObjectView,
    ObjectViewQuality,
    ObjectViewUpload,
    Observation,
    QueryRequest,
    QueryResponse,
)

__version__ = "0.1.0"

__all__ = [
    "LIFECYCLE_SCHEMA_VERSION",
    "OBJECT_REGISTRY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "AnswerStatus",
    "EnrolledObject",
    "CurrentStatus",
    "EventAction",
    "LifecycleEnvelope",
    "ObjectGallery",
    "ObjectState",
    "ObjectView",
    "ObjectViewQuality",
    "ObjectViewUpload",
    "Observation",
    "QueryRequest",
    "QueryResponse",
    "__version__",
]
