"""Identifier minting, shared so producer and consumer agree on the shapes.

ULIDs are lexicographically sortable by creation time, so a listing sorts
correctly without a join and a tie-break on id is a tie-break on creation order.
That property is load-bearing in the reducer: observations with an identical
`occurred_at` must still reduce deterministically.
"""

from __future__ import annotations

from ulid import ULID

OBSERVATION_PREFIX = "obs"
SIGNAL_PREFIX = "lc"
OBJECT_PREFIX = "object"
EVIDENCE_PREFIX = "ev"
EVENT_PREFIX = "event"
VIEW_PREFIX = "view"


def new_observation_id() -> str:
    """Mint an observation identifier, e.g. `obs_01JAB...`."""
    return f"{OBSERVATION_PREFIX}_{ULID()}"


def new_signal_id() -> str:
    return f"{SIGNAL_PREFIX}_{ULID()}"


def new_object_id() -> str:
    return f"{OBJECT_PREFIX}_{ULID()}"


def new_evidence_id() -> str:
    return f"{EVIDENCE_PREFIX}_{ULID()}"


def new_event_id() -> str:
    return f"{EVENT_PREFIX}_{ULID()}"


def new_view_id() -> str:
    return f"{VIEW_PREFIX}_{ULID()}"


def observation_idempotency_key(
    *, device_id: str, session_id: str, track_id: str, action: str, occurred_at: str
) -> str:
    """Build the key a producer should send for a perception event.

    Deterministic on purpose: a Vision service that retries after a timeout must
    produce the same key, or the retry lands as a second observation and the
    timeline gains an event that never happened.
    """
    return f"{device_id}/{session_id}/{track_id}/{action}/{occurred_at}"


def lifecycle_idempotency_key(
    *, device_id: str, session_id: str, media_epoch_id: str | None, action: str
) -> str:
    """Build the key an emitter should send for a lifecycle signal.

    Matches `docs/06`: `{device_id}/{session_id}/{media_epoch_id}/{action}`.
    A gateway that restarts mid-teardown re-sends the same key and Memory
    ignores the repeat rather than applying the transition twice.
    """
    return f"{device_id}/{session_id}/{media_epoch_id or '-'}/{action}"


__all__ = [
    "EVENT_PREFIX",
    "EVIDENCE_PREFIX",
    "OBJECT_PREFIX",
    "OBSERVATION_PREFIX",
    "SIGNAL_PREFIX",
    "VIEW_PREFIX",
    "lifecycle_idempotency_key",
    "new_event_id",
    "new_evidence_id",
    "new_object_id",
    "new_observation_id",
    "new_signal_id",
    "new_view_id",
    "observation_idempotency_key",
]
