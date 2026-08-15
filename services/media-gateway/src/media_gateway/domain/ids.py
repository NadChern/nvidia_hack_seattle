"""Identifier minting.

ULIDs are lexicographically sortable by creation time, so ids sort into
chronological order in logs and database indexes without a separate timestamp
column.

The gateway is the minting authority for `session_id` because it is the only
component that observes a session start: the device asks it for a token. The
Memory Service remains the authority for session persistence and deletion.
"""

from __future__ import annotations

from ulid import ULID

SESSION_PREFIX = "sess"
SIGNAL_PREFIX = "lc"


def new_session_id() -> str:
    """Mint a session identifier, e.g. `sess_01JAB...`."""
    return f"{SESSION_PREFIX}_{ULID()}"


def new_signal_id() -> str:
    """Mint a lifecycle signal identifier, e.g. `lc_01JAB...`."""
    return f"{SIGNAL_PREFIX}_{ULID()}"


def lifecycle_idempotency_key(
    *, device_id: str, session_id: str, scope_id: str, action: str
) -> str:
    """Build the deterministic key for a lifecycle signal.

    Deterministic so a gateway restart part-way through teardown cannot cause
    the Memory Service to apply the same transition twice.
    """
    return f"{device_id}/{session_id}/{scope_id}/{action}"


__all__ = [
    "SESSION_PREFIX",
    "SIGNAL_PREFIX",
    "lifecycle_idempotency_key",
    "new_session_id",
    "new_signal_id",
]
