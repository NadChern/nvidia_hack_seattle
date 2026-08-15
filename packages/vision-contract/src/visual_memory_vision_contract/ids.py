"""Identifier minting for candidate events.

Mirrors `visual_memory_memory_contract.ids` exactly: ULIDs are
lexicographically sortable by creation time, so a candidate listing sorts
correctly with no join, and a tie-break on id is a tie-break on creation order.
"""

from __future__ import annotations

from ulid import ULID

CANDIDATE_PREFIX = "cand"


def new_candidate_id() -> str:
    """Mint a candidate identifier, e.g. `cand_01JAB...`."""
    return f"{CANDIDATE_PREFIX}_{ULID()}"


__all__ = ["CANDIDATE_PREFIX", "new_candidate_id"]
