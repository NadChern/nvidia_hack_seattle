"""What every verifier implements: a candidate in, exactly one of three
outcomes out.

`docs/06-Data-Contract.md` rule 8: "a candidate event is not a trusted
observation until its required verification succeeds." Two adapters exist
behind this interface: `rules.py`, deterministic and the default, and a
future VLM adapter (Qwen3-VL or similar) that Person 2 owns per
`docs/05-Team-Split.md`. Swapping one for the other changes nothing above
this interface -- `emit/memory.py` only ever sees a `VerifierResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import CandidateEvent, TrackSample, VerifierResult


class Verifier(Protocol):
    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: Sequence[bytes],
        samples: Sequence[TrackSample] = (),
        decoded: Sequence[NDArray[np.uint8]] = (),
    ) -> VerifierResult:
        """Judge one candidate.

        `frames` are the window's raw sampled-frame bytes (JPEG), held
        in-process by the caller -- see `evidence/ring.py`. They are never
        embedded in `CandidateEvent` itself, which must stay a small,
        JSON-serializable contract object per docs/06's example; a
        rule-based verifier may ignore `frames` entirely, and a VLM verifier
        is what they exist for.

        `samples` are this track's own observations across the window, oldest
        first -- where the object was in each frame, which `frames` alone
        cannot say. `verify/world_motion.py` is what they exist for: judging
        whether an object moved needs its position over time, not one box.

        `decoded` are the same frames as arrays, decoded once by the caller
        rather than once per verifier. Both default to empty so a verifier
        that needs neither -- `rules.py` -- keeps a two-argument call site.

        `candidate.window.evidence_ids` is deliberately still empty at this
        point -- per `EvidenceWindow`'s docstring, it is populated only after
        confirmation, once `emit/memory.py` has actually uploaded bytes to
        the Memory Service. Evidence *presence* during verification is
        signaled by `frames` being non-empty, not by that field.
        """
        ...


__all__ = ["Verifier"]
