"""The deterministic verifier -- the default, and the fallback S04 names.

`docs/09-Spike-Plan.md` S04's stop condition: if a model-based path fails,
"substitute... a conservative rule-based candidate generator." This is that
fallback, built as the default rather than an afterthought, so a model
setback costs a config swap rather than a missing verifier.

Person 2 (spatial verification and evaluation, per docs/05) replaces this
module with a VLM adapter and nothing else in the pipeline changes -- see
`verify/base.py`.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import (
    CandidateEvent,
    DetectorRef,
    TrackSample,
    VerifierResult,
)

_VERIFIER_REF = DetectorRef(name="rules", checkpoint="n/a", revision="v1")

#: Reason codes this verifier can produce. A VLM verifier may use additional
#: ones -- docs/06 requires a reason code, not a fixed enum of them.
EVIDENCE_MISSING = "evidence_missing"
WINDOW_INCOMPLETE = "window_incomplete"
BELOW_CONFIDENCE_THRESHOLD = "below_confidence_threshold"
CONFIRMED = "meets_confidence_and_evidence_thresholds"


@dataclass(frozen=True, slots=True)
class RuleBasedVerifierConfig:
    """Thresholds this verifier judges against.

    Configuration, not constants -- reported at `/v1/status`, matching how
    the Memory Service reports its `PromotionPolicy` and the stability
    machine reports its `StabilityConfig`.
    """

    min_confidence: float = 0.6
    #: A window with fewer sampled frames than this is treated as too thin
    #: to trust, regardless of confidence.
    min_frame_count: int = 1


class RuleBasedVerifier:
    """Confirms a candidate when its evidence exists, its window is not
    trivially thin, and its detection confidence clears a threshold.

    Deliberately does not attempt "rejected: the object moved again inside
    the settling window" -- that case cannot arise from this pipeline's
    stability machine, which never emits a `placed` candidate for a track
    that moved during settling in the first place (see
    `domain/stability.py`). A verifier with access to the window's frames
    could still detect drift within them and reject on that basis; this one
    does not look at `frames` at all.
    """

    def __init__(self, config: RuleBasedVerifierConfig | None = None) -> None:
        self._config = config or RuleBasedVerifierConfig()

    @property
    def config(self) -> RuleBasedVerifierConfig:
        return self._config

    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: Sequence[bytes],
        samples: Sequence[TrackSample] = (),
        decoded: Sequence[NDArray[np.uint8]] = (),
    ) -> VerifierResult:
        # Neither the track's trajectory nor the decoded pixels are consulted
        # here -- see the class docstring. Accepted so this stays substitutable
        # for a verifier that does need them.
        del samples, decoded
        started = time.perf_counter()

        if not frames:
            return self._result(candidate, "unverified", EVIDENCE_MISSING, started)
        if candidate.window.frame_count < self._config.min_frame_count:
            return self._result(candidate, "unverified", WINDOW_INCOMPLETE, started)
        if candidate.object_candidate.confidence < self._config.min_confidence:
            return self._result(candidate, "rejected", BELOW_CONFIDENCE_THRESHOLD, started)

        return self._result(candidate, "confirmed", CONFIRMED, started)

    def _result(
        self,
        candidate: CandidateEvent,
        outcome: str,
        reason_code: str,
        started: float,
    ) -> VerifierResult:
        return VerifierResult(
            candidate_id=candidate.candidate_id,
            outcome=outcome,  # type: ignore[arg-type]
            reason_code=reason_code,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            verifier=_VERIFIER_REF,
            occurred_at=dt.datetime.now(dt.UTC),
        )


__all__ = [
    "BELOW_CONFIDENCE_THRESHOLD",
    "CONFIRMED",
    "EVIDENCE_MISSING",
    "WINDOW_INCOMPLETE",
    "RuleBasedVerifier",
    "RuleBasedVerifierConfig",
]
