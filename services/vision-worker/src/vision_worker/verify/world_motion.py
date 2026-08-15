"""Judge a candidate by where the object was in the *room*, not on screen.

The failure this exists to stop, measured on `media/clips/01-placed-on-table`
where the keys never move once: the image-space path emitted
`placed, picked_up, placed, picked_up, placed, picked_up`. Every one of those
pickups is the camera panning. No threshold on the image residual separates
the two cases -- tight and head motion reads as object motion, loose and
genuine carrying stops registering at all.

Reconstructing the window through `pose/da3.py` collapses the ambiguity: back
-projected through a real camera pose, the keys' world position on that same
footage drifts 0.4% of scene scale. A stationary object looks stationary no
matter what the head did.

So this wraps an inner verifier (`rules.py`) and adds one question: **over
this candidate's own window, did the object's world position actually change
in the way the candidate claims?** A `picked_up` whose object never moved is
rejected. A `placed` whose object was still drifting is rejected.

No torch here, and none reachable from here -- the reconstruction lives
behind `pose/base.py`'s `WindowPoseSource`, which is why
`tests/test_domain_isolation.py` still passes with this module in `verify/`.

**Degrading is the normal path, not the exception.** No adapter configured,
a checkpoint that failed to load, a window too short to reconstruct, a frame
with no valid range under the object, an OOM -- every one of these returns
the inner verifier's verdict untouched. The world check can only ever
*reject* something the rules accepted; it never promotes anything on its own,
because a reconstruction is evidence about motion, not about identity or
confidence.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import (
    CandidateEvent,
    DetectorRef,
    Point3D,
    TrackSample,
    VerifierResult,
)

from vision_worker.domain.geometry import compute_world_point
from vision_worker.pose.base import WindowGeometry, WindowPoseSource
from vision_worker.verify.base import Verifier

logger = logging.getLogger(__name__)

#: Rejected: the candidate says the object left a resting place, but its
#: world position held still. The camera moved, not the object.
DID_NOT_MOVE = "world_position_unchanged"
#: Rejected: the candidate says the object came to rest, but its world
#: position was still travelling through the window.
STILL_MOVING = "world_position_still_moving"
#: Confirmed, and the world check agreed rather than merely abstaining.
WORLD_AGREES = "world_motion_consistent"

#: Actions asserting the object *left* a resting place. A world position that
#: did not change contradicts them.
_MOTION_ACTIONS = frozenset({"picked_up", "carried"})
#: Actions asserting the object *came to* rest. A world position still moving
#: at the end of the window contradicts them.
_REST_ACTIONS = frozenset({"placed"})


@dataclass(frozen=True, slots=True)
class WorldMotionConfig:
    """Thresholds in two forms, because the reconstruction comes in two.

    When `pose/da3.py` has a metric anchor (`depth/moge.py`), `WindowGeometry.
    is_metric` is True and distances are real -- so the thresholds that get
    used are the ones in **metres**, which a person can reason about and
    argue with. Without an anchor the units are arbitrary and only meaningful
    inside one window, so the **ratio** thresholds apply instead, against
    `scene_scale`.

    Both are kept rather than one converted into the other: a ratio cannot be
    stated in centimetres, and a centimetre threshold means nothing in units
    nobody has fixed.
    """

    #: Metric. Total world drift below this counts as "did not move".
    #: Measured on `media/clips/01-placed-on-table`, keys that never moved
    #: drifted 0.3cm while the camera moved 10cm -- so 3cm is an order of
    #: magnitude above the noise and well below any real pickup.
    still_m: float = 0.03
    #: Metric. Drift across the window's final frames above this counts as
    #: "still moving". Looser than `still_m` on purpose: settling is allowed
    #: to be imperfect, and rejecting a real placement costs more than
    #: accepting a slightly early one.
    settled_m: float = 0.08

    #: The same two thresholds as fractions of scene scale, used when no
    #: metric anchor was available. 2% of a ~1m-deep scene is ~2cm.
    still_ratio: float = 0.02
    settled_ratio: float = 0.06

    #: How many of the window's final samples the settled check looks at.
    settle_tail_samples: int = 3
    #: Below this many usable world points, the window says nothing and the
    #: inner verdict stands unchanged.
    min_world_points: int = 3

    def thresholds_for(self, geometry: WindowGeometry) -> tuple[float, float]:
        """`(still, settled)` in whatever units this reconstruction speaks --
        metres when it is anchored, fractions of `scene_scale` when not."""
        if geometry.is_metric:
            return self.still_m, self.settled_m
        return self.still_ratio * geometry.scene_scale, self.settled_ratio * geometry.scene_scale


class WorldMotionVerifier:
    """Wraps `inner`, and vetoes verdicts the reconstruction contradicts."""

    def __init__(
        self,
        inner: Verifier,
        pose_source: WindowPoseSource,
        *,
        config: WorldMotionConfig | None = None,
    ) -> None:
        self._inner = inner
        self._pose_source = pose_source
        self._config = config or WorldMotionConfig()

    @property
    def config(self) -> WorldMotionConfig:
        return self._config

    async def verify(
        self,
        candidate: CandidateEvent,
        *,
        frames: Sequence[bytes],
        samples: Sequence[TrackSample] = (),
        decoded: Sequence[NDArray[np.uint8]] = (),
    ) -> VerifierResult:
        inner = await self._inner.verify(candidate, frames=frames, samples=samples)
        if inner.outcome != "confirmed":
            # Only ever a veto. Nothing here can rescue a candidate the rules
            # already rejected -- motion evidence says nothing about whether
            # the detection was confident enough to trust.
            return inner
        if not self._pose_source.is_ready or len(decoded) < 2 or len(samples) < 2:
            return inner

        started = time.perf_counter()
        geometry = await self._pose_source.estimate(decoded)
        if geometry is None:
            return inner

        points = _world_track(geometry, samples)
        if len(points) < self._config.min_world_points:
            logger.debug(
                "world check abstained: too few usable world points",
                extra={"candidate_id": candidate.candidate_id, "points": len(points)},
            )
            return inner

        verdict = self._judge(candidate.action, points, geometry)
        if verdict is None:
            return _relabel(inner, WORLD_AGREES, started)
        logger.info(
            "world motion contradicts the candidate",
            extra={
                "candidate_id": candidate.candidate_id,
                "action": candidate.action,
                "drift_m": (
                    round(_distance(points[0], points[-1]), 3) if geometry.is_metric else None
                ),
                "metric": geometry.is_metric,
            },
        )
        return _rejected(inner, verdict, started)

    def _judge(
        self, action: str, points: Sequence[Point3D], geometry: WindowGeometry
    ) -> str | None:
        """`None` when the reconstruction is consistent with the claim."""
        if geometry.scene_scale <= 0.0:
            return None
        still, settled = self._config.thresholds_for(geometry)
        total = _distance(points[0], points[-1])

        if action in _MOTION_ACTIONS and total < still:
            return DID_NOT_MOVE

        if action in _REST_ACTIONS:
            tail = points[-self._config.settle_tail_samples :]
            if len(tail) >= 2:
                drift = max(_distance(tail[i], tail[i + 1]) for i in range(len(tail) - 1))
                if drift > settled:
                    return STILL_MOVING
        return None


def _world_track(geometry: WindowGeometry, samples: Sequence[TrackSample]) -> list[Point3D]:
    """The object's world position at each reconstructed view.

    `samples` and `geometry.frames` need not be the same length -- the
    adapter subsamples a long window -- so samples are matched to views by
    even position rather than one-to-one. Both sequences describe the same
    span in the same order, which is what makes that sound.
    """
    views = geometry.frames
    if not views or not samples:
        return []

    points: list[Point3D] = []
    for view_index, view in enumerate(views):
        sample = samples[round(view_index * (len(samples) - 1) / max(len(views) - 1, 1))]
        box = sample.detection.box
        range_m = view.range_in_box(box.x_min, box.y_min, box.x_max, box.y_max)
        if range_m is None or range_m <= 0.0:
            continue
        height, width = view.ranges.shape
        points.append(
            compute_world_point(
                sample.detection.centroid,
                range_m,
                image_width=width,
                image_height=height,
                intrinsics=view.intrinsics,
                capture_pose=view.capture_pose,
            )
        )
    return points


def _distance(a: Point3D, b: Point3D) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _rejected(inner: VerifierResult, reason_code: str, started: float) -> VerifierResult:
    logger.info(
        "world motion contradicts the candidate; rejecting",
        extra={"candidate_id": inner.candidate_id, "reason_code": reason_code},
    )
    return _with(inner, outcome="rejected", reason_code=reason_code, started=started)


def _relabel(inner: VerifierResult, reason_code: str, started: float) -> VerifierResult:
    return _with(inner, outcome="confirmed", reason_code=reason_code, started=started)


def _with(
    inner: VerifierResult, *, outcome: str, reason_code: str, started: float
) -> VerifierResult:
    return VerifierResult(
        candidate_id=inner.candidate_id,
        outcome=outcome,  # type: ignore[arg-type]
        reason_code=reason_code,
        # The inner verifier's own cost plus the reconstruction's, which is
        # the number that matters for "how long does verification take".
        latency_ms=inner.latency_ms + (time.perf_counter() - started) * 1000.0,
        verifier=DetectorRef(name="world-motion", checkpoint="da3", revision="v1"),
        prompt_version=inner.prompt_version,
        occurred_at=dt.datetime.now(dt.UTC),
    )


__all__ = [
    "DID_NOT_MOVE",
    "STILL_MOVING",
    "WORLD_AGREES",
    "WorldMotionConfig",
    "WorldMotionVerifier",
]
