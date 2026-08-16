"""Observable state for dashboards and release reports.

docs/04-Evaluation-Plan.md requires the threshold set used for an evaluation
run to be recorded, which is impossible if it is baked into the reducer or
the state machine -- this endpoint is where those thresholds are visible,
matching how the Memory Service reports its `PromotionPolicy` at its own
`/v1/status`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from fastapi import APIRouter, Request

from vision_worker import __version__
from vision_worker.consume.relay import RelayConsumer
from vision_worker.deps import authorize_request
from vision_worker.overlay.hub import OverlayHub
from vision_worker.pipeline import Pipeline
from vision_worker.verify.rules import RuleBasedVerifierConfig

router = APIRouter(tags=["status"])


@router.get("/v1/status")
def status(request: Request) -> dict[str, Any]:
    """Report configuration, live thresholds, and pipeline counters."""
    authorize_request(request)

    state = request.app.state
    settings = state.settings
    pipeline: Pipeline = state.pipeline
    verifier_config: RuleBasedVerifierConfig = state.verifier_config
    overlay_hub: OverlayHub | None = getattr(state, "overlay_hub", None)
    consumer: RelayConsumer | None = getattr(state, "relay_consumer", None)
    detector = getattr(state, "detector", None)
    depth_estimator = getattr(state, "depth_estimator", None)
    identity_resolver = getattr(state, "identity_resolver", None)
    enrollment_manager = getattr(state, "enrollment_manager", None)

    reason = state.readiness.evaluate()
    started_at: dt.datetime = state.started_at

    return {
        "service": settings.service_name,
        "version": __version__,
        "environment": settings.environment,
        "ready": reason is None,
        "not_ready_reason": reason,
        "started_at": started_at,
        "uptime_s": round((dt.datetime.now(dt.UTC) - started_at).total_seconds(), 1),
        "config": {
            "gateway_video_url": settings.gateway_video_url,
            "memory_base_url": settings.memory_base_url,
            "detector_kind": settings.detector_kind,
            "depth_kind": settings.depth_kind,
            "identity_kind": settings.identity_kind,
            # What was *asked for*. `world_motion` degrades to plain rules
            # when DA3 will not load, so this is not proof the world check is
            # running -- `verifier` below reports what actually is.
            "verifier_kind": settings.verifier_kind,
            "detection_labels": list(settings.detection_labels),
            "max_detections_per_frame": settings.max_detections_per_frame,
            "evidence_ring_seconds": settings.evidence_ring_seconds,
            "clip_fps": settings.resolved_clip_fps,
            "source_fps": settings.source_fps,
            "registration_capture_seconds": settings.registration_capture_seconds,
            "registration_target_views": settings.registration_target_views,
            "registration_min_views": settings.registration_min_views,
            "registration_dedup_threshold": settings.registration_dedup_threshold,
        },
        # Both halves of the frame-rate assumption. `source_fps` is what the
        # thresholds below were derived from; `observed_fps` is what the relay
        # is really delivering. They should agree -- if they do not, every
        # frame count below means a different duration than intended, and this
        # is the only place that shows it. `null` until enough frames have
        # arrived to measure.
        "frame_rate": {
            "configured_fps": settings.source_fps,
            "observed_fps": (
                round(observed, 2) if (observed := pipeline.observed_fps) is not None else None
            ),
        },
        # The durations an operator configured. The machine compares these
        # wall-clock seconds directly against sample timestamps, so there is no
        # separate frame-count form to report -- the observed rate (above) only
        # sets how densely samples fall inside each window.
        "stability_durations_s": {
            "dwell": settings.dwell_seconds,
            "passive_confirmation": settings.passive_confirmation_seconds,
            "reacquire_within": settings.reacquire_within_seconds,
            "carried_emit_interval": settings.carried_emit_interval_seconds,
        },
        # The thresholds actually in effect (seconds), not just the settings
        # that produced them -- an evaluation run cites this, not the code.
        "stability_thresholds": dataclasses.asdict(pipeline.track_registry.config),
        "verifier_thresholds": dataclasses.asdict(verifier_config),
        # The verifier actually in effect, which is not always the one
        # configured -- see `verifier_kind` above.
        "models": {
            "detector": (
                detector.readiness_payload()
                if detector is not None and hasattr(detector, "readiness_payload")
                else {"name": type(detector).__name__ if detector is not None else "none"}
            ),
            "depth": (
                depth_estimator.readiness_payload()
                if depth_estimator is not None and hasattr(depth_estimator, "readiness_payload")
                else {
                    "name": type(depth_estimator).__name__
                    if depth_estimator is not None
                    else "none"
                }
            ),
        },
        "identity": (
            identity_resolver.status_payload()
            if identity_resolver is not None
            else {
                "enabled": False,
                "resolved": 0,
                "ambiguous": 0,
                "unmatched": 0,
                "escalated": 0,
                "average_latency_ms": 0.0,
                "gallery_objects": 0,
                "gallery_views": 0,
                "stale_views": 0,
            }
        ),
        "registration": (
            enrollment_manager.status_payload()
            if enrollment_manager is not None
            else {"attempts": 0, "succeeded": 0, "failed": 0, "active": 0}
        ),
        "verifier": type(state.verifier).__name__,
        # Verification runs off the frame loop, so it can fall behind the
        # stream without anything else looking wrong. `pending` climbing and
        # not returning is that happening; `dropped` is that having already
        # cost a real event, and must be zero.
        "verification": {
            "queue_depth": settings.verification_queue_depth,
            "concurrency": settings.verification_concurrency,
            "pending": pipeline.pending_verifications,
            "dropped": pipeline.verifications_dropped,
            "failed": pipeline.verifications_failed,
        },
        # Viewers watching `WS /v1/overlay`. `dropped` climbing means a viewer
        # is not keeping up -- which costs nothing real, since a stale overlay
        # would not have been drawn, but distinguishes a slow browser from a
        # slow pipeline when someone reports that the boxes look laggy.
        "overlay": {
            "enabled": overlay_hub is not None,
            "viewers": overlay_hub.subscriber_count if overlay_hub else 0,
            "max_viewers": settings.overlay_max_viewers,
            "dropped": overlay_hub.dropped if overlay_hub else 0,
        },
        # Frames the detector could not keep up with, superseded by a newer
        # one before anything looked at them. `observed_fps` cannot show this:
        # it measures the captured_at stamps of frames that *are* processed, so
        # it reads correct while the pipeline falls further and further behind.
        # This is the counter that says the detector is slower than the stream.
        "ingest": {
            "frames_dropped_stale": consumer.frames_dropped if consumer else 0,
            "control_dropped": consumer.control_dropped if consumer else 0,
        },
        "metrics": dataclasses.asdict(pipeline.metrics),
    }
