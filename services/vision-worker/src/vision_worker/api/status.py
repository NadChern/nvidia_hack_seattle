"""Observable state for dashboards and release reports.

docs/04-Evaluation-Plan.md requires the configuration used for an evaluation
run to be recorded rather than baked into code -- this endpoint is where the
reasoner window, the identity gate, and the live counters are visible, matching
how the Memory Service reports its `PromotionPolicy` at its own `/v1/status`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from fastapi import APIRouter, Request

from vision_worker import __version__
from vision_worker.consume.relay import RelayConsumer
from vision_worker.deps import authorize_request
from vision_worker.identity.gallery import GalleryCache
from vision_worker.pipeline import Pipeline

router = APIRouter(tags=["status"])


@router.get("/v1/status")
def status(request: Request) -> dict[str, Any]:
    """Report configuration, the identity gate, and pipeline counters."""
    authorize_request(request)

    state = request.app.state
    settings = state.settings
    pipeline: Pipeline = state.pipeline
    gallery: GalleryCache | None = getattr(state, "gallery", None)
    embedder = getattr(state, "embedder", None)
    enrollment_manager = getattr(state, "enrollment_manager", None)
    consumer: RelayConsumer | None = getattr(state, "relay_consumer", None)

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
            "reason_kind": settings.reason_kind,
            "identity_kind": settings.identity_kind,
            "evidence_ring_seconds": settings.evidence_ring_seconds,
            "clip_fps": settings.resolved_clip_fps,
            "registration_capture_seconds": settings.registration_capture_seconds,
            "registration_target_views": settings.registration_target_views,
            "registration_min_views": settings.registration_min_views,
        },
        # The reasoner window and cadence in effect -- an evaluation run cites
        # this, not the code. Cosmos is slow per call, so window/interval are
        # the levers that decide how the loop keeps up with the stream.
        "reasoner": {
            "kind": settings.reason_kind,
            "model": settings.reason_model,
            "base_url": settings.reason_base_url,
            "window_seconds": settings.reason_window_seconds,
            "interval_seconds": settings.reason_interval_seconds,
            "max_frames": settings.reason_max_frames,
            "event_cooldown_seconds": settings.event_cooldown_seconds,
            "promote_motion_events": settings.promote_motion_events,
        },
        # The identity gate: only objects matching a registered gallery entry
        # are written. `min_cosine` is the accept threshold; the gallery counts
        # say how many objects Cosmos is even asked to look for.
        "identity": {
            "embedder": (
                embedder.readiness_payload()
                if embedder is not None and hasattr(embedder, "readiness_payload")
                else {"name": type(embedder).__name__ if embedder is not None else "none"}
            ),
            "min_cosine": settings.identity_min_cosine,
            "summary_weight": settings.identity_summary_weight,
            "box_padding": settings.identity_box_padding,
            "gallery": (
                gallery.status_payload()
                if gallery is not None
                else {"registry_version": 0, "gallery_objects": 0, "gallery_views": 0}
            ),
        },
        "registration": (
            enrollment_manager.status_payload()
            if enrollment_manager is not None
            else {"attempts": 0, "succeeded": 0, "failed": 0, "active": 0}
        ),
        # Window analysis runs off the frame loop, so it can fall behind the
        # stream without anything else looking wrong. `pending` climbing and not
        # returning is that happening; `dropped` is that having already cost a
        # real event, and must be zero.
        "analysis": {
            "queue_depth": settings.verification_queue_depth,
            "pending": pipeline.pending_analyses,
            "dropped": pipeline.analyses_dropped,
            "failed": pipeline.analyses_failed,
        },
        # Frames the pipeline could not keep up with, superseded before anything
        # looked at them -- the counter that says analysis is slower than the
        # stream.
        "ingest": {
            "frames_dropped_stale": consumer.frames_dropped if consumer else 0,
            "control_dropped": consumer.control_dropped if consumer else 0,
        },
        "metrics": dataclasses.asdict(pipeline.metrics),
    }
