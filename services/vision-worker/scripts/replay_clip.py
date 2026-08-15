"""Replay a recorded clip through the real pipeline, with no infrastructure.

    uv run python scripts/replay_clip.py media/clips/02-carried-out.MOV

The plan's end-to-end verification step. `virtual-glasses --file` already
publishes a clip through LiveKit and the gateway, which is the honest test of
the *transport*; this is the complementary one, and the faster loop: it
stands in for the gateway's sampler and feeds `Pipeline` directly, so a clip
exercises detection, tracking, the stability machine, evidence, and the
verifier with no LiveKit, no gateway, no browser, and no Memory Service.

What it is not: a test of the relay. Frames here are built rather than
received, so framing, the dimension guard, epoch semantics on a real
reconnect, and backpressure are all out of scope -- run `virtual-glasses`
for those.

**It samples the clip the way the gateway does**, at `VMA_SOURCE_FPS`, so a
30fps phone recording is decimated exactly as a live stream would be and the
stability thresholds mean the same thing here as in production. Replaying
every frame of the source would silently make every threshold 4x shorter
than a live run, which is the class of bug this script exists to catch.

Nothing is written to Memory unless `--emit` is passed. The default prints
what *would* be recorded, so a clip can be replayed against a laptop with no
Memory Service running.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import io
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import av
from visual_memory_media_contract.framing import payload_digest
from visual_memory_media_contract.protocol import VideoFrame
from visual_memory_memory_contract.client import MemoryClient
from visual_memory_vision_contract.protocol import CandidateEvent, DetectorRef, VerifierResult

from vision_worker.config import Settings
from vision_worker.domain.stability import TrackRegistry
from vision_worker.emit.memory import MemoryEmitter
from vision_worker.evidence.ring import BufferedFrame, EvidenceRing
from vision_worker.logging import configure_logging
from vision_worker.main import (
    PIPELINE_VERSION,
    STATE_MACHINE_VERSION,
    build_depth_estimator,
    build_detector,
    build_stability_config,
    build_verifier,
)
from vision_worker.pipeline import Pipeline
from vision_worker.pose.image_motion import ImageMotionPose
from vision_worker.track.greedy_iou import GreedyIoUTracker

#: One synthetic epoch for the whole clip. A clip is one continuous recording,
#: so it is one media epoch by definition -- `--reconnect-cycles` on
#: `virtual-glasses` is what exercises the boundary between two.
EPOCH_ID = "TR_VCreplay"
SESSION_ID = "sess_replay"

#: The gateway's own default (`VMA_JPEG_QUALITY`), so a frame reaching the
#: detector here is compressed the same way a live one would be.
JPEG_QUALITY = 92

#: Matches what `main.lifespan` records, so a candidate produced by a replay
#: carries the same provenance a live one would.
_TRACKER_REF = DetectorRef(name="greedy-iou", checkpoint="n/a", revision="v1")


def decode_sampled(path: Path, *, sample_fps: float) -> Iterator[tuple[float, bytes, int, int]]:
    """Yield `(presentation_seconds, jpeg_bytes, width, height)` at `sample_fps`.

    Latest-wins decimation against the clip's own presentation timestamps,
    which is what the gateway's sampler does to a live stream. Timestamps come
    from the file, never from the wall clock: a replay must produce the same
    result at any speed, and the stability machine reads `captured_at`.
    """
    interval = 1.0 / sample_fps
    next_at = 0.0

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Let PyAV thread the decode; this is the slowest part of a replay.
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            at_s = float(frame.time) if frame.time is not None else next_at
            if at_s + 1e-9 < next_at:
                continue
            next_at = at_s + interval

            image = frame.to_image()
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
            yield at_s, buffer.getvalue(), image.width, image.height


def as_video_frame(
    *, sequence: int, at_s: float, payload: bytes, width: int, height: int, origin: dt.datetime
) -> VideoFrame:
    """Build the message the relay would have delivered for this frame."""
    captured_at = origin + dt.timedelta(seconds=at_s)
    frame = VideoFrame(
        session_id=SESSION_ID,
        epoch_id=EPOCH_ID,
        sequence=sequence,
        captured_at=captured_at,
        received_at=captured_at,
        relayed_at=captured_at,
        width=width,
        height=height,
        encoding="jpeg",
        pixel_format="rgb",
        payload_bytes=len(payload),
        sha256=payload_digest(payload),
    )
    return frame.attach_payload(payload)


class ReportingSink:
    """Prints every confirmed candidate, and optionally forwards it to Memory."""

    def __init__(self, emitter: MemoryEmitter | None) -> None:
        self._emitter = emitter
        self.confirmed: list[CandidateEvent] = []

    async def __call__(
        self,
        candidate: CandidateEvent,
        result: VerifierResult,
        frames: Sequence[BufferedFrame],
    ) -> None:
        self.confirmed.append(candidate)
        window_s = (
            candidate.window.window_ended_at - candidate.window.window_started_at
        ).total_seconds()
        depth = (
            f" depth={candidate.object_candidate.depth_m:.2f}m"
            if candidate.object_candidate.depth_m is not None
            else ""
        )
        # A `vanished` candidate is a question; the resolution is the answer,
        # and printing only the question hides what actually happened.
        resolved = f" -> {result.resolved_action}" if result.resolved_action else ""
        print(
            f"  {result.outcome.upper():<10} {candidate.action}{resolved:<14} "
            f"{candidate.label!r} track={candidate.track_id} "
            f"conf={candidate.object_candidate.confidence:.2f}"
            f"{depth} window={window_s:.1f}s over {len(frames)} frames"
        )
        print(f"             why: {result.reason_code}")
        if result.description:
            print(f"             where: {result.description}")
        if self._emitter is not None:
            await self._emitter.emit(candidate, result, frames)
            print("            -> recorded to Memory")


async def replay(path: Path, settings: Settings, *, emit: bool) -> int:
    detector, detector_ref = build_detector(settings)
    await detector.initialize()
    depth_estimator, depth_model_ref = build_depth_estimator(settings)
    if depth_estimator is not None:
        await depth_estimator.initialize()

    stability_config = build_stability_config(settings)
    # Whatever VMA_VERIFIER_KIND selects -- rules, or the VLM. A replay has no
    # live relay to starve, so a verifier that takes seconds is fine here even
    # though it would not be on a stream.
    verifier, _ = await build_verifier(settings)
    memory_client = MemoryClient(
        settings.memory_base_url,
        token=settings.memory_token,
        timeout=settings.memory_request_timeout_s,
    )
    emitter = MemoryEmitter(memory_client, clip_fps=settings.resolved_clip_fps) if emit else None
    sink = ReportingSink(emitter)

    pipeline = Pipeline(
        detector=detector,
        detector_ref=detector_ref,
        tracker=GreedyIoUTracker(max_age_frames=stability_config.reacquire_within_frames),
        tracker_ref=_TRACKER_REF,
        pose_source=ImageMotionPose(),
        track_registry=TrackRegistry(stability_config),
        evidence_ring=EvidenceRing(dt.timedelta(seconds=settings.evidence_ring_seconds)),
        verifier=verifier,
        detection_labels=settings.detection_labels,
        state_machine_version=STATE_MACHINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        on_confirmed=sink,
        source_fps=settings.source_fps,
        verification_queue_depth=settings.verification_queue_depth,
        verification_concurrency=settings.verification_concurrency,
        vanish_lookback_s=settings.vanish_lookback_s,
        depth_estimator=depth_estimator,
        depth_model_ref=depth_model_ref,
    )

    # A fixed origin rather than "now": a replay is deterministic, and two
    # runs of the same clip should produce identical timestamps.
    origin = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

    print(f"\n{path.name}")
    print(
        f"  detector={settings.detector_kind} verifier={settings.verifier_kind} "
        f"labels={list(settings.detection_labels) or '(open vocabulary)'} "
        f"sampling at {settings.source_fps}fps"
    )
    print(
        f"  thresholds: dwell={stability_config.dwell_frames}f "
        f"passive={stability_config.passive_confirmation_frames}f "
        f"reacquire={stability_config.reacquire_within_frames}f"
    )

    await pipeline.epoch_started(session_id=SESSION_ID, device_id="replay", epoch_id=EPOCH_ID)
    for sequence, (at_s, payload, width, height) in enumerate(
        decode_sampled(path, sample_fps=settings.source_fps)
    ):
        await pipeline.video_frame(
            session_id=SESSION_ID,
            device_id="replay",
            epoch_id=EPOCH_ID,
            frame=as_video_frame(
                sequence=sequence,
                at_s=at_s,
                payload=payload,
                width=width,
                height=height,
                origin=origin,
            ),
        )

    # The clip runs out long before the last verification answers -- a VLM call
    # takes seconds and the frame loop no longer waits for one. Without this,
    # a replay would print its summary while the interesting candidate was
    # still being thought about, and report it as never confirmed.
    await pipeline.aclose()

    await detector.aclose()
    if depth_estimator is not None:
        await depth_estimator.aclose()
    with contextlib.suppress(Exception):
        memory_client.close()

    _print_summary(pipeline, sink)
    return 0


def _print_summary(pipeline: Pipeline, sink: ReportingSink) -> None:
    metrics = pipeline.metrics
    observed = pipeline.observed_fps
    measured = f" observed_fps={observed:.1f}" if observed is not None else ""
    print(f"  frames={metrics.frames_processed}{measured}")
    print(
        f"  sightings_not_promoted={metrics.sightings_not_promoted} "
        f"proposed={metrics.candidates_proposed} "
        f"confirmed={metrics.candidates_confirmed} "
        f"rejected={metrics.candidates_rejected} "
        f"unverified={metrics.candidates_unverified}"
    )
    if pipeline.verifications_dropped or pipeline.verifications_failed:
        # A replay decodes far faster than real time, so it can outrun a
        # verifier in a way a live 8fps stream would not. Never let that be
        # silent: a dropped candidate looks exactly like one that was never
        # proposed, which would read as a detection failure that never happened.
        print(
            f"  !! dropped={pipeline.verifications_dropped} "
            f"failed={pipeline.verifications_failed} -- raise "
            f"VMA_VERIFICATION_QUEUE_DEPTH; these candidates were never verified"
        )
    actions = [candidate.action for candidate in sink.confirmed]
    print(f"  actions: {actions if actions else '(none)'}")
    for event in pipeline.recent_events:
        if event.outcome != "confirmed":
            print(f"  {event.outcome}: {event.action} -- {event.reason_code}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="replay_clip.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("clips", type=Path, nargs="+", help="video files to replay")
    parser.add_argument(
        "--emit",
        action="store_true",
        help="actually record confirmed candidates to the Memory Service",
    )
    parser.add_argument("--verbose", action="store_true", help="show pipeline logs")
    args = parser.parse_args(argv)

    configure_logging(level="DEBUG" if args.verbose else "WARNING", service="replay", version="dev")
    logging.getLogger("vision_worker.main").setLevel(logging.INFO)

    settings = Settings()
    for clip in args.clips:
        if not clip.exists():
            print(f"no such file: {clip}", file=sys.stderr)
            return 2
        asyncio.run(replay(clip, settings, emit=args.emit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
