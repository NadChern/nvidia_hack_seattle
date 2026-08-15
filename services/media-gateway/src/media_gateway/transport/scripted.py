"""A media source that needs no LiveKit.

Selected with `VMA_MEDIA_SOURCE=scripted`. It drives the sink through the cases
that are easy to get wrong -- a rejoin that changes the track SID, transition
frames of the wrong size, a stalled consumer -- so the relay, epoch, metrics,
and WebSocket paths are all exercised in CI with no server and no hardware.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from media_gateway.transport.source import (
    MediaSink,
    RawAudioFrame,
    RawVideoFrame,
    utcnow,
)

DEVICE_ID = "scripted-glasses"


@dataclass
class ScriptedPlan:
    """What the scripted source produces."""

    session_id: str = "sess_scripted"
    device_id: str = DEVICE_ID
    width: int = 320
    height: int = 180
    #: Rejoins. Each opens a new pair of track SIDs, so each is a new epoch.
    epochs: int = 2
    frames_per_epoch: int = 6
    #: Frames of the wrong size injected per epoch, mimicking the transient
    #: 8x8 frames the S01 spike saw during simulcast adaptation.
    transition_frames: int = 2
    audio_frames_per_epoch: int = 25
    audio_frame_ms: int = 20
    sample_rate: int = 48_000
    channels: int = 1
    #: Wall-clock delay between produced frames. Zero runs as fast as possible,
    #: which is what tests want; a live demo sets something realistic.
    frame_interval_s: float = 0.0
    #: Repeat forever, for a gateway left running by hand.
    loop: bool = False
    started_at: dt.datetime = field(default_factory=utcnow)


def _pattern(plan: ScriptedPlan, sequence: int) -> bytes:
    """A deterministic RGBA frame that visibly changes between frames."""
    image = np.empty((plan.height, plan.width, 4), dtype=np.uint8)
    image[:, :, 0] = (np.arange(plan.width, dtype=np.uint16) + sequence * 7).astype(np.uint8)
    image[:, :, 1] = np.arange(plan.height, dtype=np.uint16)[:, None].astype(np.uint8)
    image[:, :, 2] = np.uint8((sequence * 17) % 256)
    image[:, :, 3] = 255
    return image.tobytes()


def _tone(plan: ScriptedPlan, offset_samples: int, frequency_hz: float) -> bytes:
    """A phase-continuous sine, so a gap in the audio is audible."""
    count = plan.sample_rate * plan.audio_frame_ms // 1000
    time = np.arange(offset_samples, offset_samples + count, dtype=np.float64)
    wave = np.sin(2.0 * math.pi * frequency_hz * time / plan.sample_rate)
    samples = (wave * 0.18 * 32767.0).astype("<i2")
    if plan.channels > 1:  # pragma: no cover - mono in practice
        samples = np.repeat(samples, plan.channels)
    return samples.tobytes()


class ScriptedMediaSource:
    """Replays a deterministic plan into a sink."""

    def __init__(self, plan: ScriptedPlan | None = None) -> None:
        self.plan = plan or ScriptedPlan()
        self._stop = asyncio.Event()

    async def aclose(self) -> None:
        self._stop.set()

    async def run(self, sink: MediaSink) -> None:
        plan = self.plan
        sink.session_started(session_id=plan.session_id, device_id=plan.device_id)
        try:
            cycle = 0
            while not self._stop.is_set():
                await self._run_epochs(sink, cycle)
                if not plan.loop or self._stop.is_set():
                    break
                cycle += 1
        finally:
            sink.session_ended(session_id=plan.session_id, reason="gateway_shutdown")

    async def _run_epochs(self, sink: MediaSink, cycle: int) -> None:
        plan = self.plan
        for index in range(plan.epochs):
            if self._stop.is_set():
                return
            suffix = f"{cycle}{index}"
            video_sid = f"TR_VCscripted{suffix}"
            audio_sid = f"TR_ACscripted{suffix}"

            # Same identity across every rejoin: the track SID is what changes,
            # which is exactly the case a consumer must reset on.
            sink.epoch_started(
                session_id=plan.session_id,
                stream_kind="video",
                track_sid=video_sid,
                participant_identity=plan.device_id,
            )
            sink.epoch_started(
                session_id=plan.session_id,
                stream_kind="audio",
                track_sid=audio_sid,
                participant_identity=plan.device_id,
            )

            await self._produce(sink, epoch_index=index)

            sink.epoch_ended(
                session_id=plan.session_id,
                stream_kind="video",
                reason="track_unsubscribed",
            )
            sink.epoch_ended(
                session_id=plan.session_id,
                stream_kind="audio",
                reason="track_unsubscribed",
            )

    async def _produce(self, sink: MediaSink, *, epoch_index: int) -> None:
        plan = self.plan
        offset_samples = 0
        frequency = 440.0 + epoch_index * 30.0
        audio_per_video = max(1, plan.audio_frames_per_epoch // max(1, plan.frames_per_epoch))

        for sequence in range(plan.frames_per_epoch):
            if self._stop.is_set():
                return

            # A wrong-sized frame early in the epoch, as a real track produces
            # during simulcast adaptation. The guard must reject these.
            if sequence < plan.transition_frames:
                sink.video_frame(
                    session_id=plan.session_id,
                    frame=RawVideoFrame(
                        width=8,
                        height=8,
                        rgba=bytes(8 * 8 * 4),
                        captured_at=utcnow(),
                    ),
                )

            sink.video_frame(
                session_id=plan.session_id,
                frame=RawVideoFrame(
                    width=plan.width,
                    height=plan.height,
                    rgba=_pattern(plan, sequence),
                    captured_at=utcnow(),
                ),
            )

            for _ in range(audio_per_video):
                pcm = _tone(plan, offset_samples, frequency)
                count = plan.sample_rate * plan.audio_frame_ms // 1000
                sink.audio_frame(
                    session_id=plan.session_id,
                    frame=RawAudioFrame(
                        pcm=pcm,
                        samples=count,
                        sample_rate=plan.sample_rate,
                        channels=plan.channels,
                        captured_at=utcnow(),
                    ),
                )
                offset_samples += count

            if plan.frame_interval_s:
                await asyncio.sleep(plan.frame_interval_s)
            else:
                # Yield so the sampler task can run between frames.
                await asyncio.sleep(0)


__all__ = ["DEVICE_ID", "ScriptedMediaSource", "ScriptedPlan"]
