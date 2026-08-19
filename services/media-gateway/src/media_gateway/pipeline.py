"""Connects a media source to the relay.

Ingest is synchronous and non-blocking: a frame is dimension-checked and put
into a one-item slot, nothing more. A separate paced task per video epoch takes
the newest frame, encodes it once, and fans it out. That separation is what
keeps slow inference downstream from applying backpressure to media ingest.

Audio takes the opposite path: coalesced but never dropped, and published
immediately so transcription sees a continuous stream.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Callable
from typing import Protocol

from visual_memory_media_contract.framing import encode_message, payload_digest
from visual_memory_media_contract.protocol import (
    ActiveEpoch,
    AudioChunk,
    EpochEnded,
    EpochEndReason,
    EpochStarted,
    Keepalive,
    LifecycleEnvelope,
    LifecycleSignal,
    RelayMessage,
    SessionEnded,
    SessionEndReason,
    SessionStarted,
    StreamHello,
    StreamKind,
    VideoEncoding,
    VideoFrame,
)

from media_gateway import __version__
from media_gateway.config import Settings
from media_gateway.domain import lifecycle
from media_gateway.domain.epoch import EpochRegistry, MediaEpoch
from media_gateway.domain.metrics import MetricsRegistry
from media_gateway.domain.sampling import LatestSlot, Pacer
from media_gateway.relay.codec import encode_video
from media_gateway.relay.hub import RelayHub
from media_gateway.transport.source import RawAudioFrame, RawVideoFrame, utcnow

logger = logging.getLogger(__name__)


class LifecycleEmitter(Protocol):
    """What the pipeline needs from a lifecycle sink.

    A Protocol rather than the concrete class so the pipeline -- which is
    domain-adjacent and heavily unit-tested -- does not import httpx or the
    transport layer to end an epoch.
    """

    def emit(self, envelope: LifecycleEnvelope) -> None: ...


class _AudioAccumulator:
    """Coalesces short frames into one relay chunk.

    LiveKit delivers 20 ms frames; relaying each one would be 50 messages a
    second per subscriber for no benefit.
    """

    def __init__(self, *, target_samples: int) -> None:
        self.target_samples = target_samples
        self._chunks: list[bytes] = []
        self._samples = 0
        self._first_captured_at: dt.datetime | None = None

    def add(self, frame: RawAudioFrame) -> tuple[bytes, int, dt.datetime] | None:
        """Add a frame, returning a chunk once enough audio has accrued."""
        if self._first_captured_at is None:
            self._first_captured_at = frame.captured_at
        self._chunks.append(frame.pcm)
        self._samples += frame.samples
        if self._samples < self.target_samples:
            return None
        return self.flush()

    def flush(self) -> tuple[bytes, int, dt.datetime] | None:
        """Emit whatever has accrued, so an epoch end loses no audio."""
        if not self._chunks or self._first_captured_at is None:
            return None
        payload = b"".join(self._chunks)
        samples = self._samples
        captured_at = self._first_captured_at
        self._chunks = []
        self._samples = 0
        self._first_captured_at = None
        return payload, samples, captured_at


class MediaPipeline:
    """Implements `MediaSink`, turning raw media into relay messages."""

    def __init__(
        self,
        *,
        settings: Settings,
        hub: RelayHub,
        epochs: EpochRegistry,
        metrics: MetricsRegistry,
        pacer_factory: Callable[[float], Pacer] | None = None,
        lifecycle_sink: LifecycleEmitter | None = None,
    ) -> None:
        self._settings = settings
        self._hub = hub
        self._epochs = epochs
        self._metrics = metrics
        # Optional: the gateway is fully useful with no Memory Service, so a
        # missing sink is a no-op rather than a branch at every call site.
        self._lifecycle_sink = lifecycle_sink
        # Injectable so tests drive sampling deterministically instead of
        # waiting on real time and hoping the scheduler cooperates.
        self._pacer_factory = pacer_factory or Pacer
        self._slots: dict[str, LatestSlot[RawVideoFrame]] = {}
        self._samplers: dict[str, asyncio.Task[None]] = {}
        self._audio: dict[str, _AudioAccumulator] = {}
        self._device_ids: dict[str, str] = {}
        self._sessions_started: dict[str, dt.datetime] = {}

    # --- Sink ------------------------------------------------------------

    def session_started(self, *, session_id: str, device_id: str) -> None:
        started_at = utcnow()
        self._device_ids[session_id] = device_id
        self._sessions_started[session_id] = started_at
        self._metrics.sessions_created += 1
        self._broadcast(
            SessionStarted(session_id=session_id, device_id=device_id, started_at=started_at)
        )

    def epoch_started(
        self,
        *,
        session_id: str,
        stream_kind: StreamKind,
        track_sid: str,
        participant_identity: str,
    ) -> None:
        started_at = utcnow()
        epoch, displaced = self._epochs.begin(
            session_id=session_id,
            stream_kind=stream_kind,
            track_sid=track_sid,
            participant_identity=participant_identity,
            at=started_at,
        )
        if displaced is not None:
            # A track vanished without an unsubscribe. Close it out so a
            # consumer never sees two epochs claiming the same stream.
            self._finish_epoch(displaced, displaced.end_reason or "track_unsubscribed")

        self._metrics.epochs_started += 1
        self._publish(self._epoch_started_message(epoch), stream_kind)

        if stream_kind == "video":
            slot: LatestSlot[RawVideoFrame] = LatestSlot()
            self._slots[epoch.epoch_id] = slot
            self._samplers[epoch.epoch_id] = asyncio.create_task(
                self._sample(epoch, slot), name=f"sample-{epoch.epoch_id}"
            )
        else:
            self._audio[epoch.epoch_id] = _AudioAccumulator(
                target_samples=(
                    self._settings.audio_sample_rate * self._settings.audio_chunk_ms // 1000
                )
            )

    def video_frame(self, *, session_id: str, frame: RawVideoFrame) -> None:
        epoch = self._epochs.active_for(session_id, "video")
        if epoch is None or epoch.guard is None:
            return

        epoch.received += 1
        metrics = self._metrics.video
        metrics.received += 1

        if not epoch.guard.admit(frame.width, frame.height):
            metrics.rejected_dimensions += 1
            return
        metrics.admitted += 1

        slot = self._slots.get(epoch.epoch_id)
        if slot is not None:
            slot.offer(frame)

    def audio_frame(self, *, session_id: str, frame: RawAudioFrame) -> None:
        epoch = self._epochs.active_for(session_id, "audio")
        if epoch is None:
            return

        epoch.received += 1
        self._metrics.audio.received += 1
        self._metrics.audio.admitted += 1

        accumulator = self._audio.get(epoch.epoch_id)
        if accumulator is None:  # pragma: no cover - created with the epoch
            return
        chunk = accumulator.add(frame)
        if chunk is not None:
            self._emit_audio(epoch, frame, chunk)

    def epoch_ended(
        self, *, session_id: str, stream_kind: StreamKind, reason: EpochEndReason
    ) -> None:
        epoch = self._epochs.end_active(
            session_id=session_id, stream_kind=stream_kind, reason=reason
        )
        if epoch is not None:
            self._finish_epoch(epoch, reason)

    def session_ended(self, *, session_id: str, reason: SessionEndReason) -> None:
        for epoch in self._epochs.end_session(session_id, reason="session_ended"):
            self._finish_epoch(epoch, "session_ended")
        self._metrics.sessions_ended += 1
        ended_at = utcnow()
        self._emit_lifecycle(
            lifecycle.session_ended(
                session_id=session_id,
                device_id=self._device_ids.get(session_id, "unknown"),
                reason=reason,
                occurred_at=ended_at,
            )
        )
        self._broadcast(SessionEnded(session_id=session_id, ended_at=ended_at, reason=reason))
        self._epochs.forget_session(session_id)
        self._device_ids.pop(session_id, None)
        self._sessions_started.pop(session_id, None)

    # --- Relay -----------------------------------------------------------

    def build_hello(self, *, stream_kind: StreamKind, encoding: VideoEncoding | None) -> bytes:
        """The first message a subscriber receives.

        Re-announcing active epochs is what lets a consumer that connected mid
        epoch, or reconnected, reset rather than resume against stale state.
        """
        active = [
            ActiveEpoch(
                session_id=epoch.session_id,
                epoch_id=epoch.epoch_id,
                stream_kind=epoch.stream_kind,
                started_at=epoch.started_at,
            )
            for epoch in self._epochs.active()
            if epoch.stream_kind == stream_kind
        ]
        return encode_message(
            StreamHello(
                gateway_version=__version__,
                stream_kind=stream_kind,
                encoding=encoding,
                active_sessions=sorted({epoch.session_id for epoch in active}),
                active_epochs=active,
            )
        )

    def replay_epochs_for(self, stream_kind: StreamKind) -> list[bytes]:
        """Synthetic `epoch_started` messages for epochs already running."""
        return [
            encode_message(self._epoch_started_message(epoch))
            for epoch in self._epochs.active()
            if epoch.stream_kind == stream_kind
        ]

    def keepalive(self, stream_kind: StreamKind) -> bytes:
        del stream_kind
        return encode_message(Keepalive(sent_at=utcnow()))

    async def stop(self) -> None:
        """Cancel sampler tasks and flush any buffered audio."""
        for task in list(self._samplers.values()):
            task.cancel()
        for task in list(self._samplers.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                logger.exception("sampler task failed during shutdown")
        self._samplers.clear()
        self._slots.clear()
        self._audio.clear()

    # --- Internals -------------------------------------------------------

    def _epoch_started_message(self, epoch: MediaEpoch) -> EpochStarted:
        is_video = epoch.stream_kind == "video"
        size = epoch.guard.latched if epoch.guard else None
        return EpochStarted(
            session_id=epoch.session_id,
            epoch_id=epoch.epoch_id,
            stream_kind=epoch.stream_kind,
            track_sid=epoch.track_sid,
            participant_identity=epoch.participant_identity,
            started_at=epoch.started_at,
            width=size[0] if size else None,
            height=size[1] if size else None,
            encoding=self._settings.video_encoding if is_video else None,
            pixel_format=("rgba" if self._settings.video_encoding == "rgba_raw" else "rgb")
            if is_video
            else None,
            sample_rate=None if is_video else self._settings.audio_sample_rate,
            channels=None if is_video else self._settings.audio_channels,
        )

    def _emit_lifecycle(
        self, envelope: LifecycleEnvelope, stream_kind: StreamKind | None = None
    ) -> None:
        """Relay the signal in band and hand a copy to the Memory Service.

        Both, not either. The in-band copy is ordered against the media, so a
        consumer sees it in the right place in the stream; the posted copy is
        what actually changes trusted state. A consumer that only watches the
        relay and a Memory that only sees the POST must agree, and they do
        because it is the same envelope.
        """
        self._metrics.lifecycle_signals_emitted += 1
        message = LifecycleSignal(envelope=envelope)
        if stream_kind is None:
            self._broadcast(message)
        else:
            self._publish(message, stream_kind)
        if self._lifecycle_sink is not None:
            # Never awaits: a slow Memory must not stall the media path.
            self._lifecycle_sink.emit(envelope)

    def _finish_epoch(self, epoch: MediaEpoch, reason: EpochEndReason) -> None:
        task = self._samplers.pop(epoch.epoch_id, None)
        if task is not None:
            task.cancel()
        self._slots.pop(epoch.epoch_id, None)

        accumulator = self._audio.pop(epoch.epoch_id, None)
        if accumulator is not None:
            remainder = accumulator.flush()
            if remainder is not None:
                self._emit_audio(epoch, None, remainder)

        self._metrics.epochs_ended += 1
        ended_at = epoch.ended_at or utcnow()
        # The signal explains why the epoch is ending, so it precedes the
        # terminal message. `epoch_ended` stays the last word on an epoch and
        # `session_ended` the last word on a session -- a consumer that stops
        # reading at either must not have missed anything.
        self._emit_lifecycle(
            lifecycle.track_lost(
                session_id=epoch.session_id,
                device_id=self._device_ids.get(epoch.session_id, "unknown"),
                media_epoch_id=epoch.epoch_id,
                reason=reason,
                occurred_at=ended_at,
            ),
            epoch.stream_kind,
        )
        self._publish(
            EpochEnded(
                session_id=epoch.session_id,
                epoch_id=epoch.epoch_id,
                ended_at=ended_at,
                reason=reason,
            ),
            epoch.stream_kind,
        )

    async def _sample(self, epoch: MediaEpoch, slot: LatestSlot[RawVideoFrame]) -> None:
        """Take the newest frame at the configured rate and relay it."""
        pacer = self._pacer_factory(1.0 / self._settings.sample_fps)
        already_dropped = 0
        while True:
            await pacer.wait()
            frame = slot.take()
            if frame is None:
                continue

            dropped = slot.dropped - already_dropped
            already_dropped = slot.dropped
            try:
                self._relay_video(epoch, frame, dropped)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "failed to relay a frame",
                    extra={"session_id": epoch.session_id, "media_epoch_id": epoch.epoch_id},
                )

    def _relay_video(self, epoch: MediaEpoch, frame: RawVideoFrame, dropped: int) -> None:
        encodings = self._hub.required_video_encodings()
        if not encodings:
            return

        relayed_at = utcnow()
        payloads: dict[VideoEncoding, bytes] = {}
        sequence = epoch.take_sequence()

        for encoding in encodings:
            payload, pixel_format = encode_video(
                frame.rgba,
                width=frame.width,
                height=frame.height,
                encoding=encoding,
                quality=self._settings.jpeg_quality,
                subsampling=self._settings.jpeg_subsampling,
            )
            message = VideoFrame(
                session_id=epoch.session_id,
                epoch_id=epoch.epoch_id,
                sequence=sequence,
                captured_at=frame.captured_at,
                received_at=frame.received_at or frame.captured_at,
                relayed_at=relayed_at,
                width=frame.width,
                height=frame.height,
                encoding=encoding,
                pixel_format=pixel_format,
                payload_bytes=len(payload),
                sha256=payload_digest(payload),
                dropped_since_previous=dropped,
            )
            payloads[encoding] = encode_message(message, payload)

        self._hub.publish_video(payloads)
        epoch.relayed += 1
        metrics = self._metrics.video
        metrics.relayed += 1
        metrics.dropped_before_sampling += dropped
        metrics.relay_latency.observe((relayed_at - frame.captured_at).total_seconds())

    def _emit_audio(
        self,
        epoch: MediaEpoch,
        frame: RawAudioFrame | None,
        chunk: tuple[bytes, int, dt.datetime],
    ) -> None:
        payload, samples, captured_at = chunk

        if not self._hub.subscribers("audio"):
            # Nobody is listening. Advance the timeline anyway -- the audio did
            # happen, and pts must stay continuous for whoever connects next --
            # but do not build a message or count it as relayed. Video already
            # skips work with no subscribers; counting audio as relayed with
            # none would make the two disagree on a dashboard.
            epoch.pts_samples += samples
            return

        message = AudioChunk(
            session_id=epoch.session_id,
            epoch_id=epoch.epoch_id,
            sequence=epoch.take_sequence(),
            pts_samples=epoch.pts_samples,
            samples=samples,
            sample_rate=frame.sample_rate if frame else self._settings.audio_sample_rate,
            channels=frame.channels if frame else self._settings.audio_channels,
            sample_format="s16le",
            first_sample_captured_at=captured_at,
            payload_bytes=len(payload),
        )
        epoch.pts_samples += samples
        epoch.relayed += 1

        self._hub.publish_audio(encode_message(message, payload))
        metrics = self._metrics.audio
        metrics.relayed += 1
        metrics.relay_latency.observe((utcnow() - captured_at).total_seconds())

    def _publish(self, message: RelayMessage, stream_kind: StreamKind) -> None:
        self._hub.publish_control(encode_message(message), stream_kind)

    def _broadcast(self, message: RelayMessage) -> None:
        self._publish(message, "video")
        self._publish(message, "audio")


__all__ = ["MediaPipeline"]
