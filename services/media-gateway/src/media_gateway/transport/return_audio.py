"""Publish synthesized speech back to the device.

The gateway joins each room as a participant and publishes one audio track, so
the assistant can be heard on the glasses. The Speech Service feeds PCM in; the
gateway owns the track and the pacing.

Backpressure comes from the SDK: `capture_frame` awaits until the source's
queue has room, so a producer that outruns real time is slowed rather than
buffering without limit.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from livekit import rtc

logger = logging.getLogger(__name__)

#: Audio is fed in whole frames of this length.
FRAME_MS = 20
SAMPLE_WIDTH = 2  # int16


class ReturnAudio:
    """One outbound audio track for one room."""

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        queue_size_ms: int,
        track_name: str,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.track_name = track_name
        self._queue_size_ms = queue_size_ms
        self._source: rtc.AudioSource | None = None
        self._phase = 0
        self.frames_sent = 0

    @property
    def samples_per_frame(self) -> int:
        return self.sample_rate * FRAME_MS // 1000

    @property
    def published(self) -> bool:
        return self._source is not None

    async def publish_to(self, room: rtc.Room) -> None:
        """Create the track and publish it into `room`."""
        source = rtc.AudioSource(self.sample_rate, self.channels, queue_size_ms=self._queue_size_ms)
        track = rtc.LocalAudioTrack.create_audio_track(self.track_name, source)
        # LiveKit has no dedicated "assistant" source, so this is a microphone
        # as far as the protocol is concerned; the track name carries meaning.
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self._source = source
        logger.info("publishing return audio", extra={"track_name": self.track_name})

    async def feed(self, pcm: bytes) -> int:
        """Send interleaved int16 PCM, in whole frames. Returns frames sent.

        A trailing partial frame is padded with silence rather than dropped, so
        the end of an utterance is not clipped.
        """
        if self._source is None:
            raise RuntimeError("return audio has not been published")

        frame_bytes = self.samples_per_frame * self.channels * SAMPLE_WIDTH
        sent = 0
        for start in range(0, len(pcm), frame_bytes):
            chunk = pcm[start : start + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + bytes(frame_bytes - len(chunk))
            await self._source.capture_frame(
                rtc.AudioFrame(
                    data=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels,
                    samples_per_channel=self.samples_per_frame,
                )
            )
            sent += 1
        self.frames_sent += sent
        return sent

    async def play_tone(self, *, hz: float, seconds: float, amplitude: float = 0.18) -> int:
        """Play a continuous tone. A stand-in until the Speech Service exists.

        Phase carries across calls so consecutive tones do not click.
        """
        frames = max(1, int(seconds * 1000 / FRAME_MS))
        sent = 0
        for _ in range(frames):
            start = self._phase
            time = np.arange(start, start + self.samples_per_frame, dtype=np.float64)
            wave = np.sin(2.0 * math.pi * hz * time / self.sample_rate)
            samples = (wave * amplitude * 32767.0).astype("<i2")
            if self.channels > 1:  # pragma: no cover - mono in practice
                samples = np.repeat(samples, self.channels)
            sent += await self.feed(samples.tobytes())
            self._phase = (start + self.samples_per_frame) % self.sample_rate
        return sent

    async def aclose(self) -> None:
        if self._source is not None:
            await self._source.aclose()
            self._source = None


def silence(sample_rate: int, channels: int, ms: int) -> bytes:
    """Silent PCM, for padding or for testing."""
    return bytes(sample_rate * ms // 1000 * channels * SAMPLE_WIDTH)


__all__ = ["FRAME_MS", "SAMPLE_WIDTH", "ReturnAudio", "silence"]
