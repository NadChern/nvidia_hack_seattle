"""Where a virtual glasses publisher gets its media.

Two sources: a deterministic synthetic pattern, and a prerecorded file. Both
yield presentation-timed frames so the publisher can pace them the way a real
camera would.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 1
AUDIO_FRAME_MS = 20


class PublisherDependencyError(RuntimeError):
    """An optional dependency needed for this source is missing."""


@dataclass(frozen=True, slots=True)
class VideoOut:
    """One frame, tightly packed RGBA, with its presentation time."""

    rgba: bytes
    width: int
    height: int
    at_s: float


@dataclass(frozen=True, slots=True)
class AudioOut:
    """One audio frame: interleaved little-endian int16."""

    pcm: bytes
    samples: int
    at_s: float


Media = VideoOut | AudioOut


def synthetic(
    *,
    width: int,
    height: int,
    fps: float,
    seconds: float,
    tone_hz: float = 440.0,
) -> Iterator[Media]:
    """A deterministic moving pattern with a matching tone.

    Content changes every frame so a stalled pipeline is obvious when watching
    the relay, and the tone is phase-continuous so audio gaps are audible.
    """
    frame_count = max(1, int(seconds * fps))
    audio_per_frame = max(1, int((1000.0 / fps) / AUDIO_FRAME_MS))
    samples_per_frame = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
    offset = 0

    columns = np.arange(width, dtype=np.uint16)
    rows = np.arange(height, dtype=np.uint16)[:, None]

    for index in range(frame_count):
        at_s = index / fps
        image = np.empty((height, width, 4), dtype=np.uint8)
        image[:, :, 0] = ((columns + index * 7) % 256).astype(np.uint8)
        image[:, :, 1] = (rows % 256).astype(np.uint8)
        image[:, :, 2] = np.uint8((index * 17) % 256)
        image[:, :, 3] = 255
        yield VideoOut(rgba=image.tobytes(), width=width, height=height, at_s=at_s)

        for _ in range(audio_per_frame):
            time = np.arange(offset, offset + samples_per_frame, dtype=np.float64)
            wave = np.sin(2.0 * math.pi * tone_hz * time / AUDIO_SAMPLE_RATE)
            pcm = (wave * 0.18 * 32767.0).astype("<i2").tobytes()
            yield AudioOut(
                pcm=pcm,
                samples=samples_per_frame,
                at_s=at_s + offset / AUDIO_SAMPLE_RATE,
            )
            offset += samples_per_frame


def from_file(path: Path, *, width: int | None, height: int | None) -> Iterator[Media]:
    """Decode a media file, resampling audio to the relay's format.

    PyAV is an optional dependency: it bundles FFmpeg, but the publisher is a
    test harness and does not ship in the runtime image.
    """
    try:
        import av
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by extras
        raise PublisherDependencyError(
            "reading a media file needs PyAV; install it with: uv sync --group publisher"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"no such media file: {path}")

    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE)
        for frame in container.decode(video=0, audio=0):
            if isinstance(frame, av.VideoFrame):
                target_w = width or frame.width
                target_h = height or frame.height
                converted = frame.reformat(width=target_w, height=target_h, format="rgba")
                yield VideoOut(
                    rgba=bytes(converted.to_ndarray().tobytes()),
                    width=target_w,
                    height=target_h,
                    at_s=float(frame.time or 0.0),
                )
            elif isinstance(frame, av.AudioFrame):
                for resampled in resampler.resample(frame):
                    array = resampled.to_ndarray()
                    yield AudioOut(
                        pcm=array.astype("<i2").tobytes(),
                        samples=int(array.shape[-1]),
                        at_s=float(frame.time or 0.0),
                    )


__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_FRAME_MS",
    "AUDIO_SAMPLE_RATE",
    "AudioOut",
    "Media",
    "PublisherDependencyError",
    "VideoOut",
    "from_file",
    "synthetic",
]
