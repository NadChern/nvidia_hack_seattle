"""The virtual glasses publisher.

Joining a real room needs LiveKit, so that is integration territory. What is
testable here is the media generation and the CLI contract.
"""

from pathlib import Path

import numpy as np
import pytest

from media_gateway.publisher.cli import build_media, parse_args
from media_gateway.publisher.sources import (
    AUDIO_SAMPLE_RATE,
    AudioOut,
    PublisherDependencyError,
    VideoOut,
    from_file,
    synthetic,
)


def test_synthetic_yields_video_and_audio() -> None:
    items = list(synthetic(width=64, height=32, fps=10, seconds=1))

    videos = [i for i in items if isinstance(i, VideoOut)]
    audios = [i for i in items if isinstance(i, AudioOut)]

    assert len(videos) == 10
    assert audios


def test_synthetic_frames_are_tightly_packed_rgba() -> None:
    """The relay's encoder rejects a buffer that disagrees with its size."""
    frame = next(i for i in synthetic(width=64, height=32, fps=5, seconds=1))

    assert isinstance(frame, VideoOut)
    assert len(frame.rgba) == 64 * 32 * 4


def test_synthetic_content_changes_every_frame() -> None:
    """A frozen pipeline should be obvious, not look like valid stillness."""
    videos = [
        i for i in synthetic(width=32, height=16, fps=5, seconds=1) if isinstance(i, VideoOut)
    ]

    digests = {frame.rgba for frame in videos}

    assert len(digests) == len(videos)


def test_synthetic_audio_is_int16_at_the_relay_rate() -> None:
    audio = next(
        i for i in synthetic(width=32, height=16, fps=5, seconds=1) if isinstance(i, AudioOut)
    )

    samples = np.frombuffer(audio.pcm, dtype="<i2")

    assert len(samples) == audio.samples
    assert audio.samples == AUDIO_SAMPLE_RATE * 20 // 1000


def test_presentation_times_advance() -> None:
    videos = [
        i for i in synthetic(width=32, height=16, fps=10, seconds=1) if isinstance(i, VideoOut)
    ]

    times = [frame.at_s for frame in videos]

    assert times == sorted(times)
    assert times[1] == pytest.approx(0.1)


def test_a_missing_file_is_reported_clearly() -> None:
    with pytest.raises((FileNotFoundError, PublisherDependencyError)):
        list(from_file(Path("/nonexistent/clip.mp4"), width=None, height=None))


def test_a_source_must_be_chosen() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_the_two_sources_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--synthetic", "--file", "clip.mp4"])


def test_synthetic_defaults_to_full_resolution() -> None:
    """720p by default; the guard's 320x180 default is for the scripted source."""
    args = parse_args(["--synthetic"])

    assert (args.width, args.height) == (1280, 720)


def test_build_media_honours_the_requested_size() -> None:
    args = parse_args(["--synthetic", "--width", "96", "--height", "48", "--seconds", "1"])

    frame = next(i for i in build_media(args) if isinstance(i, VideoOut))

    assert (frame.width, frame.height) == (96, 48)
    assert len(frame.rgba) == 96 * 48 * 4
