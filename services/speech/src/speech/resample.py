"""Resamples raw PCM to whatever rate the STT model expects.

Local resampling is a deliberate choice, not a default nobody thought about:
the gateway's audio relay defaults to 48 kHz (`config.py`'s
`expected_audio_sample_rate`), and Parakeet checkpoints commonly expect
16 kHz mono input -- neither number is going away on its own. Whether the
gateway's *own* configured rate should ever change is a separate, open
question being raised with Alex; resampling inside this service is the
rules-safe default in the meantime and does not require his answer to move
forward (`role-prompts/Speech.md`, hard rule 4 -- never touch shared
contracts, and the gateway's config is Alex's, not this service's).

Uses `soxr` (bindings to libsoxr) rather than naive decimation or sample
duplication, which would alias -- audibly and audio-model-confusingly
distort the signal -- for any ratio that is not a small integer. `soxr` ships
prebuilt wheels for macOS arm64 and Linux arm64/x86_64, installs with no
compilation, and needs no GPU, so it holds the laptop-first rule cleanly.
"""

from __future__ import annotations

import numpy as np
import soxr  # pyright: ignore[reportMissingTypeStubs]  # soxr ships no type stubs
from visual_memory_media_contract.protocol import SampleFormat

from speech.ingest import AudioSegment

_BYTES_PER_SAMPLE = 2  # s16le


def resample_pcm(
    pcm: bytes,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    channels: int,
    sample_format: SampleFormat = "s16le",
) -> bytes:
    """Resample raw interleaved PCM from `source_sample_rate` to `target_sample_rate`.

    Decodes int16 -> float64, resamples with `soxr`, then re-encodes to
    int16. The output length is normalized to exactly
    `round(input_samples * target_sample_rate / source_sample_rate)` --
    `soxr`'s one-shot resampler does not itself guarantee that exact count
    (filter delay can leave it a few samples short or long), and letting that
    drift propagate would make segment durations nondeterministic downstream
    for no benefit.
    """
    if sample_format != "s16le":
        raise ValueError(f"resample_pcm only supports s16le, got {sample_format!r}")
    if source_sample_rate == target_sample_rate:
        return pcm

    samples_int16 = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
    input_samples = samples_int16.shape[0]
    samples_float = samples_int16.astype(np.float64) / 32768.0

    resampled_float = soxr.resample(samples_float, source_sample_rate, target_sample_rate)

    expected_samples = round(input_samples * target_sample_rate / source_sample_rate)
    actual_samples = resampled_float.shape[0]
    if actual_samples < expected_samples:
        pad = np.zeros((expected_samples - actual_samples, channels), dtype=np.float64)
        resampled_float = np.concatenate([resampled_float, pad], axis=0)
    elif actual_samples > expected_samples:
        resampled_float = resampled_float[:expected_samples]

    resampled_clipped = np.clip(resampled_float, -1.0, 1.0)
    resampled_int16 = (resampled_clipped * 32767.0).round().astype("<i2")

    return resampled_int16.tobytes()


def resample_segment(segment: AudioSegment, *, target_sample_rate: int) -> AudioSegment:
    """Return a copy of `segment` with its PCM resampled to `target_sample_rate`.

    Only `sample_rate`, `samples`, and `pcm` change. `pts_samples_start` is
    left untouched on purpose: it is a pointer back to where this audio came
    from in the source epoch's own sample numbering, not a description of
    this payload's own encoding, so resampling must not touch it.
    """
    resampled_pcm = resample_pcm(
        segment.pcm,
        source_sample_rate=segment.sample_rate,
        target_sample_rate=target_sample_rate,
        channels=segment.channels,
        sample_format=segment.sample_format,
    )
    resampled_samples = len(resampled_pcm) // _BYTES_PER_SAMPLE // segment.channels
    return segment.model_copy(
        update={
            "sample_rate": target_sample_rate,
            "samples": resampled_samples,
            "pcm": resampled_pcm,
        }
    )


__all__ = ["resample_pcm", "resample_segment"]
