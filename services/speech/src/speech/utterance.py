"""Deciding when the wearer stopped talking.

`ingest.py` produces *contiguous* audio -- a run with no `pts_samples` gap
inside it -- and that is the right definition for what it does. It is not a
useful definition of an utterance. A microphone keeps producing samples while
nobody is speaking, so a pause creates no gap, and a live session yields
nothing at all until the epoch ends. Measured exactly that way: the gateway
admitted 54,252 audio frames and relayed them, and no transcript existed
because no boundary had occurred.

An assistant cannot answer a question until it knows the question finished, so
"contiguous audio" has to be split into "somebody said something" somewhere.
This is that somewhere, kept out of `ingest.py` so its contract stays exactly
what its docstring says.

**Silero, not an amplitude threshold.** Energy alone cannot tell speech from a
fan, a door, or a hand brushing a desk, and a close-talking microphone in a
demo room hears all three. Measured on this project's own synthesized speech:
mean speech probability 0.775 with 77% of frames above 0.5, against 0.005 and
0% for digital silence.

`pysilero-vad` rather than `silero-vad` or `livekit-plugins-silero`: it vendors
the ONNX model and has **no runtime dependencies at all**, where `silero-vad`
requires torch and torchaudio -- which the Apple Silicon profile does not
install and CI has no reason to download -- and `livekit-plugins-silero`
requires the whole `livekit-agents` framework for one model. Wheels exist for
macOS arm64, Linux x86_64, and Linux aarch64, which is every machine this runs
on including the GN100.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

#: What Silero expects: 16 kHz mono, in 512-sample frames (32 ms).
VAD_SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2


class VoiceDetector(Protocol):
    """The model half: how likely is this 512-sample frame to be speech.

    Injectable so the silence/length state machine below can be tested against
    scripted probabilities. Without that seam every test of "when does an
    utterance end" would need audio a real model agrees is speech, which makes
    the tests slow, and -- worse -- makes a state-machine bug indistinguishable
    from the model disagreeing.
    """

    # Positional-only: the real detector names this parameter `audio`, and a
    # Protocol with a named parameter would only match a callable using the
    # same name.
    def __call__(self, frame: bytes, /) -> float: ...

    def reset(self) -> None: ...


class SpeechBoundary(Protocol):
    """Decides, chunk by chunk, whether an utterance just ended."""

    def feed(self, pcm: bytes, *, sample_rate: int) -> bool:
        """Consume audio; return True exactly once when speech has ended."""
        ...

    @property
    def leading_silence_seconds(self) -> float:
        """How much of the current utterance elapsed before anyone spoke.

        Read by `ingest.py` so it can drop that dead air -- but only once
        speech has actually been found. Trimming on the *expectation* of speech
        would mean a detector false negative silently destroyed the audio.
        """
        ...

    def consume_boundary(self) -> None:
        """Begin a fresh utterance, after the caller has acted on a boundary.

        Separate from `feed` returning True so `leading_silence_seconds` is
        still readable in between -- the caller needs it to know how much dead
        air to drop.
        """
        ...

    def reset(self) -> None:
        """Forget everything -- a new epoch is not a continuation."""
        ...


#: Frames between speech-probability summaries: ~8 s at 32 ms per frame.
_TELEMETRY_FRAMES = 250


class SileroBoundary:
    """Ends an utterance after a run of silence following speech.

    Deliberately requires speech *before* silence can end anything. Otherwise
    a session that opens with someone not talking would emit an endless series
    of empty utterances, each one a model call producing "". The emergency
    length ceiling likewise starts with first speech, not with idle microphone
    time before the wearer begins a turn.
    """

    def __init__(
        self,
        *,
        # Mirrors `Settings.stt_utterance_silence_ms`, which is what the only
        # production caller passes. Kept in step so the two cannot disagree.
        silence_ms: int = 1_000,
        max_seconds: float = 8.0,
        threshold: float = 0.5,
        detector: VoiceDetector | None = None,
        telemetry_frames: int = _TELEMETRY_FRAMES,
    ) -> None:
        if detector is None:
            from pysilero_vad import SileroVoiceActivityDetector

            self._vad: VoiceDetector = SileroVoiceActivityDetector()
        else:
            self._vad = detector
        self._threshold = threshold
        self._silence_frames_needed = max(
            1, round(silence_ms / 1000 * VAD_SAMPLE_RATE) // VAD_FRAME_SAMPLES
        )
        self._max_frames = max(1, round(max_seconds * VAD_SAMPLE_RATE) // VAD_FRAME_SAMPLES)
        self._buffer = bytearray()
        self._heard_speech = False
        self._silent_frames = 0
        # The emergency ceiling measures the utterance, not idle microphone
        # time. Counting before first speech makes a wearer who pauses longer
        # than ``max_seconds`` get cut off on their first spoken frame.
        self._frames_since_speech = 0
        #: Frames seen before the first speech of the current utterance. What
        #: makes trimming safe: it is only ever non-zero once speech was found,
        #: so a detector that hears nothing causes nothing to be discarded.
        self._frames_before_speech = 0
        #: Telemetry. Whether the detector agrees with the microphone is not
        #: something anyone can tell from the outside -- a wearer who cannot
        #: trigger the assistant sees the same symptom whether noise reads as
        #: speech or speech reads as noise, and those need opposite fixes.
        #: Statistics only; never audio. See `_observe`.
        self._telemetry_frames = max(0, telemetry_frames)
        self._seen = 0
        self._sum = 0.0
        self._max = 0.0
        self._above = 0

    def _observe(self, probability: float) -> None:
        """Summarize what the detector is actually seeing, periodically.

        Reads as follows, and the two failures need opposite fixes:

        - `above_threshold` near 1.0 while nobody is talking means the
          microphone's gain or noise suppression is presenting room noise as
          speech. No silence is ever found, so utterances run to `max_seconds`
          and are cut mid-word. Raise `stt_vad_threshold`.
        - `above_threshold` near 0.0 while somebody *is* talking means the
          opposite, and the threshold should come down.

        A wearer who cannot trigger the assistant sees the identical symptom in
        both cases, which is why this is measured rather than guessed at.
        Statistics only -- never audio, never a transcript. See docs/07.
        """
        if self._telemetry_frames == 0:
            return
        self._seen += 1
        self._sum += probability
        self._max = max(self._max, probability)
        if probability >= self._threshold:
            self._above += 1

        if self._seen < self._telemetry_frames:
            return
        logger.info(
            "voice detector probability summary",
            extra={
                "frames": self._seen,
                "seconds": round(self._seen * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE, 1),
                "mean_probability": round(self._sum / self._seen, 3),
                "max_probability": round(self._max, 3),
                "above_threshold": round(self._above / self._seen, 3),
                "threshold": self._threshold,
            },
        )
        self._seen = 0
        self._sum = 0.0
        self._max = 0.0
        self._above = 0

    @property
    def leading_silence_seconds(self) -> float:
        # Zero until speech has actually been found. Belt and braces: a
        # boundary cannot fire without speech, so `ingest.py` never reads this
        # otherwise -- but the value it would read is audio it is about to
        # discard, and "silence before the speech" is meaningless when there
        # is no speech.
        if not self._heard_speech:
            return 0.0
        return self._frames_before_speech * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE

    def reset(self) -> None:
        self._vad.reset()
        self._buffer.clear()
        self._heard_speech = False
        self._silent_frames = 0
        self._frames_since_speech = 0
        self._frames_before_speech = 0

    def feed(self, pcm: bytes, *, sample_rate: int) -> bool:
        # Resampled per chunk rather than through a streaming resampler. The
        # edge effects that introduces are a fraction of a millisecond at each
        # chunk join, which matters for audio a model will transcribe -- and
        # this copy is never transcribed, only measured. The audio that
        # reaches Parakeet is resampled once, from the whole segment.
        if sample_rate != VAD_SAMPLE_RATE:
            # Imported here, not at module scope: `resample.py` imports
            # `AudioSegment` from `ingest.py`, and `ingest.py` imports this
            # module, so a top-level import closes the cycle and breaks every
            # import of the package.
            from speech.resample import resample_pcm

            pcm = resample_pcm(
                pcm,
                source_sample_rate=sample_rate,
                target_sample_rate=VAD_SAMPLE_RATE,
                channels=1,
            )
        self._buffer += pcm

        ended = False
        while len(self._buffer) >= VAD_FRAME_BYTES:
            frame = bytes(self._buffer[:VAD_FRAME_BYTES])
            del self._buffer[:VAD_FRAME_BYTES]

            probability = self._vad(frame)
            self._observe(probability)

            if probability >= self._threshold:
                self._heard_speech = True
                self._frames_since_speech += 1
                self._silent_frames = 0
                continue

            if not self._heard_speech:
                # Silence before anyone has spoken is not the end of anything,
                # but it is worth counting: it is the dead air between one
                # utterance and the next.
                self._frames_before_speech += 1
                continue

            self._frames_since_speech += 1
            self._silent_frames += 1
            if self._silent_frames >= self._silence_frames_needed:
                ended = True
                break

        # A safety net, not a feature: without it, audio that Silero reads as
        # continuous speech -- sustained noise, a fan close to the microphone --
        # would buffer without limit and never reach the model at all.
        #
        # Gated on having heard speech, for the same reason the silence path
        # is. Without that gate a quiet session emits an empty utterance every
        # `max_seconds` forever, each one a model call returning "". A session
        # where nobody ever speaks still buffers until its epoch ends, which is
        # exactly what it did before this module existed.
        if not ended and self._heard_speech and self._frames_since_speech >= self._max_frames:
            logger.info(
                "utterance reached its length limit before any silence; cutting it",
                extra={"max_frames": self._max_frames},
            )
            ended = True

        return ended

    def consume_boundary(self) -> None:
        """Start a fresh utterance. Called once the caller has acted on the
        boundary, so `leading_silence_seconds` is still readable until then."""
        self._heard_speech = False
        self._silent_frames = 0
        self._frames_since_speech = 0
        self._frames_before_speech = 0


__all__ = [
    "VAD_FRAME_SAMPLES",
    "VAD_SAMPLE_RATE",
    "SileroBoundary",
    "SpeechBoundary",
    "VoiceDetector",
]
