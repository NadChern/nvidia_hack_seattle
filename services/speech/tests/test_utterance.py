"""SileroBoundary: deciding when somebody stopped talking.

The gap this closes is not subtle. `ingest.py` ends a segment on a
`pts_samples` gap, an epoch, or the session -- and a microphone produces
samples continuously through silence, so none of those happen while somebody
is publishing. Measured before this existed: the gateway relayed 54,252 audio
frames and no transcript was ever produced, because no boundary had occurred.

The detector is injected throughout. Driving these against real audio would
mean every test needed a clip a real model agrees is speech, which is slow and
-- worse -- makes a state-machine bug indistinguishable from the model
disagreeing. That Silero itself separates speech from silence is a property of
Silero, measured once: 0.775 mean probability on synthesized speech against
0.005 on digital silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech.ingest import PREROLL_SECONDS
from speech.utterance import VAD_FRAME_SAMPLES, VAD_SAMPLE_RATE, SileroBoundary


class ScriptedDetector:
    """Returns a fixed probability per frame, from a repeating script."""

    def __init__(self, *probabilities: float) -> None:
        self._script = list(probabilities) or [0.0]
        self._index = 0
        self.resets = 0

    def __call__(self, frame: bytes, /) -> float:
        del frame
        value = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return value

    def reset(self) -> None:
        self.resets += 1


def pcm(frames: int, *, rate: int = VAD_SAMPLE_RATE) -> bytes:
    """`frames` VAD-sized frames of any content -- a scripted detector never
    looks at the samples, only at how many frames they amount to."""
    del rate
    return np.zeros(frames * VAD_FRAME_SAMPLES, dtype=np.int16).tobytes()


def speaking(seconds: float = 10.0) -> ScriptedDetector:
    del seconds
    return ScriptedDetector(1.0)


def test_silence_alone_never_ends_an_utterance() -> None:
    """A session that opens with nobody talking must not emit an endless run
    of empty utterances, each one a model call returning ""."""
    boundary = SileroBoundary(silence_ms=100, detector=ScriptedDetector(0.0))

    assert boundary.feed(pcm(200), sample_rate=VAD_SAMPLE_RATE) is False


def test_speech_then_a_pause_ends_exactly_one_utterance() -> None:
    detector = ScriptedDetector(*([1.0] * 10 + [0.0] * 100))
    # 100ms of silence is ~3 frames of 32ms.
    boundary = SileroBoundary(silence_ms=100, detector=detector)

    assert boundary.feed(pcm(10), sample_rate=VAD_SAMPLE_RATE) is False, "still speaking"
    assert boundary.feed(pcm(5), sample_rate=VAD_SAMPLE_RATE) is True, "the pause ends it"


def test_a_brief_gap_between_words_does_not_end_an_utterance() -> None:
    """Below roughly half a second this cuts people off at natural breaths --
    which is why the threshold is a duration rather than a single quiet frame."""
    # Speech, one quiet frame, then speech again.
    detector = ScriptedDetector(*([1.0] * 5 + [0.0] + [1.0] * 20))
    boundary = SileroBoundary(silence_ms=700, detector=detector)

    assert boundary.feed(pcm(26), sample_rate=VAD_SAMPLE_RATE) is False


def test_the_length_limit_cuts_speech_that_never_stops() -> None:
    """A ceiling, not a feature: audio the detector reads as unbroken speech
    would otherwise buffer without limit and never reach the model."""
    boundary = SileroBoundary(silence_ms=700, max_seconds=1.0, detector=speaking())

    assert boundary.feed(pcm(100), sample_rate=VAD_SAMPLE_RATE) is True


def test_the_length_limit_does_not_fire_on_a_quiet_session() -> None:
    """The bug this test was written for. Gating the limit on speech is what
    stops a silent session emitting an empty utterance every `max_seconds`,
    forever, each one a wasted model call."""
    boundary = SileroBoundary(silence_ms=700, max_seconds=0.5, detector=ScriptedDetector(0.0))

    fired = [boundary.feed(pcm(50), sample_rate=VAD_SAMPLE_RATE) for _ in range(10)]

    assert not any(fired)


def test_idle_time_does_not_spend_the_spoken_utterance_limit() -> None:
    """A wearer may pause indefinitely before speaking.

    The ceiling starts on the first speech frame; otherwise eight seconds of
    idle microphone time makes the first word end the turn immediately.
    """
    detector = ScriptedDetector(*([0.0] * 625 + [1.0] * 20))
    boundary = SileroBoundary(silence_ms=700, max_seconds=0.5, detector=detector)

    assert boundary.feed(pcm(625), sample_rate=VAD_SAMPLE_RATE) is False
    assert boundary.feed(pcm(1), sample_rate=VAD_SAMPLE_RATE) is False
    assert boundary.feed(pcm(13), sample_rate=VAD_SAMPLE_RATE) is False
    assert boundary.feed(pcm(1), sample_rate=VAD_SAMPLE_RATE) is True


def test_the_threshold_decides_what_counts_as_speech() -> None:
    quiet = SileroBoundary(silence_ms=100, threshold=0.9, detector=ScriptedDetector(0.6))
    # 0.6 is below 0.9, so nothing is ever speech and nothing can end.
    assert quiet.feed(pcm(50), sample_rate=VAD_SAMPLE_RATE) is False

    loud = SileroBoundary(
        silence_ms=100, threshold=0.5, detector=ScriptedDetector(*([0.6] * 5 + [0.0] * 20))
    )
    assert loud.feed(pcm(30), sample_rate=VAD_SAMPLE_RATE) is True


def test_reset_forgets_that_speech_was_heard() -> None:
    """A new epoch is not a continuation. Carrying `heard_speech` across one
    would let the first silence of a new epoch close an utterance belonging to
    the previous one."""
    detector = ScriptedDetector(*([1.0] * 5 + [0.0] * 200))
    boundary = SileroBoundary(silence_ms=100, max_seconds=1.0, detector=detector)
    boundary.feed(pcm(5), sample_rate=VAD_SAMPLE_RATE)

    boundary.reset()

    assert detector.resets == 1, "the model carries its own state and must be reset too"
    assert boundary.feed(pcm(100), sample_rate=VAD_SAMPLE_RATE) is False


@pytest.mark.parametrize("rate", [48_000, 24_000, 16_000])
def test_audio_is_accepted_at_whatever_rate_the_relay_sends(rate: int) -> None:
    """The gateway relays at its own rate (48kHz by default) and Silero only
    accepts 16kHz, so the conversion happens here rather than being assumed
    upstream. A wrong frame size would desynchronise the detector silently."""
    detector = ScriptedDetector(*([1.0] * 4 + [0.0] * 40))
    boundary = SileroBoundary(silence_ms=100, detector=detector)
    tenth = np.zeros(int(rate * 0.1), dtype=np.int16).tobytes()

    ended = any(boundary.feed(tenth, sample_rate=rate) for _ in range(10))

    assert ended, f"a pause at {rate}Hz should still end an utterance"


# --- Dead air between utterances ---------------------------------------------


def test_no_leading_silence_is_reported_before_speech_is_found() -> None:
    """The property that makes trimming safe. `ingest.py` drops this much audio,
    so a detector that hears nothing must report nothing to drop -- otherwise a
    false negative silently destroys a segment nobody can get back."""
    boundary = SileroBoundary(silence_ms=100, detector=ScriptedDetector(0.0))

    boundary.feed(pcm(200), sample_rate=VAD_SAMPLE_RATE)

    assert boundary.leading_silence_seconds == 0.0


def test_the_pause_before_speech_is_measured() -> None:
    """A sentence after a ten-second pause otherwise arrives as a twelve-second
    segment: mostly dead air, reported as the length of what was said, and six
    times more audio than the model needs to read."""
    # 10 quiet frames (~0.32s), then speech, then a pause.
    detector = ScriptedDetector(*([0.0] * 10 + [1.0] * 10 + [0.0] * 50))
    boundary = SileroBoundary(silence_ms=100, detector=detector)

    ended = boundary.feed(pcm(30), sample_rate=VAD_SAMPLE_RATE)

    assert ended is True
    expected = 10 * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE
    assert boundary.leading_silence_seconds == pytest.approx(expected)


def test_the_measurement_survives_until_the_caller_consumes_it() -> None:
    """`feed` returning True and the utterance restarting are deliberately two
    steps: the caller reads `leading_silence_seconds` in between, and resetting
    inside `feed` would leave it nothing to read."""
    detector = ScriptedDetector(*([0.0] * 5 + [1.0] * 5 + [0.0] * 50))
    boundary = SileroBoundary(silence_ms=100, detector=detector)
    boundary.feed(pcm(20), sample_rate=VAD_SAMPLE_RATE)
    assert boundary.leading_silence_seconds > 0

    boundary.consume_boundary()

    assert boundary.leading_silence_seconds == 0.0


@pytest.mark.parametrize("attack_frames", [0, 3, 6, 10])
def test_the_preroll_survives_a_slow_detector_attack(attack_frames: int) -> None:
    """The trim must not eat the first phoneme when the detector is late.

    A detector reports where it became *confident*, not where the sound began,
    and every frame it spent getting there is counted as leading silence and
    then trimmed. The margin actually kept is `preroll - attack`, so a slow
    attack cuts into real speech.

    Reported from the glasses as "Hey memory" losing its first sound, after
    which the wake prefix can never match and the assistant cannot be
    triggered at all. Measured against the old 0.3 s pre-roll: a 10-frame
    (320 ms) attack kept -4 ms, i.e. the onset was gone.
    """
    lead_frames = 60  # roughly two seconds of genuine silence first
    detector = ScriptedDetector(*([0.0] * (lead_frames + attack_frames) + [1.0] * 25 + [0.0] * 200))
    boundary = SileroBoundary(silence_ms=1_000, detector=detector)

    assert boundary.feed(pcm(lead_frames + attack_frames + 25 + 40), sample_rate=VAD_SAMPLE_RATE)

    reported_silence = boundary.leading_silence_seconds
    true_silence = lead_frames * VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE
    trimmed = reported_silence - PREROLL_SECONDS
    kept_before_speech = true_silence - trimmed

    assert kept_before_speech > 0.0, (
        f"a {attack_frames}-frame attack trimmed {-kept_before_speech * 1000:.0f} ms "
        "into the speech itself; the wake prefix would arrive clipped"
    )


def test_the_detector_reports_what_it_is_actually_seeing(caplog) -> None:
    """Telemetry has to distinguish the two opposite VAD failures.

    Measured on the glasses: 27 of 27 utterances hit the length ceiling and no
    silence boundary ever fired, so Parakeet was handed 20-second blocks and
    ran the GPU out of memory. Nothing in the logs said whether noise was
    reading as speech or the reverse, and those need opposite fixes.
    """
    boundary = SileroBoundary(silence_ms=1_000, detector=ScriptedDetector(0.9), telemetry_frames=10)

    with caplog.at_level("INFO", logger="speech.utterance"):
        boundary.feed(pcm(10), sample_rate=VAD_SAMPLE_RATE)

    summary = next(r for r in caplog.records if "probability summary" in r.message)
    assert summary.above_threshold == 1.0, "constant speech must read as 100% above"
    assert summary.mean_probability == pytest.approx(0.9)
    assert not hasattr(summary, "pcm"), "telemetry must never carry audio"


def test_telemetry_can_be_switched_off() -> None:
    boundary = SileroBoundary(silence_ms=1_000, detector=ScriptedDetector(0.9), telemetry_frames=0)

    assert boundary.feed(pcm(5), sample_rate=VAD_SAMPLE_RATE) is False
