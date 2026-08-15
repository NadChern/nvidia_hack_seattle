"""Audio-ingestion consumer: turns the Media Gateway's audio relay into
contiguous PCM segments. No decoding, playback, resampling, or
transcription -- those are later stages of the Speech role, not this one.

Implements, per `role-prompts/Speech.md`'s "Audio in" section:

- **Reset on `epoch_started`.** A reconnect is *not* handled as a separate
  case: `MediaClient` re-sends a synthetic `epoch_started` for every
  still-active epoch after reconnecting (see its own docstring), so a
  consumer that resets correctly on `epoch_started` already covers both a
  real epoch change and a reconnect. There is no separate "reconnected"
  signal exposed to hook into, and none is needed. Resetting means
  *discarding* whatever audio was buffered, not finishing it as a segment --
  it can only be an interrupted fragment, since a legitimately complete run
  ends via `epoch_ended`, not `epoch_started`.
- **Never silently concatenate non-contiguous audio.** Every chunk is checked
  against the previous one via `continuity.py`; a detected gap finishes the
  current segment (if it has one) and starts a new one.
- **Compare the gateway's declared sample rate against what this service
  expects**, logging a mismatch instead of assuming -- see
  `Settings.expected_audio_sample_rate` in `config.py`.

Explicitly NOT implemented here, and not faked with dead code: distinctly
logging a `1011 audio_backpressure` close as something other than a generic
disconnect. `docs/12-Media-Relay-Contract.md` documents that the gateway
sends 1011 specifically when a subscriber's audio queue overflows, but
`MediaClient`'s public API (`packages/media-contract/src/
visual_memory_media_contract/client.py`) does not expose *why* a connection
closed to its consumer. Only close code 1008 (policy violation) gets special
handling there, re-raised as `MediaClientError`; every other close --
including 1011 -- is re-raised as a bare `websockets.ConnectionClosed`,
caught generically inside `MediaClient.__aiter__`'s broad
`except (OSError, websockets.WebSocketException)`, logged by the *client's
own* logger as a generic "disconnected, reconnecting" warning, and never
surfaced to this module in any form. There is no private/internal path
around this that isn't "hand-parsing the stream" or reaching into
`MediaClient` internals, both of which are against this role's hard rules.
Flagged to SY to raise with Alex: the same special-case pattern already used
for 1008 could plausibly be extended to 1011.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict
from visual_memory_media_contract.client import MediaClient, ReconnectPolicy
from visual_memory_media_contract.protocol import (
    AudioChunk,
    EpochEnded,
    EpochStarted,
    RelayMessage,
    SampleFormat,
    SessionEnded,
    UtcTimestamp,
)

from speech.config import get_settings
from speech.continuity import ContinuityTracker
from speech.utterance import SpeechBoundary

logger = logging.getLogger(__name__)

#: How much audio before *detected* speech is kept.
#:
#: A detector reports where it became confident, not where the sound began, and
#: every frame it spent getting there is counted as leading silence and then
#: trimmed. The effective margin is therefore `PREROLL - attack_delay`, so a
#: slow attack eats straight into the first phoneme.
#:
#: Measured against the real boundary with a scripted detector: at a 320 ms
#: attack a 0.3 s pre-roll kept **-4 ms** before speech -- the onset was gone.
#: On the glasses that is the common case, not the corner: "hey" opens on a
#: low-energy /h/, and the head-worn microphone runs noise suppression and AGC
#: that flatten exactly that kind of onset. The wake prefix then never matches
#: and the assistant cannot be triggered at all.
#:
#: 0.6 s because the risk is asymmetric. Keeping too much costs a fraction of a
#: second of audio the model would have skipped; keeping too little silently
#: breaks the only way to talk to the assistant.
PREROLL_SECONDS = 0.6


class AudioSegment(BaseModel):
    """A run of contiguous PCM audio with no detected `pts_samples` gap inside it.

    Boundary model for whatever consumes this module's output later (real STT
    wiring is a later stage, not this one). Carries raw PCM bytes rather than
    a decoded array -- decoding is also a later stage's job.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    epoch_id: str
    sample_rate: int
    channels: int
    sample_format: SampleFormat
    pts_samples_start: int
    samples: int
    first_sample_captured_at: UtcTimestamp
    #: Raw audio. Never pass this to a logger -- `logging.py`'s redaction
    #: filter would reduce it to a byte count as defense in depth, but the
    #: intent is to never hand it to a log call at all.
    pcm: bytes

    @property
    def duration_seconds(self) -> float:
        return self.samples / self.sample_rate


@dataclass
class _SegmentBuilder:
    """Accumulates chunks into one contiguous `AudioSegment`.

    Initialized from the first chunk it sees, since that is where the actual
    per-chunk metadata (sample rate, format, starting `pts_samples`) lives.
    """

    session_id: str
    epoch_id: str
    sample_rate: int
    channels: int
    sample_format: SampleFormat
    pts_samples_start: int
    first_sample_captured_at: UtcTimestamp
    _payload: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _samples: int = field(default=0, init=False)

    @classmethod
    def starting_with(cls, chunk: AudioChunk) -> _SegmentBuilder:
        builder = cls(
            session_id=chunk.session_id,
            epoch_id=chunk.epoch_id,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            sample_format=chunk.sample_format,
            pts_samples_start=chunk.pts_samples,
            first_sample_captured_at=chunk.first_sample_captured_at,
        )
        builder.add(chunk)
        return builder

    @property
    def samples_so_far(self) -> int:
        return self._samples

    def drop_leading_seconds(self, seconds: float) -> None:
        """Discard `seconds` from the front of this segment.

        Without it a segment starts at the previous utterance's boundary, so a
        two-second sentence after a ten-second pause arrives as a twelve-second
        segment: the duration reported to a consumer is mostly dead air, and
        the model transcribes six times more audio than it needs to.

        `pts_samples_start` and `first_sample_captured_at` move with the data.
        They are what locates this audio back in the relay stream, and leaving
        them pointing at discarded samples would make every downstream
        timestamp wrong by the length of the silence.
        """
        dropped = min(round(seconds * self.sample_rate), self._samples)
        if dropped <= 0:
            return
        del self._payload[: dropped * self.channels * 2]
        self._samples -= dropped
        self.pts_samples_start += dropped
        self.first_sample_captured_at += dt.timedelta(seconds=dropped / self.sample_rate)

    def add(self, chunk: AudioChunk) -> None:
        settings = get_settings()
        if chunk.sample_rate != settings.expected_audio_sample_rate:
            logger.warning(
                "audio chunk sample rate does not match this service's configured "
                "expectation; not resampling, later stages will need this resolved",
                extra={
                    "expected_sample_rate": settings.expected_audio_sample_rate,
                    "actual_sample_rate": chunk.sample_rate,
                    "session_id": chunk.session_id,
                    "epoch_id": chunk.epoch_id,
                },
            )
        self._payload += chunk.payload
        self._samples += chunk.samples

    def finish(self) -> AudioSegment:
        return AudioSegment(
            session_id=self.session_id,
            epoch_id=self.epoch_id,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_format=self.sample_format,
            pts_samples_start=self.pts_samples_start,
            samples=self._samples,
            first_sample_captured_at=self.first_sample_captured_at,
            pcm=bytes(self._payload),
        )


async def _segments_from_messages(
    messages: AsyncIterable[RelayMessage],
    *,
    session_id: str | None = None,
    boundary: SpeechBoundary | None = None,
    preroll_seconds: float = PREROLL_SECONDS,
) -> AsyncGenerator[AudioSegment, None]:
    """`session_id`, when given, scopes this to one session's messages.

    The relay has no per-session URL (`docs/12-Media-Relay-Contract.md`):
    every consumer connects to the same shared stream, and every message
    type matched below carries `session_id` as a field for exactly this
    reason. Without this filter, two sessions active on the same relay
    connection at once would interleave their epochs and audio chunks into
    one `ContinuityTracker`/`_SegmentBuilder` pair, corrupting both --
    `pts_samples` from session A would be compared against session B's
    previous chunk. `None` (the default) means "don't filter," preserving
    the original single-session-assumed behaviour exactly.
    """
    tracker = ContinuityTracker()
    builder: _SegmentBuilder | None = None

    def _for_this_session(message_session_id: str) -> bool:
        return session_id is None or message_session_id == session_id

    async for message in messages:
        match message:
            case EpochStarted():
                if not _for_this_session(message.session_id):
                    continue
                # Discard, don't finish. If a builder exists here, it can only
                # be a fragment left over from an epoch that never reached a
                # clean `epoch_ended` -- e.g. a reconnect cut it short. It is
                # not a complete, trustworthy segment, so it must not be
                # emitted as one (role-prompts/Speech.md, "Audio in": reset
                # discards "any partially accumulated temporal event").
                if builder is not None:
                    logger.warning(
                        "epoch_started arrived with unflushed audio buffered; "
                        "discarding it as an incomplete fragment",
                        extra={
                            "discarded_samples": builder.samples_so_far,
                            "session_id": builder.session_id,
                            "epoch_id": builder.epoch_id,
                        },
                    )
                builder = None
                tracker.reset()
                if boundary is not None:
                    boundary.reset()
            case EpochEnded():
                if not _for_this_session(message.session_id):
                    continue
                if builder is not None:
                    yield builder.finish()
                builder = None
                tracker.reset()
                if boundary is not None:
                    boundary.reset()
            case AudioChunk():
                if not _for_this_session(message.session_id):
                    continue
                gap = tracker.check(message)
                if gap is not None:
                    logger.info(
                        "audio continuity gap detected; starting a new segment",
                        extra={
                            "session_id": message.session_id,
                            "epoch_id": message.epoch_id,
                            "lost_samples": gap.lost_samples,
                            "lost_seconds": round(gap.lost_seconds, 3),
                        },
                    )
                    if builder is not None:
                        yield builder.finish()
                    builder = None
                if builder is None:
                    builder = _SegmentBuilder.starting_with(message)
                else:
                    builder.add(message)

                # An *utterance* boundary, distinct from the continuity
                # boundary above. A pause in speech produces no `pts_samples`
                # gap -- a microphone keeps sending samples through silence --
                # so without this a live session yields nothing until its
                # epoch ends. `None` (the default) keeps the original
                # gap-and-epoch-only behaviour exactly. See `utterance.py`.
                if boundary is not None and boundary.feed(
                    message.payload, sample_rate=message.sample_rate
                ):
                    # Drop the dead air before the speech -- but only here,
                    # where speech has definitely been found. Trimming
                    # speculatively, while waiting to see whether anyone
                    # speaks, would mean a detector false negative silently
                    # discarded audio nobody could get back.
                    builder.drop_leading_seconds(boundary.leading_silence_seconds - preroll_seconds)
                    boundary.consume_boundary()
                    yield builder.finish()
                    builder = None
            case SessionEnded():
                if not _for_this_session(message.session_id):
                    # A *different* session ending must not stop this one --
                    # the relay connection is shared.
                    continue
                # The session is over -- there is nothing left to ingest, ever,
                # on this connection. This also has to be the thing that ends
                # iteration: with `reconnect=True`, `MediaClient` cannot tell
                # "the session legitimately ended" apart from "the gateway
                # blipped," so a clean close alone makes it retry forever.
                if builder is not None:
                    yield builder.finish()
                return
            case _:
                continue

    if builder is not None:
        yield builder.finish()


async def ingest_segments(
    url: str,
    *,
    token: str | None = None,
    reconnect: bool = True,
    policy: ReconnectPolicy | None = None,
    session_id: str | None = None,
    boundary: SpeechBoundary | None = None,
    preroll_seconds: float = PREROLL_SECONDS,
) -> AsyncGenerator[AudioSegment, None]:
    """Connect to the gateway's audio relay and yield contiguous PCM segments.

    `session_id` scopes this to one session, ignoring every other session's
    messages on the same shared relay connection -- see
    `_segments_from_messages`.

    `boundary`, when given, additionally ends a segment where somebody stopped
    talking. Without it a segment ends only on a continuity gap, an epoch, or
    the session -- none of which happen while a live microphone is publishing,
    so nothing is emitted until the wearer stops. See `utterance.py`.
    """
    async with MediaClient(url, token=token, reconnect=reconnect, policy=policy) as client:
        async for segment in _segments_from_messages(
            client,
            session_id=session_id,
            boundary=boundary,
            preroll_seconds=preroll_seconds,
        ):
            yield segment


__all__ = ["AudioSegment", "ingest_segments"]
