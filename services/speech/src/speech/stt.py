"""Typed STT boundary: the `Transcript` result shape, the `SpeechToText`
interface, and a deterministic stub implementation with no real model.

Scope, deliberately minimal: nothing here concerns itself with Memory queries
or an Agent layer. Whether a `Transcript` goes to Memory directly or through
a separate Agent-layer component is an open question tracked in
`role-prompts/Speech.md`, not something this boundary model should presuppose
an answer to.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from speech.ingest import AudioSegment


class Transcript(BaseModel):
    """What one `AudioSegment` transcribed to.

    Carries enough to locate the source audio it came from -- `session_id`,
    `epoch_id`, and `pts_samples_start` together point back into the
    original relay stream regardless of anything a later resampling step did
    to the segment's own encoding. `samples`/`sample_rate` describe the audio
    that was actually fed to the model, which may be resampled.
    """

    model_config = ConfigDict(frozen=True)

    #: The wearer's own speech. Never pass this to a logger -- docs/07 treats
    #: transcript content the same as raw media.
    text: str
    session_id: str
    epoch_id: str
    pts_samples_start: int
    samples: int
    sample_rate: int


class SpeechToText(Protocol):
    """Interface every STT backend (stub or real) implements."""

    async def transcribe(self, segment: AudioSegment) -> Transcript: ...


class StubSpeechToText:
    """Deterministic placeholder -- no real model.

    Exists so the full pipeline (ingest -> resample -> transcribe) is
    testable end to end before Parakeet is wired in. The "text" it returns is
    derived entirely from the segment's own metadata, not from the audio
    content, so it is intentionally not meaningful as a transcript -- only
    useful for proving the plumbing around it works.
    """

    async def transcribe(self, segment: AudioSegment) -> Transcript:
        return Transcript(
            text=(
                f"<stub transcript: epoch={segment.epoch_id} "
                f"pts_start={segment.pts_samples_start} samples={segment.samples}>"
            ),
            session_id=segment.session_id,
            epoch_id=segment.epoch_id,
            pts_samples_start=segment.pts_samples_start,
            samples=segment.samples,
            sample_rate=segment.sample_rate,
        )


__all__ = ["SpeechToText", "StubSpeechToText", "Transcript"]
