"""No-hardware start path (role-prompts/Speech.md): prove gap detection before
touching any real STT/TTS model.

`audio_session_basic` is a recorded fixture with one deliberate 500ms silence
gap where `sequence` stays perfectly contiguous but `pts_samples` (the
cumulative sample count since the epoch began) jumps. A consumer that infers
audio continuity from message count -- or from `sequence` -- will never
notice the gap and will hand a transcriber two unrelated stretches of speech
stitched together as if they were continuous. This test flows the fixture
through the real `MediaClient`, with no gateway, no LiveKit, and no model
running, and asserts `continuity.py` -- this service's own continuity check,
not duplicated logic -- catches it by reading `pts_samples`.
"""

import pytest
from visual_memory_media_contract.client import MediaClient
from visual_memory_media_contract.protocol import AudioChunk, RelayMessage
from visual_memory_media_contract.testing import replay_server

from speech.continuity import ContinuityTracker


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_speech_detects_the_deliberate_gap_in_audio_session_basic() -> None:
    received: list[RelayMessage] = []
    async with replay_server("audio_session_basic") as url:
        async with MediaClient(url, reconnect=False) as client:
            async for message in client:
                received.append(message)

    chunks = [message for message in received if isinstance(message, AudioChunk)]
    assert chunks, "fixture produced no audio_chunk messages"

    # `sequence` alone looks perfectly continuous -- this is the check that
    # would pass even though audio was lost, which is the whole reason
    # `continuity.py` is built around `pts_samples` instead.
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))

    tracker = ContinuityTracker()
    gaps = [gap for chunk in chunks if (gap := tracker.check(chunk)) is not None]

    assert len(gaps) == 1, f"expected exactly one gap in this fixture, found {len(gaps)}"
    assert gaps[0].lost_seconds == pytest.approx(0.5, abs=0.01), (
        f"expected the fixture's documented 500ms gap, detected {gaps[0].lost_seconds * 1000:.0f}ms"
    )
