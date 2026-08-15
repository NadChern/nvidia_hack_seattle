from __future__ import annotations

import io
import wave

import pytest

from agent.config import Settings
from agent.reply import ReplyTransport, pcm_from_wav

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def wav_bytes(*, sample_rate: int = 24_000, channels: int = 1, frames: int = 2_400) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames * channels * 2))
    return output.getvalue()


def test_wav_is_resampled_to_the_gateway_contract() -> None:
    pcm = pcm_from_wav(
        wav_bytes(),
        target_sample_rate=48_000,
        target_channels=1,
        max_bytes=1_000_000,
    )

    # ratecv may differ by one output sample at the boundary.
    assert 9_590 <= len(pcm) <= 9_600


def test_invalid_or_oversized_wav_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        pcm_from_wav(
            b"not a wav",
            target_sample_rate=48_000,
            target_channels=1,
            max_bytes=4,
        )


async def test_reply_synthesizes_and_sends_only_pcm() -> None:
    sent: list[tuple[str, bytes]] = []
    synthesized: list[str] = []

    async def synthesize(text: str) -> bytes:
        synthesized.append(text)
        return wav_bytes(sample_rate=48_000, frames=960)

    async def send(session_id: str, pcm: bytes) -> None:
        sent.append((session_id, pcm))

    transport = ReplyTransport(
        Settings(environment="ci"),
        synthesize=synthesize,
        send_pcm=send,
    )

    await transport.send("sess_01", "A guarded answer.")

    assert synthesized == ["A guarded answer."]
    assert sent == [("sess_01", bytes(960 * 2))]
