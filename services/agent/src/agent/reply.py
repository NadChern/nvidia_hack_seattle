"""Synthesize a guarded answer and stream PCM to gateway return audio."""

from __future__ import annotations

import audioop
import io
import wave
from collections.abc import Awaitable, Callable

import httpx
from websockets.asyncio.client import connect

from agent.config import Settings
from agent.errors import DependencyUnavailableError

Synthesize = Callable[[str], Awaitable[bytes]]
SendPcm = Callable[[str, bytes], Awaitable[None]]


def pcm_from_wav(
    payload: bytes,
    *,
    target_sample_rate: int,
    target_channels: int,
    max_bytes: int,
) -> bytes:
    """Validate a self-describing WAV and convert it to gateway s16le PCM."""
    if not payload or len(payload) > max_bytes:
        raise ValueError("synthesized WAV is empty or exceeds the configured limit")

    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("synthesized WAV must contain uncompressed PCM")
            sample_width = wav_file.getsampwidth()
            source_channels = wav_file.getnchannels()
            source_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error) as exc:
        raise ValueError("speech service returned an invalid WAV") from exc

    if sample_width != 2:
        raise ValueError("synthesized WAV must contain 16-bit PCM")
    if source_channels not in {1, 2} or target_channels not in {1, 2}:
        raise ValueError("only mono or stereo reply audio is supported")
    if source_rate <= 0:
        raise ValueError("synthesized WAV has an invalid sample rate")

    if source_channels == 2 and target_channels == 1:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
    elif source_channels == 1 and target_channels == 2:
        pcm = audioop.tostereo(pcm, sample_width, 1.0, 1.0)

    if source_rate != target_sample_rate:
        pcm, _ = audioop.ratecv(
            pcm,
            sample_width,
            target_channels,
            source_rate,
            target_sample_rate,
            None,
        )
    return pcm


class ReplyTransport:
    """Owns the Speech HTTP and gateway WebSocket reply path."""

    def __init__(
        self,
        settings: Settings,
        *,
        synthesize: Synthesize | None = None,
        send_pcm: SendPcm | None = None,
    ) -> None:
        self._settings = settings
        self._synthesize = synthesize or self._synthesize_over_http
        self._send_pcm = send_pcm or self._send_over_websocket

    def _headers(self) -> dict[str, str]:
        token = self._settings.internal_api_token
        return {"authorization": f"Bearer {token.get_secret_value()}"} if token is not None else {}

    async def _synthesize_over_http(self, text: str) -> bytes:
        async with httpx.AsyncClient(
            base_url=self._settings.speech_base_url,
            headers=self._headers(),
            timeout=self._settings.request_timeout_s,
        ) as client:
            response = await client.post("/v1/synthesize", json={"text": text})
            response.raise_for_status()
            return response.content

    async def _send_over_websocket(self, session_id: str, pcm: bytes) -> None:
        base = self._settings.gateway_base_url
        scheme = "wss" if base.startswith("https://") else "ws"
        host_and_path = base.split("://", maxsplit=1)[1]
        url = f"{scheme}://{host_and_path}/v1/return-audio/{session_id}"
        async with connect(
            url,
            additional_headers=self._headers(),
            max_size=self._settings.max_synthesis_bytes,
            open_timeout=self._settings.request_timeout_s,
        ) as websocket:
            await websocket.send(pcm)

    async def send(self, session_id: str, text: str) -> None:
        """Synthesize and forward one answer; never log ``text``."""
        try:
            wav = await self._synthesize(text)
            pcm = pcm_from_wav(
                wav,
                target_sample_rate=self._settings.gateway_audio_sample_rate,
                target_channels=self._settings.gateway_audio_channels,
                max_bytes=self._settings.max_synthesis_bytes,
            )
            await self._send_pcm(session_id, pcm)
        except DependencyUnavailableError:
            raise
        except (httpx.HTTPError, OSError, TimeoutError, ValueError) as exc:
            raise DependencyUnavailableError("reply audio path is unavailable") from exc


__all__ = ["ReplyTransport", "SendPcm", "Synthesize", "pcm_from_wav"]
