"""Publish media into the gateway's LiveKit room.

Stands in for the glasses. Asks the gateway for a token exactly as a real
client would, so this exercises the token endpoint from the client side rather
than reaching around it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from livekit import rtc

from media_gateway.publisher.sources import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    Media,
    VideoOut,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Grant:
    """What the gateway returns for a session request."""

    session_id: str
    room: str
    livekit_url: str
    token: str
    identity: str


def request_session(gateway: str, *, device_id: str, token: str | None = None) -> Grant:
    """Ask the gateway to start a session and issue a publisher token."""
    request = urllib.request.Request(
        f"{gateway.rstrip('/')}/v1/sessions",
        data=json.dumps({"device_id": device_id}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    if token:
        request.add_header("authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"gateway refused the session ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach the gateway at {gateway}: {exc.reason}") from exc

    return Grant(
        session_id=body["session_id"],
        room=body["room"],
        livekit_url=body["livekit_url"],
        token=body["token"],
        identity=body["identity"],
    )


def delete_session(gateway: str, session_id: str, *, token: str | None = None) -> None:
    """End a session so it does not hold a slot until its TTL expires."""
    request = urllib.request.Request(
        f"{gateway.rstrip('/')}/v1/sessions/{session_id}", method="DELETE"
    )
    if token:
        request.add_header("authorization", f"Bearer {token}")
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except urllib.error.URLError as exc:  # pragma: no cover - best effort
        logger.warning("could not delete the session: %s", exc)


class VirtualGlasses:
    """Joins a room and publishes one video and one audio track."""

    def __init__(self, *, grant: Grant, width: int, height: int, realtime: bool) -> None:
        self._grant = grant
        self._width = width
        self._height = height
        self._realtime = realtime
        self._room = rtc.Room()
        self._video: rtc.VideoSource | None = None
        self._audio: rtc.AudioSource | None = None
        self._return_task: asyncio.Task[None] | None = None
        self.return_audio_frames = 0

    async def __aenter__(self) -> VirtualGlasses:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def connect(self) -> None:
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            # The gateway publishes synthesized speech back on this path.
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(
                    "receiving return audio '%s' from %s",
                    publication.name,
                    participant.identity,
                )
                self._spawn_return_audio(track)

        self._room.on("track_subscribed", on_track_subscribed)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        await self._room.connect(
            self._grant.livekit_url,
            self._grant.token,
            options=rtc.RoomOptions(auto_subscribe=True, connect_timeout=10.0),
        )

        self._video = rtc.VideoSource(self._width, self._height)
        video_track = rtc.LocalVideoTrack.create_video_track("camera", self._video)
        await self._room.local_participant.publish_track(
            video_track,
            # One layer. With simulcast the SFU picks a layer per subscriber and
            # in practice keeps sending the lowest, so a 720p source would reach
            # detection downscaled.
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False),
        )

        self._audio = rtc.AudioSource(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, queue_size_ms=200)
        audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", self._audio)
        await self._room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

    def _spawn_return_audio(self, track: rtc.Track) -> None:
        async def drain() -> None:
            stream = rtc.AudioStream(track, capacity=1)
            try:
                async for _ in stream:
                    self.return_audio_frames += 1
            finally:
                await stream.aclose()

        # Held on the instance so an in-flight consumer is not collected.
        self._return_task = asyncio.create_task(drain())

    async def publish(self, media: Iterator[Media]) -> tuple[int, int]:
        """Publish a stream of frames, returning the video and audio counts."""
        if self._video is None or self._audio is None:  # pragma: no cover - guarded by connect
            raise RuntimeError("connect() must run before publish()")

        started = time.monotonic()
        videos = audios = 0

        for item in media:
            if self._realtime:
                # Absolute deadlines: relative sleeps accumulate drift over a
                # long clip.
                delay = item.at_s - (time.monotonic() - started)
                if delay > 0:
                    await asyncio.sleep(delay)

            if isinstance(item, VideoOut):
                self._video.capture_frame(
                    rtc.VideoFrame(item.width, item.height, rtc.VideoBufferType.RGBA, item.rgba)
                )
                videos += 1
            else:
                await self._audio.capture_frame(
                    rtc.AudioFrame(
                        data=item.pcm,
                        sample_rate=AUDIO_SAMPLE_RATE,
                        num_channels=AUDIO_CHANNELS,
                        samples_per_channel=item.samples,
                    )
                )
                audios += 1

            if not self._realtime:
                # Yield so the SDK's senders drain rather than queueing behind
                # a tight loop.
                await asyncio.sleep(0)

        return videos, audios

    async def close(self) -> None:
        """Close the sources first, then leave the room.

        Closing a source unpublishes its track, which needs a renegotiation.
        Doing that after disconnecting makes the SDK attempt it on a closed
        peer connection and log a confusing "wrong state: closed" error, even
        though the publish itself succeeded.
        """
        if self._return_task is not None:
            self._return_task.cancel()
        if self._video is not None:
            await self._video.aclose()
        if self._audio is not None:
            await self._audio.aclose()
        await self._room.disconnect()


__all__ = ["Grant", "VirtualGlasses", "delete_session", "request_session"]
