"""One LiveKit room, for one session.

The handler and teardown shape is the S01 spike's, which passed three
disconnect/rejoin cycles: synchronous `@room.on` callbacks that spawn consumer
tasks held by strong references, and an ordered shutdown that cancels the
consumer, closes the native stream, and only then disconnects the room.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from collections.abc import Callable
from typing import Any

from livekit import rtc
from visual_memory_media_contract.protocol import EpochEndReason, StreamKind

from media_gateway.config import Settings
from media_gateway.domain.session import Session
from media_gateway.relay.codec import RGBA_CHANNELS
from media_gateway.transport.return_audio import ReturnAudio
from media_gateway.transport.source import (
    MediaSink,
    RawAudioFrame,
    RawVideoFrame,
    utcnow,
)
from media_gateway.transport.tokens import mint_access_token, worker_identity

logger = logging.getLogger(__name__)

#: Configured names mapped to the SDK's protobuf enum.
VIDEO_QUALITY = {
    "low": rtc.VideoQuality.VIDEO_QUALITY_LOW,
    "medium": rtc.VideoQuality.VIDEO_QUALITY_MEDIUM,
    "high": rtc.VideoQuality.VIDEO_QUALITY_HIGH,
}


def _subscribe(room: rtc.Room, event: str, handler: Callable[..., None]) -> None:
    """Register a room callback.

    The SDK types `Room.on` as returning an unknown callable, so the
    suppression lives here once instead of at every call site.
    """
    room.on(event, handler)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]


class _SenderClock:
    """Maps the publisher's frame timestamps onto our wall clock.

    `VideoFrameEvent.timestamp_us` is in the sender's domain, so its absolute
    value cannot be compared against `utcnow()`. Anchoring it to our clock on
    the first frame keeps what actually matters -- the *spacing* between
    frames, which is the publisher's true cadence. Even production and bursty
    delivery then look different: `captured_at` stays evenly spaced while
    `received_at` clumps.

    Falls back to our own clock, once and loudly, if the transport reports no
    usable timestamp. Silence there would look exactly like the bug this
    replaces.
    """

    __slots__ = ("_offset", "_session_id", "_warned")

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._offset: dt.timedelta | None = None
        self._warned = False

    def stamp(self, event: object, received_at: dt.datetime) -> dt.datetime:
        raw = getattr(event, "timestamp_us", None)
        if not raw:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "transport reported no frame timestamp; "
                    "capture cadence will mirror receipt",
                    extra={"session_id": self._session_id},
                )
            return received_at
        sender = dt.datetime.fromtimestamp(raw / 1_000_000, tz=dt.UTC)
        if self._offset is None:
            self._offset = received_at - sender
            logger.info(
                "anchored sender clock",
                extra={
                    "session_id": self._session_id,
                    "offset_s": round(self._offset.total_seconds(), 3),
                },
            )
        return sender + self._offset


class RoomWorker:
    """Subscribes to one room's media and reports it into a sink."""

    def __init__(
        self,
        *,
        settings: Settings,
        session: Session,
        sink: MediaSink,
        on_participant_left: Callable[[str, str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._session = session
        self._sink = sink
        self._room = rtc.Room()
        self._tasks: set[asyncio.Task[None]] = set()
        self._connected = False
        #: Called with (session_id, participant_identity) whenever a
        #: participant other than the session's own publisher disconnects --
        #: today that is only ever a remote-assist helper. Lets the caller
        #: (main.py) end an accepted assist request without this worker
        #: knowing anything about the assist domain.
        self._on_participant_left = on_participant_left
        self.return_audio = ReturnAudio(
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            queue_size_ms=settings.return_audio_queue_ms,
            track_name=settings.return_audio_track_name,
        )

    @property
    def session(self) -> Session:
        return self._session

    @property
    def connected(self) -> bool:
        return self._connected

    # --- Lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Join the room and begin reporting media."""
        token = mint_access_token(
            self._settings,
            identity=worker_identity(self._session.session_id),
            room=self._session.room,
            role="worker",
        )
        self._register_handlers()

        await self._room.connect(
            self._settings.livekit_url,
            token.token,
            options=rtc.RoomOptions(
                auto_subscribe=True,
                connect_timeout=self._settings.livekit_connect_timeout_s,
            ),
        )
        self._connected = True
        # Publish the assistant's audio track immediately, so the device is
        # already subscribed when the first utterance arrives.
        await self.return_audio.publish_to(self._room)
        self._sink.session_started(
            session_id=self._session.session_id, device_id=self._session.device_id
        )
        logger.info(
            "joined a livekit room",
            extra={"session_id": self._session.session_id, "room": self._session.room},
        )

    async def stop(self, reason: EpochEndReason = "gateway_shutdown") -> None:
        """Leave the room, in the order the spike proved."""
        for stream_kind in ("video", "audio"):
            self._sink.epoch_ended(
                session_id=self._session.session_id,
                stream_kind=stream_kind,  # type: ignore[arg-type]
                reason=reason,
            )

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=2.0)
        self._tasks.clear()

        # Close the source before leaving: closing it unpublishes the track,
        # which needs a renegotiation that fails on a closed connection.
        with contextlib.suppress(Exception):
            await self.return_audio.aclose()
        with contextlib.suppress(Exception):
            await self._room.disconnect()
        self._connected = False
        logger.info("left a livekit room", extra={"session_id": self._session.session_id})

    def _spawn(self, coroutine: Any) -> None:
        """Keep a strong reference so an in-flight consumer is not collected."""
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- Room events -----------------------------------------------------

    def _register_handlers(self) -> None:
        """Register callbacks by call rather than by decorator.

        The SDK's `Room.on` is untyped, and as a decorator it erases the
        signature of everything it wraps. Registering by call keeps the
        handlers themselves fully typed.
        """
        room = self._room
        _subscribe(room, "track_subscribed", self._on_track_subscribed)
        _subscribe(room, "track_unsubscribed", self._on_track_unsubscribed)
        _subscribe(room, "participant_connected", self._on_participant_connected)
        _subscribe(room, "participant_disconnected", self._on_participant_disconnected)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        # Ingest only the session's own publisher. A room with a remote-assist
        # helper in it has a second participant able to publish a microphone
        # (docs/12's `helper` grant); without this check that track would
        # start a new audio epoch and reach Speech/the Agent, which would
        # transcribe and might reply to the helper instead of the wearer.
        if participant.identity != self._session.device_id:
            logger.info(
                "ignored a track from a non-publisher participant",
                extra={
                    "session_id": self._session.session_id,
                    "participant_identity": participant.identity,
                    "track_kind": "video" if track.kind == rtc.TrackKind.KIND_VIDEO else "audio",
                },
            )
            return

        # A new track SID is a new media epoch even when the participant
        # identity is unchanged. That is the spike's central finding.
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self._request_quality(publication)
            self._begin_epoch("video", publication.sid, participant.identity)
            self._spawn(self._consume_video(track))
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            self._begin_epoch("audio", publication.sid, participant.identity)
            self._spawn(self._consume_audio(track))

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del publication, participant
        kind: StreamKind = "video" if track.kind == rtc.TrackKind.KIND_VIDEO else "audio"
        self._sink.epoch_ended(
            session_id=self._session.session_id,
            stream_kind=kind,
            reason="track_unsubscribed",
        )

    def _on_participant_connected(self, participant: rtc.RemoteParticipant) -> None:
        # A helper joining must not be mistaken for the wearer (re)connecting
        # -- same identity check as the track filter above, for the same
        # reason: this room can now hold a second participant.
        if participant.identity != self._session.device_id:
            return
        self._session.publisher_present = True
        self._session.ever_published = True
        self._session.touch()

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity != self._session.device_id:
            if self._on_participant_left is not None:
                self._on_participant_left(self._session.session_id, participant.identity)
            return
        self._session.publisher_present = False
        for kind in ("video", "audio"):
            self._sink.epoch_ended(
                session_id=self._session.session_id,
                stream_kind=kind,  # type: ignore[arg-type]
                reason="participant_disconnected",
            )

    def _request_quality(self, publication: rtc.RemoteTrackPublication) -> None:
        """Ask for a specific simulcast layer.

        A publisher that simulcasts sends several layers and the SFU picks one
        per subscriber, defaulting to the lowest. This asks for a better one.

        It is a request, not a guarantee. Measured against a browser publisher
        the SFU kept sending 320x180 of a 1280x720 camera despite this call,
        with the requested and published sizes both logged below so the
        discrepancy is visible rather than silent. The reliable control is
        publisher-side: a single layer leaves the SFU no choice, which is what
        the dev publisher page does. Verify the size a real client actually
        delivers rather than assuming this settles it.
        """
        if not publication.simulcasted:
            return
        quality = VIDEO_QUALITY[self._settings.subscribe_video_quality]
        try:
            publication.set_video_quality(quality)
        except Exception:  # pragma: no cover - SDK or transport specific
            logger.warning(
                "could not select a simulcast layer; the lowest will be used",
                extra={
                    "session_id": self._session.session_id,
                    "requested": self._settings.subscribe_video_quality,
                },
            )
            return
        logger.info(
            "requested a simulcast layer",
            extra={
                "session_id": self._session.session_id,
                "track_sid": publication.sid,
                "requested": self._settings.subscribe_video_quality,
                "published_size": f"{publication.width}x{publication.height}",
            },
        )

    def _begin_epoch(
        self, stream_kind: StreamKind, track_sid: str, participant_identity: str
    ) -> None:
        self._session.publisher_present = True
        self._session.ever_published = True
        self._session.touch()
        self._sink.epoch_started(
            session_id=self._session.session_id,
            stream_kind=stream_kind,
            track_sid=track_sid,
            participant_identity=participant_identity,
        )

    # --- Consumers -------------------------------------------------------

    async def _consume_video(self, track: rtc.Track) -> None:
        """Report decoded frames. The dimension guard runs downstream."""
        stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGBA)
        sender_clock = _SenderClock(self._session.session_id)
        reported_rotation = False
        try:
            async for event in stream:
                received_at = utcnow()
                frame = event.frame
                buffer = bytes(frame.data)
                if not reported_rotation:
                    # The relay has always arrived portrait with nothing in this
                    # service asking for it; this says whether the publisher is
                    # rotating and by how much.
                    reported_rotation = True
                    logger.info(
                        "video track rotation",
                        extra={
                            "session_id": self._session.session_id,
                            "rotation": int(getattr(event, "rotation", 0) or 0),
                            "width": frame.width,
                            "height": frame.height,
                        },
                    )
                expected = frame.width * frame.height * RGBA_CHANNELS
                if len(buffer) != expected:
                    # A buffer that disagrees with its own dimensions cannot be
                    # decoded; drop it here rather than failing in the encoder.
                    logger.warning(
                        "discarded a video frame with an inconsistent buffer",
                        extra={
                            "session_id": self._session.session_id,
                            "width": frame.width,
                            "height": frame.height,
                            "buffer_bytes": len(buffer),
                        },
                    )
                    continue
                self._sink.video_frame(
                    session_id=self._session.session_id,
                    frame=RawVideoFrame(
                        width=frame.width,
                        height=frame.height,
                        rgba=buffer,
                        captured_at=sender_clock.stamp(event, received_at),
                        received_at=received_at,
                    ),
                )
        except asyncio.CancelledError:
            raise
        finally:
            await stream.aclose()

    async def _consume_audio(self, track: rtc.Track) -> None:
        """Report PCM. The SDK resamples and reframes to our settings."""
        stream = rtc.AudioStream(
            track,
            capacity=1,
            sample_rate=self._settings.audio_sample_rate,
            num_channels=self._settings.audio_channels,
            frame_size_ms=self._settings.audio_frame_ms,
        )
        try:
            async for event in stream:
                frame = event.frame
                self._sink.audio_frame(
                    session_id=self._session.session_id,
                    frame=RawAudioFrame(
                        pcm=bytes(frame.data),
                        samples=frame.samples_per_channel,
                        sample_rate=frame.sample_rate,
                        channels=frame.num_channels,
                        captured_at=utcnow(),
                    ),
                )
        except asyncio.CancelledError:
            raise
        finally:
            await stream.aclose()


__all__ = ["RoomWorker"]
