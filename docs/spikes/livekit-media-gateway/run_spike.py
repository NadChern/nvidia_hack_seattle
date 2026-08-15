from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import math
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from livekit import api, rtc


API_KEY = "visual-memory-spike"
API_SECRET = "visual-memory-spike-secret-with-at-least-32-characters"
ROOM_NAME = "visual-memory-media-gateway-spike"
SERVER_URL = "ws://127.0.0.1:7880"
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 180
VIDEO_FPS = 10
SAMPLE_FPS = 2
AUDIO_SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 1
AUDIO_FRAME_MS = 20
PUBLISH_SECONDS = 1.8
RECONNECT_CYCLES = 3


@dataclass
class WorkerState:
    video_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    audio_tracks: dict[str, dict[str, Any]] = field(default_factory=dict)
    subscribed_events: list[dict[str, str]] = field(default_factory=list)
    unsubscribed_track_sids: list[str] = field(default_factory=list)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    return_audio_frames_sent: int = 0

    def add_task(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def token_for(identity: str, *, can_publish: bool, can_subscribe: bool) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(dt.timedelta(minutes=5))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM_NAME,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=False,
            )
        )
        .to_jwt()
    )


def publish_options(source: int, *, video: bool = False) -> rtc.TrackPublishOptions:
    if video:
        return rtc.TrackPublishOptions(
            source=source,
            simulcast=True,
            video_encoding=rtc.VideoEncoding(
                max_framerate=VIDEO_FPS,
                max_bitrate=800_000,
            ),
            degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
        )
    return rtc.TrackPublishOptions(source=source)


def make_video_frame(sequence: int) -> rtc.VideoFrame:
    frame = np.empty((VIDEO_HEIGHT, VIDEO_WIDTH, 4), dtype=np.uint8)
    frame[:, :, 0] = (sequence * 17) % 255
    frame[:, :, 1] = np.arange(VIDEO_WIDTH, dtype=np.uint16) % 255
    frame[:, :, 2] = np.arange(VIDEO_HEIGHT, dtype=np.uint16)[:, None] % 255
    frame[:, :, 3] = 255
    return rtc.VideoFrame(
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        rtc.VideoBufferType.RGBA,
        frame.tobytes(),
    )


def make_audio_frame(start_sample: int, frequency_hz: float) -> rtc.AudioFrame:
    samples_per_channel = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
    sample_numbers = np.arange(
        start_sample,
        start_sample + samples_per_channel,
        dtype=np.float64,
    )
    wave = np.sin(2.0 * math.pi * frequency_hz * sample_numbers / AUDIO_SAMPLE_RATE)
    pcm = (wave * 0.18 * np.iinfo(np.int16).max).astype(np.int16)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=AUDIO_SAMPLE_RATE,
        num_channels=AUDIO_CHANNELS,
        samples_per_channel=samples_per_channel,
    )


async def publish_video(source: rtc.VideoSource, seconds: float) -> int:
    count = int(seconds * VIDEO_FPS)
    interval = 1.0 / VIDEO_FPS
    started = time.perf_counter()
    for sequence in range(count):
        source.capture_frame(make_video_frame(sequence))
        deadline = started + (sequence + 1) * interval
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
    return count


async def publish_audio(
    source: rtc.AudioSource,
    seconds: float,
    frequency_hz: float,
) -> int:
    count = int(seconds * 1000 / AUDIO_FRAME_MS)
    samples_per_channel = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
    for sequence in range(count):
        frame = make_audio_frame(sequence * samples_per_channel, frequency_hz)
        await source.capture_frame(frame)
    return count


async def receive_worker_video(
    track_sid: str,
    track: rtc.Track,
    state: WorkerState,
) -> None:
    metrics = state.video_tracks[track_sid]
    latest: asyncio.Queue[tuple[int, int]] = asyncio.Queue(maxsize=1)
    stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.RGBA)

    async def sample_latest() -> None:
        interval = 1.0 / SAMPLE_FPS
        while True:
            await asyncio.sleep(interval)
            try:
                width, height = latest.get_nowait()
            except asyncio.QueueEmpty:
                continue
            metrics["sampled_frames"] += 1
            metrics["last_sample_dimensions"] = [width, height]

    sampler = asyncio.create_task(sample_latest())
    try:
        async for event in stream:
            metrics["received_frames"] += 1
            dimension_key = f"{event.frame.width}x{event.frame.height}"
            metrics["received_dimensions"][dimension_key] = (
                metrics["received_dimensions"].get(dimension_key, 0) + 1
            )
            if (event.frame.width, event.frame.height) != (
                VIDEO_WIDTH,
                VIDEO_HEIGHT,
            ):
                metrics["ignored_unexpected_dimensions"] += 1
                continue
            if latest.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    latest.get_nowait()
                    metrics["dropped_before_sampling"] += 1
            latest.put_nowait((event.frame.width, event.frame.height))
    finally:
        sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler
        await stream.aclose()


async def receive_worker_audio(
    track_sid: str,
    track: rtc.Track,
    state: WorkerState,
) -> None:
    metrics = state.audio_tracks[track_sid]
    stream = rtc.AudioStream(
        track,
        capacity=1,
        sample_rate=AUDIO_SAMPLE_RATE,
        num_channels=AUDIO_CHANNELS,
        frame_size_ms=AUDIO_FRAME_MS,
    )
    try:
        async for event in stream:
            metrics["received_frames"] += 1
            metrics["received_samples"] += event.frame.samples_per_channel
    finally:
        await stream.aclose()


async def return_audio_loop(
    source: rtc.AudioSource,
    state: WorkerState,
    stop: asyncio.Event,
) -> None:
    samples_per_channel = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
    sequence = 0
    while not stop.is_set():
        frame = make_audio_frame(sequence * samples_per_channel, 660.0)
        await source.capture_frame(frame)
        sequence += 1
        state.return_audio_frames_sent += 1


async def connect_worker() -> tuple[rtc.Room, WorkerState, rtc.AudioSource, asyncio.Event]:
    state = WorkerState()
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        state.subscribed_events.append(
            {
                "track_sid": publication.sid,
                "participant": participant.identity,
                "kind": str(track.kind),
            }
        )
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            state.video_tracks[publication.sid] = {
                "participant": participant.identity,
                "received_frames": 0,
                "sampled_frames": 0,
                "dropped_before_sampling": 0,
                "last_sample_dimensions": None,
                "received_dimensions": {},
                "ignored_unexpected_dimensions": 0,
            }
            state.add_task(receive_worker_video(publication.sid, track, state))
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            state.audio_tracks[publication.sid] = {
                "participant": participant.identity,
                "received_frames": 0,
                "received_samples": 0,
            }
            state.add_task(receive_worker_audio(publication.sid, track, state))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del track, participant
        state.unsubscribed_track_sids.append(publication.sid)

    await room.connect(
        SERVER_URL,
        token_for("gateway-worker", can_publish=True, can_subscribe=True),
        options=rtc.RoomOptions(auto_subscribe=True, connect_timeout=5.0),
    )

    return_source = rtc.AudioSource(
        AUDIO_SAMPLE_RATE,
        AUDIO_CHANNELS,
        queue_size_ms=200,
    )
    return_track = rtc.LocalAudioTrack.create_audio_track("assistant-tts", return_source)
    await room.local_participant.publish_track(
        return_track,
        publish_options(rtc.TrackSource.SOURCE_MICROPHONE),
    )
    stop = asyncio.Event()
    state.add_task(return_audio_loop(return_source, state, stop))
    return room, state, return_source, stop


async def run_publisher_cycle(cycle: int) -> dict[str, Any]:
    room = rtc.Room()
    reply_metrics = {"frames": 0, "samples": 0}
    reply_tasks: set[asyncio.Task[Any]] = set()

    async def receive_reply(track: rtc.Track) -> None:
        stream = rtc.AudioStream(
            track,
            capacity=1,
            sample_rate=AUDIO_SAMPLE_RATE,
            num_channels=AUDIO_CHANNELS,
            frame_size_ms=AUDIO_FRAME_MS,
        )
        try:
            async for event in stream:
                reply_metrics["frames"] += 1
                reply_metrics["samples"] += event.frame.samples_per_channel
        finally:
            await stream.aclose()

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del publication
        if (
            participant.identity == "gateway-worker"
            and track.kind == rtc.TrackKind.KIND_AUDIO
        ):
            task = asyncio.create_task(receive_reply(track))
            reply_tasks.add(task)
            task.add_done_callback(reply_tasks.discard)

    started = time.perf_counter()
    await room.connect(
        SERVER_URL,
        token_for("glasses-spike", can_publish=True, can_subscribe=True),
        options=rtc.RoomOptions(auto_subscribe=True, connect_timeout=5.0),
    )
    connect_ms = (time.perf_counter() - started) * 1000.0

    video_source = rtc.VideoSource(VIDEO_WIDTH, VIDEO_HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("camera", video_source)
    video_publication = await room.local_participant.publish_track(
        video_track,
        publish_options(rtc.TrackSource.SOURCE_CAMERA, video=True),
    )

    audio_source = rtc.AudioSource(
        AUDIO_SAMPLE_RATE,
        AUDIO_CHANNELS,
        queue_size_ms=200,
    )
    audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
    audio_publication = await room.local_participant.publish_track(
        audio_track,
        publish_options(rtc.TrackSource.SOURCE_MICROPHONE),
    )

    video_count, audio_count = await asyncio.gather(
        publish_video(video_source, PUBLISH_SECONDS),
        publish_audio(audio_source, PUBLISH_SECONDS, 440.0 + cycle * 30.0),
    )
    await asyncio.sleep(0.6)

    await room.disconnect()
    await video_source.aclose()
    await audio_source.aclose()
    if reply_tasks:
        _, pending = await asyncio.wait(reply_tasks, timeout=2.0)
        for task in pending:
            task.cancel()

    return {
        "cycle": cycle,
        "connect_ms": round(connect_ms, 1),
        "video_track_sid": video_publication.sid,
        "audio_track_sid": audio_publication.sid,
        "video_frames_published": video_count,
        "audio_frames_published": audio_count,
        "return_audio_frames_received": reply_metrics["frames"],
        "return_audio_samples_received": reply_metrics["samples"],
    }


async def invalid_token_is_rejected() -> tuple[bool, str]:
    good_token = token_for("invalid-auth-test", can_publish=False, can_subscribe=False)
    header, payload, signature = good_token.split(".")
    replacement = "a" if signature[0] != "a" else "b"
    bad_token = ".".join((header, payload, replacement + signature[1:]))
    room = rtc.Room()
    try:
        await room.connect(
            SERVER_URL,
            bad_token,
            options=rtc.RoomOptions(connect_timeout=3.0),
        )
    except Exception as exc:
        return True, type(exc).__name__
    else:
        await room.disconnect()
        return False, "connection unexpectedly succeeded"


def wait_for_port(process: subprocess.Popen[str], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"LiveKit exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", 7880), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("LiveKit did not open port 7880")


def inspect_server_network(process: psutil.Process) -> dict[str, Any]:
    listeners: list[str] = []
    nonlocal_established: list[str] = []
    local_addresses = {"127.0.0.1", "::1"}
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family in {socket.AF_INET, socket.AF_INET6}:
                local_addresses.add(address.address.split("%")[0])
    for connection in process.net_connections(kind="inet"):
        local = (
            f"{connection.laddr.ip}:{connection.laddr.port}"
            if connection.laddr
            else ""
        )
        remote = (
            f"{connection.raddr.ip}:{connection.raddr.port}"
            if connection.raddr
            else ""
        )
        if connection.status == psutil.CONN_LISTEN:
            listeners.append(local)
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr:
            remote_ip = connection.raddr.ip.split("%")[0]
            if remote_ip not in local_addresses:
                nonlocal_established.append(f"{local}->{remote}")
    return {
        "listeners": sorted(set(listeners)),
        "local_interface_addresses": sorted(local_addresses),
        "nonlocal_established_connections": sorted(set(nonlocal_established)),
    }


async def exercise_gateway(server_process: subprocess.Popen[str]) -> dict[str, Any]:
    auth_rejected, auth_error = await invalid_token_is_rejected()

    worker_started = time.perf_counter()
    worker_room, worker_state, return_source, return_stop = await connect_worker()
    worker_connect_ms = (time.perf_counter() - worker_started) * 1000.0

    cycles: list[dict[str, Any]] = []
    for cycle in range(1, RECONNECT_CYCLES + 1):
        cycles.append(await run_publisher_cycle(cycle))
        await asyncio.sleep(0.5)

    await asyncio.sleep(0.8)
    process = psutil.Process(server_process.pid)
    network = inspect_server_network(process)
    memory_rss_mb = process.memory_info().rss / (1024 * 1024)

    return_stop.set()
    await worker_room.disconnect()
    await return_source.aclose()
    if worker_state.tasks:
        _, pending = await asyncio.wait(worker_state.tasks, timeout=2.0)
        for task in pending:
            task.cancel()

    video_track_sids = [cycle["video_track_sid"] for cycle in cycles]
    audio_track_sids = [cycle["audio_track_sid"] for cycle in cycles]
    assertions = {
        "invalid_token_rejected": auth_rejected,
        "three_publish_cycles_completed": len(cycles) == RECONNECT_CYCLES,
        "new_video_track_sid_per_cycle": len(set(video_track_sids))
        == RECONNECT_CYCLES,
        "new_audio_track_sid_per_cycle": len(set(audio_track_sids))
        == RECONNECT_CYCLES,
        "worker_received_every_video_track": len(worker_state.video_tracks)
        == RECONNECT_CYCLES
        and all(
            item["received_frames"] > 0 for item in worker_state.video_tracks.values()
        ),
        "worker_received_every_audio_track": len(worker_state.audio_tracks)
        == RECONNECT_CYCLES
        and all(
            item["received_frames"] > 0 for item in worker_state.audio_tracks.values()
        ),
        "bounded_sampler_processed_frames": all(
            item["sampled_frames"] > 0
            and item["sampled_frames"] < item["received_frames"]
            and item["dropped_before_sampling"] > 0
            for item in worker_state.video_tracks.values()
        ),
        "decoded_video_dimensions_preserved": all(
            item["last_sample_dimensions"] == [VIDEO_WIDTH, VIDEO_HEIGHT]
            for item in worker_state.video_tracks.values()
        ),
        "return_audio_received_every_cycle": all(
            cycle["return_audio_frames_received"] > 0 for cycle in cycles
        ),
        "server_has_no_nonlocal_established_connection": not network[
            "nonlocal_established_connections"
        ],
    }

    return {
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "assertions": assertions,
        "authentication": {
            "invalid_token_error": auth_error,
        },
        "worker_connect_ms": round(worker_connect_ms, 1),
        "publisher_cycles": cycles,
        "worker_video_tracks": worker_state.video_tracks,
        "worker_audio_tracks": worker_state.audio_tracks,
        "worker_unsubscribed_track_sids": worker_state.unsubscribed_track_sids,
        "return_audio_frames_sent": worker_state.return_audio_frames_sent,
        "server": {
            "pid": server_process.pid,
            "rss_mb": round(memory_rss_mb, 1),
            **network,
        },
    }


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    default_server = (
        directory / ".tools" / "livekit_1.13.4" / "livekit-server.exe"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-bin", type=Path, default=default_server)
    parser.add_argument("--config", type=Path, default=directory / "livekit.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.server_bin.exists():
        print(f"LiveKit server not found: {args.server_bin}", file=sys.stderr)
        return 2

    server_version = subprocess.run(
        [str(args.server_bin), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    logs: deque[str] = deque(maxlen=40)
    server = subprocess.Popen(
        [str(args.server_bin), "--config", str(args.config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def drain_logs() -> None:
        assert server.stdout is not None
        for line in server.stdout:
            logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    try:
        wait_for_port(server)
        started = time.perf_counter()
        result = asyncio.run(exercise_gateway(server))
        result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        result["environment"] = {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "livekit_server": server_version,
            "livekit_rtc": "1.1.13",
            "livekit_api": "1.2.0",
            "server_url": SERVER_URL,
            "room": ROOM_NAME,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    except Exception as exc:
        failure = {
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "server_log_tail": list(logs),
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        log_thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
