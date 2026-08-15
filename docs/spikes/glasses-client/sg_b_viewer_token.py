"""SG-B: does a read-only console viewer perturb gateway ingest?

Three participants in one room, against a real local LiveKit:
  glasses  - publishes 1280x720 simulcast video + audio (what the X3 Pro will do)
  worker   - the gateway: can_publish + can_subscribe, subscribes to everything
  viewer   - the proposed console role: can_publish=False, can_subscribe=True

Questions:
  B1 Does a viewer token with can_publish=False actually join and receive video?
  B2 Is a publish attempt from that token refused by the server?
  B3 Does the viewer joining mid-stream change the frame dimensions the worker
     sees? docs/12 records transient 8x8 frames during simulcast adaptation, so
     a new subscriber forcing a layer change is a real risk to the guard.
  B4 Does an already-connected participant survive its token's expiry?
"""

import asyncio
import datetime as dt
import os
import time
from collections import Counter

import numpy as np
from livekit import api, rtc

URL = "ws://127.0.0.1:7880"
ROOM = "sg-b-viewer"
KEY = os.environ["VMA_LIVEKIT_API_KEY"]
SECRET = os.environ["VMA_LIVEKIT_API_SECRET"]

W, H, FPS = 1280, 720, 15
SECONDS_BEFORE_VIEWER = 6.0
SECONDS_AFTER_VIEWER = 6.0


def token(identity, *, can_publish, can_subscribe, ttl_s=300):
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(dt.timedelta(seconds=ttl_s))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=False,
            )
        )
        .to_jwt()
    )


def frame(seq):
    buf = np.empty((H, W, 4), dtype=np.uint8)
    buf[:, :, 0] = (seq * 17) % 255
    buf[:, :, 1] = np.arange(W, dtype=np.uint16) % 255
    buf[:, :, 2] = np.arange(H, dtype=np.uint16)[:, None] % 255
    buf[:, :, 3] = 255
    return rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, buf.tobytes())


class Sink:
    """Counts frames and their dimensions, tagged by phase."""

    def __init__(self, name):
        self.name = name
        self.dims = Counter()
        self.phase_dims = {"before": Counter(), "after": Counter()}
        self.phase = "before"
        self.first_at = None

    def record(self, w, h):
        if self.first_at is None:
            self.first_at = time.perf_counter()
        self.dims[(w, h)] += 1
        self.phase_dims[self.phase][(w, h)] += 1


async def consume(track, sink, stop):
    stream = rtc.VideoStream(track)
    async for event in stream:
        if stop.is_set():
            break
        sink.record(event.frame.width, event.frame.height)
    await stream.aclose()


async def join(identity, *, can_publish, can_subscribe, sink=None, stop=None, ttl_s=300):
    room = rtc.Room()
    tasks = []

    @room.on("track_subscribed")
    def on_sub(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO and sink is not None:
            tasks.append(asyncio.create_task(consume(track, sink, stop)))

    await room.connect(
        URL,
        token(identity, can_publish=can_publish, can_subscribe=can_subscribe, ttl_s=ttl_s),
        rtc.RoomOptions(auto_subscribe=True),
    )
    return room, tasks


async def main():
    results = {}
    stop = asyncio.Event()

    def record(key, value):
        """Print as we go: a killed pipe must not take the findings with it."""
        results[key] = value
        print(f"{key}: {value}", flush=True)

    worker_sink = Sink("worker")
    viewer_sink = Sink("viewer")

    # 1. gateway worker joins first, as it does today
    worker, worker_tasks = await join(
        "gateway-worker", can_publish=True, can_subscribe=True, sink=worker_sink, stop=stop
    )

    # 2. glasses publish
    glasses = rtc.Room()
    await glasses.connect(URL, token("glasses", can_publish=True, can_subscribe=True))
    source = rtc.VideoSource(W, H)
    await glasses.local_participant.publish_track(
        rtc.LocalVideoTrack.create_video_track("camera", source),
        rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=True,
            video_encoding=rtc.VideoEncoding(max_framerate=FPS, max_bitrate=1_500_000),
            degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
        ),
    )

    async def pump(seconds):
        n = int(seconds * FPS)
        started = time.perf_counter()
        for seq in range(n):
            source.capture_frame(frame(seq))
            await asyncio.sleep(max(0.0, started + (seq + 1) / FPS - time.perf_counter()))

    await pump(SECONDS_BEFORE_VIEWER)
    before_total = sum(worker_sink.dims.values())

    # 3. viewer joins mid-stream with a read-only grant
    worker_sink.phase = "after"
    try:
        viewer, viewer_tasks = await join(
            "console-viewer", can_publish=False, can_subscribe=True, sink=viewer_sink, stop=stop
        )
        record("B1_viewer_joined", True)
    except Exception as exc:  # noqa: BLE001
        record("B1_viewer_joined", f"FAILED: {exc}")
        viewer, viewer_tasks = None, []

    await pump(SECONDS_AFTER_VIEWER)

    # 4. can the viewer publish?
    if viewer is not None:
        try:
            vs = rtc.VideoSource(320, 180)
            await asyncio.wait_for(
                viewer.local_participant.publish_track(
                    rtc.LocalVideoTrack.create_video_track("sneaky", vs),
                    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
                ),
                timeout=10,
            )
            record("B2_viewer_publish_refused", "NO - publish SUCCEEDED")
        except Exception as exc:  # noqa: BLE001
            record("B2_viewer_publish_refused", f"yes ({type(exc).__name__})")

    stop.set()
    await asyncio.sleep(0.5)

    record("B3_worker_frames_before", before_total)
    record("B3_worker_frames_after", sum(worker_sink.phase_dims["after"].values()))
    record("B3_worker_dims_before", dict(worker_sink.phase_dims["before"]))
    record("B3_worker_dims_after", dict(worker_sink.phase_dims["after"]))
    record("B3_viewer_dims", dict(viewer_sink.dims))

    for room in (glasses, worker, viewer):
        if room is not None:
            await room.disconnect()

    # 5. token expiry on a live connection
    short = rtc.Room()
    try:
        await asyncio.wait_for(
            short.connect(URL, token("short-lived", can_publish=True, can_subscribe=True, ttl_s=5)),
            timeout=20,
        )
        await asyncio.sleep(12)
        record(
            "B4_alive_past_expiry",
            short.connection_state == rtc.ConnectionState.CONN_CONNECTED,
        )
        await short.disconnect()
    except Exception as exc:  # noqa: BLE001
        record("B4_alive_past_expiry", f"INCONCLUSIVE: {type(exc).__name__}")

    # 6. can an expired token start a new connection?
    expired = token("expired", can_publish=True, can_subscribe=True, ttl_s=1)
    await asyncio.sleep(3)
    rejoin = rtc.Room()
    try:
        await asyncio.wait_for(rejoin.connect(URL, expired), timeout=10)
        record("B4_expired_token_rejoin_refused", "NO - join SUCCEEDED")
        await rejoin.disconnect()
    except Exception as exc:  # noqa: BLE001
        record("B4_expired_token_rejoin_refused", f"yes ({type(exc).__name__})")

    print("\n=== SG-B results ===")
    for k, v in results.items():
        print(f"{k}: {v}")


asyncio.run(main())
