"""SG-C: production gateway high layer plus a low-layer operator viewer.

Requires a real LiveKit server and the production media-gateway process. The
Gateway must use a strict 1280x720 guard so its own admitted/rejected counters
make a layer collapse observable without adding spike instrumentation to it.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import Counter
from typing import Any

import httpx
import numpy as np
from livekit import rtc

GATEWAY = os.getenv("VMA_GATEWAY_URL", "http://127.0.0.1:8080")
SIMULCAST = os.getenv("SG_C_SIMULCAST", "true").casefold() == "true"
W, H, FPS = 1280, 720, 15
PHASE_SECONDS = 8.0


def headers() -> dict[str, str]:
    token = os.getenv("VMA_INTERNAL_API_TOKEN")
    return {"authorization": f"Bearer {token}"} if token else {}


def frame(sequence: int) -> rtc.VideoFrame:
    pixels = np.empty((H, W, 4), dtype=np.uint8)
    pixels[:, :, 0] = (sequence * 17) % 255
    pixels[:, :, 1] = np.arange(W, dtype=np.uint16) % 255
    pixels[:, :, 2] = np.arange(H, dtype=np.uint16)[:, None] % 255
    pixels[:, :, 3] = 255
    return rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, pixels.tobytes())


def video_metrics(status: dict[str, Any]) -> dict[str, int]:
    metrics = status["metrics"]["video"]
    return {
        "received": int(metrics["received"]),
        "admitted": int(metrics["admitted"]),
        "rejected_dimensions": int(metrics["rejected_dimensions"]),
    }


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


async def main() -> None:
    async with httpx.AsyncClient(base_url=GATEWAY, headers=headers(), timeout=10.0) as http:
        created_response = await http.post("/v1/sessions", json={"device_id": "sg-c-glasses"})
        created_response.raise_for_status()
        created = created_response.json()
        session_id = created["session_id"]

        publisher = rtc.Room()
        viewer: rtc.Room | None = None
        viewer_tasks: list[asyncio.Task[None]] = []
        viewer_dimensions: Counter[tuple[int, int]] = Counter()
        stop = asyncio.Event()

        async def consume_viewer(track: rtc.Track) -> None:
            stream = rtc.VideoStream(track)
            try:
                async for event in stream:
                    if stop.is_set():
                        break
                    viewer_dimensions[(event.frame.width, event.frame.height)] += 1
            finally:
                await stream.aclose()

        try:
            await publisher.connect(created["livekit_url"], created["token"])
            source = rtc.VideoSource(W, H)
            await publisher.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("camera", source),
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_CAMERA,
                    simulcast=SIMULCAST,
                    video_encoding=rtc.VideoEncoding(
                        max_framerate=FPS,
                        max_bitrate=1_500_000,
                    ),
                    degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
                ),
            )

            sequence = 0

            async def pump(seconds: float) -> None:
                nonlocal sequence
                count = int(seconds * FPS)
                started = time.perf_counter()
                for index in range(count):
                    source.capture_frame(frame(sequence))
                    sequence += 1
                    deadline = started + (index + 1) / FPS
                    await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

            baseline = video_metrics((await http.get("/v1/status")).json())
            await pump(PHASE_SECONDS)
            before_viewer = video_metrics((await http.get("/v1/status")).json())

            viewer_grant = (await http.post(f"/v1/sessions/{session_id}/viewer")).json()
            viewer = rtc.Room()

            @viewer.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                _participant: rtc.RemoteParticipant,
            ) -> None:
                if track.kind != rtc.TrackKind.KIND_VIDEO:
                    return
                if publication.simulcasted:
                    publication.set_video_quality(rtc.VideoQuality.VIDEO_QUALITY_LOW)
                viewer_tasks.append(asyncio.create_task(consume_viewer(track)))

            await viewer.connect(
                viewer_grant["livekit_url"],
                viewer_grant["token"],
                options=rtc.RoomOptions(auto_subscribe=True),
            )
            await pump(PHASE_SECONDS)
            after_viewer = video_metrics((await http.get("/v1/status")).json())

            before = delta(before_viewer, baseline)
            after = delta(after_viewer, before_viewer)
            admitted_ratio = after["admitted"] / max(1, after["received"])

            print("SG-C subscription pinning")
            print(f"publisher simulcast: {SIMULCAST}")
            print(f"gateway config: {(await http.get('/v1/status')).json()['config']}")
            print(f"gateway before viewer: {before}")
            print(f"gateway after viewer:  {after}")
            print(f"gateway admitted ratio after viewer: {admitted_ratio:.3f}")
            print(f"viewer dimensions: {dict(viewer_dimensions)}")
            print(
                "acceptance: "
                + (
                    "PASS"
                    if admitted_ratio >= 0.8 and after["admitted"] > 0
                    else "FAIL/INCONCLUSIVE"
                )
            )
        finally:
            stop.set()
            if viewer is not None:
                await viewer.disconnect()
            for task in viewer_tasks:
                task.cancel()
            if viewer_tasks:
                await asyncio.gather(*viewer_tasks, return_exceptions=True)
            await publisher.disconnect()
            await http.delete(f"/v1/sessions/{session_id}")


asyncio.run(main())
