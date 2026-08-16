"""Drive the whole stack once against a real LiveKit server.

The S01 spike proved ten things about LiveKit
(docs/spikes/livekit-media-gateway/RESULTS.md). Those proofs were a one-off
script; this module makes them a regression suite so a refactor that breaks one
of them fails a test rather than a demo.

Running is expensive -- three real join/publish/rejoin cycles paced in real
time -- so the exercise runs once and every assertion reads from the recorded
`Roundtrip`. That mirrors the spike, which also collected first and asserted at
the end.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import psutil
from visual_memory_media_contract.client import MediaClient
from visual_memory_media_contract.protocol import RelayMessage

from media_gateway.config import Settings
from media_gateway.main import create_app
from media_gateway.publisher.publish import Grant, VirtualGlasses, request_session
from media_gateway.publisher.sources import synthetic
from tests.serving import serve

#: Matches the spike: 320x180 at 10 FPS, sampled down to 2 FPS. Publishing
#: faster than the sample rate is what makes the "sampler sheds load"
#: assertion meaningful.
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 180
PUBLISH_FPS = 10.0
PUBLISH_SECONDS = 2.5
CYCLES = 3

TONE_HZ = 660.0
TONE_SECONDS = 0.5

#: How long `record_live_status` may wait for the guard to see a frame. Bounded
#: well inside PUBLISH_SECONDS: the point is to observe a *live* publisher, so
#: waiting past the end of the publish would trade one flake for another.
LIVE_STATUS_BUDGET_S = 1.5


@dataclass
class Cycle:
    """One join / publish / leave cycle, standing in for a Wi-Fi drop."""

    video_published: int = 0
    audio_published: int = 0
    return_audio_frames: int = 0


@dataclass
class Roundtrip:
    """Everything the exercise observed, for the assertions to read."""

    video: list[RelayMessage] = field(default_factory=list)
    audio: list[RelayMessage] = field(default_factory=list)
    cycles: list[Cycle] = field(default_factory=list)
    status_live: dict[str, Any] = field(default_factory=dict)
    status_final: dict[str, Any] = field(default_factory=dict)
    unauthenticated_session_status: int = 0
    invalid_livekit_token_rejected: bool = False
    nonlocal_connections: list[str] = field(default_factory=list)


# --- Small HTTP helpers ---------------------------------------------------
#
# urllib rather than httpx: `request_session` in the publisher already uses it,
# and a test that reaches the gateway the same way the publisher does is one
# fewer client library whose behaviour has to be reasoned about.


def _get_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        body: dict[str, Any] = json.load(response)
    return body


def _post_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=b"", headers={"authorization": f"Bearer {token}"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body: dict[str, Any] = json.load(response)
    return body


def _post_status(url: str, token: str | None = None) -> int:
    """POST and return the status code, treating an HTTP error as an answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps({"device_id": "integration"}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    if token:
        request.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def _delete(url: str, token: str) -> None:
    request = urllib.request.Request(
        url, headers={"authorization": f"Bearer {token}"}, method="DELETE"
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


# --- Privacy sweep --------------------------------------------------------


def nonlocal_established_connections(process: psutil.Process) -> list[str]:
    """Established sockets to something that is not this machine.

    Ported verbatim in intent from the spike's `inspect_server_network`. It is
    the mechanical enforcement of the rule in docs/07-Privacy-and-Security.md
    that no media leaves the trust boundary: if anyone puts a tunnel, a TURN
    relay, or a cloud API next to the media path, this goes red.
    """
    local_addresses = {"127.0.0.1", "::1"}
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family in {socket.AF_INET, socket.AF_INET6}:
                local_addresses.add(address.address.split("%")[0])

    found: set[str] = set()
    for connection in process.net_connections(kind="inet"):
        if connection.status != psutil.CONN_ESTABLISHED or not connection.raddr:
            continue
        remote_ip = connection.raddr.ip.split("%")[0]
        if remote_ip not in local_addresses:
            local = f"{connection.laddr.ip}:{connection.laddr.port}" if connection.laddr else ""
            found.add(f"{local}->{connection.raddr.ip}:{connection.raddr.port}")
    return sorted(found)


# --- The exercise ---------------------------------------------------------


class Tap:
    """A relay subscriber that records everything until told to stop.

    It holds the client rather than cancelling the consuming task: cancelling
    an async generator leaves its `finally` to generator finalization, so the
    WebSocket can outlive the test and uvicorn's graceful shutdown then waits
    on a connection nobody is reading.
    """

    def __init__(self, url: str, token: str, into: list[RelayMessage]) -> None:
        self._client = MediaClient(url, token=token, reconnect=False)
        self._into = into

    async def run(self) -> None:
        async for message in self._client:
            self._into.append(message)

    async def stop(self) -> None:
        await self._client.aclose()


def _tamper_jwt_signature(token: str) -> str:
    """Change one significant signature character while preserving JWT claims.

    The prior version inspected the *last* character but replaced the *first*.
    When a valid signature started with ``A`` and ended with anything else, the
    replacement was also ``A`` and the supposedly invalid token was unchanged.
    That made the real-LiveKit security assertion fail roughly once per 64 CI
    runs. Inspecting and replacing the same first character makes mutation
    unconditional and avoids base64url padding bits at the end of the segment.
    """
    segments = token.split(".")
    if len(segments) != 3 or any(not segment for segment in segments):
        raise ValueError("token is not a three-segment JWT")
    header, claims, signature = segments
    replacement = "A" if signature[0] != "A" else "B"
    return f"{header}.{claims}.{replacement}{signature[1:]}"


async def _invalid_livekit_token_is_rejected(livekit_url: str, token: str) -> bool:
    """A tampered signature must not open a room."""
    from livekit import rtc

    room = rtc.Room()
    tampered = _tamper_jwt_signature(token)
    try:
        await room.connect(livekit_url, tampered, options=rtc.RoomOptions(connect_timeout=5.0))
    except Exception:
        return True
    await room.disconnect()
    return False


async def _run_cycle(
    grant: Grant,
    base: str,
    token: str,
    while_publishing: Callable[[], Awaitable[None]] | None = None,
) -> Cycle:
    cycle = Cycle()
    async with VirtualGlasses(
        grant=grant, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, realtime=True
    ) as glasses:
        media = synthetic(
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=PUBLISH_FPS, seconds=PUBLISH_SECONDS
        )
        publishing = asyncio.create_task(glasses.publish(media))

        # Let the subscription to `assistant-tts` settle before speaking, or
        # the tone plays to nobody and the return-audio assertion is a lie.
        await asyncio.sleep(0.5)
        await asyncio.to_thread(
            _post_json,
            f"http://{base}/v1/return-audio/{grant.session_id}"
            f"/tone?hz={TONE_HZ}&seconds={TONE_SECONDS}",
            token,
        )
        # Anything that must observe a live publisher happens here, inside the
        # context manager. Once it exits the tracks are gone and the gateway
        # correctly reports nobody attached.
        if while_publishing is not None:
            await while_publishing()

        cycle.video_published, cycle.audio_published = await publishing
        # Return audio arrives asynchronously; give the last frames a moment.
        await asyncio.sleep(0.3)
        cycle.return_audio_frames = glasses.return_audio_frames

    return cycle


async def exercise(settings: Settings, token: str) -> Roundtrip:
    """Run the full stack once and record what happened."""
    result = Roundtrip()
    app = create_app(settings)

    async with serve(app) as base:
        result.unauthenticated_session_status = await asyncio.to_thread(
            _post_status, f"http://{base}/v1/sessions"
        )

        taps = (
            Tap(f"ws://{base}/v1/stream/video", token, result.video),
            Tap(f"ws://{base}/v1/stream/audio", token, result.audio),
        )
        readers = [asyncio.create_task(tap.run()) for tap in taps]
        # Subscribe before the first publish, so no epoch is missed.
        await asyncio.sleep(0.3)

        grant = await asyncio.to_thread(
            request_session, f"http://{base}", device_id="integration", token=token
        )
        result.invalid_livekit_token_rejected = await _invalid_livekit_token_is_rejected(
            grant.livekit_url, grant.token
        )

        async def record_live_status() -> None:
            """Snapshot `/v1/status` while a publisher is attached.

            Polled until the guard has actually seen a frame, not taken once.
            `/v1/status` reports `epochs.active()`, so by this last cycle the
            earlier epochs are gone and the histogram being asserted on belongs
            to an epoch created moments ago -- empty until the rejoined
            publisher's track delivers its first frame. Half a second is
            usually enough and on a loaded CI runner is not.
            `test_the_dimension_histogram_records_the_real_camera_size` failed
            exactly that way, with nothing wrong but timing, and passed on a
            rerun of the same commit.

            The last response is kept either way: every other assertion about
            the live status still needs one, and a timeout should fail on the
            claim being tested rather than on a missing fixture.
            """
            deadline = time.monotonic() + LIVE_STATUS_BUDGET_S
            while True:
                result.status_live = await asyncio.to_thread(
                    _get_json, f"http://{base}/v1/status", token
                )
                seen = any(
                    f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}" in epoch.get("guard", {}).get("dimensions", {})
                    for epoch in result.status_live["epochs"]
                )
                if seen or time.monotonic() >= deadline:
                    return
                await asyncio.sleep(0.1)

        try:
            for index in range(CYCLES):
                last = index == CYCLES - 1
                result.cycles.append(
                    await _run_cycle(
                        grant, base, token, while_publishing=record_live_status if last else None
                    )
                )
                # A rejoin keeps the identity and changes the track SIDs.
                await asyncio.sleep(0.5)
        finally:
            await asyncio.to_thread(_delete, f"http://{base}/v1/sessions/{grant.session_id}", token)

        # The sweep runs against this process: the gateway is what must not
        # talk to anything off-machine.
        result.nonlocal_connections = nonlocal_established_connections(psutil.Process())
        result.status_final = await asyncio.to_thread(_get_json, f"http://{base}/v1/status", token)

        for tap in taps:
            await tap.stop()
        await asyncio.wait(readers, timeout=5)

    return result


__all__ = [
    "CYCLES",
    "PUBLISH_FPS",
    "PUBLISH_SECONDS",
    "VIDEO_HEIGHT",
    "VIDEO_WIDTH",
    "Cycle",
    "Roundtrip",
    "exercise",
    "nonlocal_established_connections",
]
