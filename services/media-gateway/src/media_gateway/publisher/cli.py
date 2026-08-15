"""The `virtual-glasses` test publisher.

Stands in for the glasses so the pipeline can be exercised without hardware:

    virtual-glasses --synthetic --seconds 20
    virtual-glasses --file clips/kitchen-keys.mp4 --realtime

Each run requests its own session token from the gateway, so it exercises the
token endpoint from the client side rather than reaching around it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from media_gateway.publisher.publish import (
    VirtualGlasses,
    delete_session,
    request_session,
)
from media_gateway.publisher.sources import (
    Media,
    PublisherDependencyError,
    from_file,
    synthetic,
)

logger = logging.getLogger("virtual-glasses")

DEFAULT_GATEWAY = "http://127.0.0.1:8080"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="virtual-glasses",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="gateway base URL")
    parser.add_argument("--device-id", default="virtual-glasses")
    parser.add_argument("--token", default=None, help="internal API bearer token")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="publish a test pattern")
    source.add_argument("--file", type=Path, help="publish a prerecorded media file")

    parser.add_argument("--seconds", type=float, default=20.0, help="synthetic duration")
    parser.add_argument("--fps", type=float, default=15.0, help="synthetic frame rate")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace to presentation time instead of publishing as fast as possible",
    )
    parser.add_argument(
        "--reconnect-cycles",
        type=int,
        default=1,
        help="publish this many times, leaving and rejoining between each",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def build_media(args: argparse.Namespace) -> Iterator[Media]:
    if args.file is not None:
        return from_file(args.file, width=args.width, height=args.height)
    return synthetic(
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
    )


async def run(args: argparse.Namespace) -> int:
    grant = request_session(args.gateway, device_id=args.device_id, token=args.token)
    logger.info("session %s in room %s", grant.session_id, grant.room)

    try:
        for cycle in range(1, max(1, args.reconnect_cycles) + 1):
            async with VirtualGlasses(
                grant=grant,
                width=args.width,
                height=args.height,
                realtime=args.realtime or args.file is not None,
            ) as glasses:
                videos, audios = await glasses.publish(build_media(args))
                logger.info(
                    "cycle %d: published %d video and %d audio frames, received %d back",
                    cycle,
                    videos,
                    audios,
                    glasses.return_audio_frames,
                )
            if cycle < args.reconnect_cycles:
                # A rejoin keeps the identity and changes the track SIDs, which
                # is what the gateway treats as a new media epoch.
                logger.info("rejoining; track SIDs will change, identity will not")
                await asyncio.sleep(0.5)
    finally:
        delete_session(args.gateway, grant.session_id, token=args.token)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except (PublisherDependencyError, FileNotFoundError, RuntimeError) as exc:
        print(f"virtual-glasses: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
