"""Print a relay stream's messages for debugging.

    uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/video

Prints headers and payload sizes only. Payload bytes, transcripts, and tokens
are never printed, per the logging rules in docs/07-Privacy-and-Security.md.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from visual_memory_media_contract.client import MediaClient
from visual_memory_media_contract.protocol import AudioChunk, RelayMessage, VideoFrame


def describe(message: RelayMessage) -> str:
    """One line per message: type, ids, and the fields that matter live."""
    if isinstance(message, VideoFrame):
        detail = (
            f"seq={message.sequence} {message.width}x{message.height} "
            f"{message.encoding} {message.payload_bytes}B "
            f"sha={message.sha256[:12]} dropped={message.dropped_since_previous}"
        )
    elif isinstance(message, AudioChunk):
        detail = (
            f"seq={message.sequence} pts={message.pts_samples} "
            f"samples={message.samples} {message.sample_rate}Hz "
            f"ch={message.channels} {message.payload_bytes}B"
        )
    else:
        dumped = message.model_dump(mode="json", exclude={"type", "protocol_version"})
        detail = " ".join(f"{key}={value!r}" for key, value in dumped.items())
    return f"{message.type:<16} {detail}"


async def tap(url: str, *, token: str | None, limit: int | None, reconnect: bool) -> int:
    seen = 0
    async with MediaClient(url, token=token, reconnect=reconnect) as client:
        async for message in client:
            print(describe(message), flush=True)
            seen += 1
            if limit is not None and seen >= limit:
                break
    return seen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="relay stream URL")
    parser.add_argument("--token", default=None, help="bearer token, if the relay requires one")
    parser.add_argument("--max", type=int, default=None, help="stop after N messages")
    parser.add_argument(
        "--reconnect",
        action="store_true",
        help="keep retrying instead of exiting when the stream ends",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        seen = asyncio.run(
            tap(args.url, token=args.token, limit=args.max, reconnect=args.reconnect)
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except OSError as exc:
        print(f"could not read {args.url}: {exc}", file=sys.stderr)
        return 1
    print(f"--- {seen} messages ---", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
