"""Recorded relay streams and the builders that produce them.

A fixture is a `.bin` file holding concatenated length-prefixed wire frames.
Consumers replay one to exercise their pipeline with no gateway and no LiveKit
running; the gateway asserts its own output against the same files, which is
how the Team Split rule -- an interface is complete only when the provider
fixture passes in the consumer's harness -- is mechanized.
"""

from __future__ import annotations

import datetime as dt
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path

from visual_memory_media_contract.framing import decode_message, encode_message
from visual_memory_media_contract.protocol import RelayMessage

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

_LENGTH = struct.Struct(">I")

#: Fixed instant every fixture is built from, so files are byte-stable.
FIXTURE_EPOCH = dt.datetime(2026, 7, 30, 18, 0, 0, tzinfo=dt.UTC)


def at(offset_ms: int) -> dt.datetime:
    """Return the fixture clock advanced by `offset_ms`."""
    return FIXTURE_EPOCH + dt.timedelta(milliseconds=offset_ms)


def write_fixture(path: Path, frames: Sequence[bytes]) -> None:
    """Write wire frames to `path`, each prefixed with its length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for frame in frames:
            handle.write(_LENGTH.pack(len(frame)))
            handle.write(frame)


def read_fixture(path: Path) -> list[bytes]:
    """Read the wire frames recorded in `path`."""
    data = path.read_bytes()
    frames: list[bytes] = []
    offset = 0
    while offset < len(data):
        (length,) = _LENGTH.unpack_from(data, offset)
        offset += _LENGTH.size
        frames.append(data[offset : offset + length])
        offset += length
    return frames


def load_fixture(name: str) -> list[bytes]:
    """Read a named fixture from the packaged fixtures directory."""
    return read_fixture(FIXTURES_DIR / f"{name}.bin")


def iter_messages(name: str) -> Iterator[RelayMessage]:
    """Decode a named fixture into messages."""
    for frame in load_fixture(name):
        yield decode_message(frame)


def build_frames(messages: Sequence[tuple[RelayMessage, bytes]]) -> list[bytes]:
    """Encode `(message, payload)` pairs into wire frames."""
    return [encode_message(message, payload) for message, payload in messages]


__all__ = [
    "FIXTURES_DIR",
    "FIXTURE_EPOCH",
    "at",
    "build_frames",
    "iter_messages",
    "load_fixture",
    "read_fixture",
    "write_fixture",
]
