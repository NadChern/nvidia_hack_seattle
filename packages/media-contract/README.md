# visual-memory-media-contract

Wire models, framing codec, and the `MediaClient` for the Media Gateway relay.

Consumers (Vision, Speech) depend on this package instead of LiveKit. The
normative protocol definition lives in [docs/12](../../docs/12-Media-Relay-Contract.md).

## Development

```text
uv sync --frozen --all-groups
uv run pytest
```

## Checks

```text
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

## Consuming this package

Services depend on it by relative path so each service keeps its own lockfile:

```toml
[tool.uv.sources]
visual-memory-media-contract = { path = "../../packages/media-contract", editable = true }
```

`editable = true` is deliberate. A non-editable path dependency is **copied**
into the consumer's environment, and `uv sync --frozen` then audits it as
already satisfied — so edits here stay invisible to the consumer and it keeps
running stale contract code with no error. Editable installs resolve through
the source tree instead, and they work inside the container too because the
Dockerfile copies this package to the same relative offset.

**Changing a dependency here still requires every consumer to re-run
`uv lock`.** Source edits propagate automatically; dependency changes do not,
because the consumer's lockfile records this package's resolved dependency set
rather than a hash of its source.

Consumers today:

- `services/media-gateway`

## Using it

```python
from visual_memory_media_contract import MediaClient
from visual_memory_media_contract.protocol import EpochStarted, VideoFrame

async for message in MediaClient("ws://localhost:8080/v1/stream/video"):
    match message:
        case EpochStarted():
            tracker.reset(message.epoch_id)  # a rejoin invalidates track ids
        case VideoFrame():
            tracker.step(message.rgb)  # (H, W, 3) uint8
```

Decoding JPEG needs Pillow, which is an optional extra so the Speech Service
does not inherit an image library:

```text
uv add "visual-memory-media-contract[images]"
```

## Testing your consumer — no gateway, no LiveKit, no hardware

```python
from visual_memory_media_contract.testing import assert_matches_fixture, replay_server

async with replay_server("video_session_basic") as url:
    received = [m async for m in MediaClient(url, reconnect=False)]

assert_matches_fixture(received, "video_session_basic")
```

`flaky_replay_server(..., drop_after=5)` cuts the first connection mid-stream
to exercise reconnect handling.

Inspect a live stream with `uv run python -m visual_memory_media_contract.tap
<url>`; it prints headers and payload sizes, never payload bytes.

## Layout

- `protocol.py` — the message models. Frozen, discriminated on `type`, and
  tolerant of unknown fields so a pinned consumer survives a minor bump.
- `framing.py` — `encode_message` / `decode_message` for the `VMA1` binary
  frame, including payload length and digest verification.
- `client.py` — `MediaClient`, a reconnecting async iterator.
- `images.py` — payload decoding, reached through `VideoFrame.rgb` / `.rgba`.
- `testing.py` — replay servers and `assert_matches_fixture`.
- `fixtures/` — recorded streams. Regenerate with
  `uv run python scripts/build_fixtures.py`; output is deterministic, so a
  rebuild with no protocol change leaves `git diff` empty.
