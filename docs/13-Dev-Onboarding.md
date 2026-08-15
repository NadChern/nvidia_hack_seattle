# Dev Onboarding — agent reference

Precise reference for coding agents working in this repository. **Humans should read [`onboarding.html`](onboarding.html) instead** — it covers the same ground and explains the media concepts from zero.

This document assumes you have already read [`AGENTS.md`](../AGENTS.md) and the repository standards skill it names.

## Invariants

Violating any of these is a defect regardless of whether tests pass.

| Invariant | Where enforced |
|---|---|
| `epoch_id` is the LiveKit track SID; consumers reset all per-track state on `epoch_started` | `packages/media-contract`, consumer implementations |
| Only the Media Gateway subscribes to LiveKit; no other service holds a token or speaks WebRTC | `docs/12-Media-Relay-Contract.md` |
| Audio is never dropped; on overflow the subscriber's socket closes with `1011 audio_backpressure` | `relay/hub.py` |
| Video is latest-wins and never applies backpressure to ingest | `domain/sampling.py` |
| `raw_buffer_seconds` is `0` and only accepts `0` | `config.py::_no_raw_buffer_yet` |
| No cloud speech, signaling, TURN, telemetry, or model APIs on the default path | integration sweep for non-local established sockets |
| No secrets in source, logs, images, Compose files, or CI artifacts | `logging.py::RedactionFilter`, `.gitignore` |
| Logs carry no raw media, transcripts, tokens, or precise evidence paths | `logging.py`, `test_logging.py` |
| `--workers 1` for the gateway; a second worker doubles decode and splits relay state | `Dockerfile`, `compose.yaml` |
| Python is 3.11 — not 3.12 (numpy 2.5 dropped 3.11, lockfiles pin 2.4.6) | `.python-version`, `pyproject.toml` |

## Repository map

| Path | Contents |
|---|---|
| `services/media-gateway/` | the gateway service; owns its `pyproject.toml`, `uv.lock`, tests, Dockerfile |
| `packages/media-contract/` | wire models, `MediaClient`, fixtures, `tap` — the consumer-facing dependency |
| `tools/dev-livekit/` | local LiveKit: compose config, pinned-binary downloader, listener check |
| `deploy/livekit.yaml` | release LiveKit config (no `keys:` block; credentials via `LIVEKIT_KEYS`) |
| `compose.dev.yaml` | LiveKit alone, for local development |
| `compose.yaml` | GN100 release topology; release owner owns this file |
| `compose.gpu.yaml` | overlay on `compose.yaml`: real models, a GPU, the VLM verifier. Needs an nvidia container runtime and an Ollama on the host |
| `.agents/skills/` | repository standards; vendored LiveKit skill — read its `PROJECT-NOTE.md` before following it |

## Commands

All `uv` invocations must run **from the service or package directory**. From the repository root `uv run` resolves a different environment and fails with `ModuleNotFoundError: No module named 'media_gateway'`.

### Verify everything

```bash
cd services/media-gateway && ./scripts/verify_local.sh
```

Standards → contract package → gateway → LiveKit → listener exposure → live round trip asserting the S01 spike's ten findings. `--quick` skips the live portion. `--docker` adds the image build and the emulated ARM64 packaging gate. `scripts/verify_local.ps1` is the Windows twin.

### Per-target checks

```bash
python3 .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
```

```bash
cd services/media-gateway && uv sync --frozen --all-groups && uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest
```

```bash
cd packages/media-contract && uv sync --frozen --all-groups && uv run ruff check . && uv run pyright && uv run pytest
```

`pytest` deselects the integration suite by default (`-m 'not livekit'`), so it passes with no LiveKit server. The integration suite runs only when `VMA_TEST_LIVEKIT_URL` is set.

### Bring the stack up

```bash
cd services/media-gateway && ./scripts/dev_up.sh
```

Generates credentials on first run into a gitignored repo-root `.env` and reuses them afterwards, starts LiveKit if it is not already up, runs the listener check, launches the gateway, waits for readiness, and prints the publisher URL. Ctrl-C stops what it started.

Flags: `--scripted` (no LiveKit at all), `--strict-guard`, `--port N`, `--keep-livekit`.

Credentials must match between the LiveKit container and the gateway; a mismatch presents as a connection failure rather than a configuration error, which is why this is a script.

### Run the pieces manually

```bash
cd services/media-gateway && VMA_MEDIA_SOURCE=scripted uv run uvicorn media_gateway.main:app --port 8080
```

```bash
docker compose -f compose.dev.yaml up -d livekit
```

```bash
cd services/media-gateway && VMA_MEDIA_SOURCE=livekit VMA_DIMENSION_GUARD_MODE=sustained uv run uvicorn media_gateway.main:app --port 8080
```

`VMA_LIVEKIT_API_KEY` and `VMA_LIVEKIT_API_SECRET` must be set in the same shell for both. The gateway rejects LiveKit's well-known `devkey`/`secret`, rejects the spike's published pair, and requires a secret of at least 32 characters.

### Inspect a running gateway

```bash
cd services/media-gateway && uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/video
```

```bash
curl -fsS localhost:8080/v1/status | python3 -m json.tool
```

`epochs[].guard.dimensions` is a histogram of every frame size the track produced, tallied **before** the admit decision. It is the field that distinguishes "no publisher connected" from "publisher sending a size the guard rejects"; both otherwise present as a pipeline receiving nothing.

## Consuming the relay

Depend on `packages/media-contract`. Do not parse the wire format independently.

```python
from visual_memory_media_contract import MediaClient
from visual_memory_media_contract.protocol import EpochEnded, EpochStarted, VideoFrame

async for message in MediaClient("ws://localhost:8080/v1/stream/video"):
    match message:
        case EpochStarted():
            tracker.reset(message.epoch_id)
        case VideoFrame():
            tracker.step(message.rgb, message.captured_at)
        case EpochEnded():
            tracker.finalize(message.reason)
```

Required behaviour:

- **Reset on `epoch_started`.** Discard tracker state, identity assignments, and any partially accumulated temporal event for that stream. `sequence` restarts at 0 per epoch.
- **Treat a reconnect as at least as strong a reset.** `MediaClient` re-sends `stream_hello` plus a synthetic `epoch_started` for each still-active epoch after reconnecting, so correct `epoch_started` handling covers both.
- **Ignore unknown header fields.** Additive fields are a minor version bump and must not break a pinned consumer.
- **Read `pts_samples` for audio continuity**, not message counts. A gap is detectable by arithmetic; counting messages will miss it.
- Video and audio are separate LiveKit tracks with **different** `epoch_id` values, correlated by `session_id`.

### Testing a consumer with no gateway

```python
from visual_memory_media_contract.testing import assert_matches_fixture, replay_server

async with replay_server("video_session_basic") as url:
    received = [message async for message in MediaClient(url, reconnect=False)]

assert_matches_fixture(received, "video_session_basic")
```

| Fixture | Covers |
|---|---|
| `video_session_basic` | session, two epochs across a rejoin under an unchanged participant identity, a reported frame drop, a lifecycle signal, a keepalive, clean shutdown |
| `audio_session_basic` | 3 s of 48 kHz mono with a deliberate 500 ms gap where `pts_samples` jumps while `sequence` stays contiguous |

`flaky_replay_server` drops the first connection mid-stream to exercise reconnect handling.

Per [Team Split](05-Team-Split.md), an interface is complete only when the provider fixture passes inside the consumer's harness. Both sides assert against the same files.

## Creating a consuming service

Services are generated, never hand-written; the generator produces the layout, Dockerfile, config, health endpoints, and lockfile `validate_repo.py` requires.

```bash
python3 .agents/skills/visual-memory-repo-standards/scripts/new_service.py vision --kind worker --owner "Person 1"
```

`--kind` is `application`, `worker`, or `inference`. Vision and Speech are workers. `inference` carries extra deployment obligations and is only for platform-specific model packages.

Then add the media contract to the generated `pyproject.toml` — **both** stanzas:

```toml
[project]
dependencies = [
    "visual-memory-media-contract",
]

[tool.uv.sources]
visual-memory-media-contract = { path = "../../packages/media-contract", editable = true }
```

**`editable = true` is required, not stylistic.** Without it uv copies the package into the environment; `uv sync --frozen` then audits it as satisfied, so the service silently runs a stale snapshot of the contract and every shared-package fix passes it by. It presents as the gateway sending wrong data. This cost real debugging time on the gateway itself.

```bash
uv lock && uv sync --frozen --all-groups
```

Verify the mapping resolved to the repository rather than a copy — this must print a path under `packages/media-contract`, not under `.venv/lib`:

```bash
uv run python -c "import visual_memory_media_contract as m; print(m.__file__)"
```

The lockfile records the local package's resolved dependency set, not its source, so adding a dependency to `media-contract` means every consuming service must re-run `uv lock`.

## Configuration

Environment variables use the `VMA_` prefix. Full set in `services/media-gateway/src/media_gateway/config.py`.

| Variable | Note |
|---|---|
| `VMA_MEDIA_SOURCE` | `livekit` or `scripted`; `scripted` needs no server |
| `VMA_DIMENSION_GUARD_MODE` | `strict` (configured WxH), `sustained` (latch a size that holds for ~24 frames; use this for a real publisher), or `first_frame_wins` (latch the very first frame — wrong for anything that ramps) |
| `VMA_LIVEKIT_URL` | how **this process** reaches LiveKit |
| `VMA_LIVEKIT_PUBLIC_URL` | how a **device** reaches LiveKit; differs under Compose. Defaults to `VMA_LIVEKIT_URL` |
| `VMA_DEVICE_ID_ALLOWLIST` | comma-separated, **not** JSON; a JSON value is refused rather than misread |
| `VMA_ENVIRONMENT` | `dev` \| `ci` \| `deploy`. `deploy` requires `internal_api_token` and a non-empty allowlist, and serves no OpenAPI schema, docs, or `/dev/*` routes |

## Failure modes

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'media_gateway'` | `uv run` from the repository root. `cd services/media-gateway` first |
| Signaling connects, media never flows (WSL2) | Default WSL2 NAT forwards localhost TCP only; ICE over UDP 7882 fails. `livekit.dev.yaml` sets `force_tcp: true`; alternative is `networkingMode=mirrored` in `.wslconfig` plus `wsl --shutdown` |
| Frames arrive 320x180 from a 720p source | Simulcast — the SFU serves the lowest layer by default. Publish a single layer (`simulcast=False`). Subscriber-side quality requests are advisory and may be ignored |
| Pipeline receives nothing, no error | Check `guard.dimensions` in `/v1/status`. `admitted=0` with a populated histogram means the guard is rejecting every frame; use `sustained`. If `expected` is far smaller than the sizes in the histogram, the guard latched a rung of the encoder's ramp-up — that is `first_frame_wins`, and `sustained` is the fix |
| Tests fail only on one machine | Ambient `VMA_*` variables in the shell leaking into tests. An autouse fixture clears them; suspect this first for any similar case |
| `Bind for 127.0.0.1:7880 failed` | `compose.dev.yaml` and `compose.yaml` both bind 7880 and cannot run together. `docker compose -f <file> down` needs the same env vars exported for interpolation, or it fails silently |
| Container exits with a uv cache error | The runtime image runs `.venv/bin/uvicorn` directly, not `uv run`, because uv needs a writable cache and the container root filesystem is read-only |

## Open items

- **Lifecycle signals (`track_lost` / `session_ended`) are not emitted to the Memory Service.** The envelope, scope semantics, and three open questions are in [Data Contract § Lifecycle signals](06-Data-Contract.md#lifecycle-signals), pending the Memory Service owner's sign-off. The signals exist in band on the relay only. Do not build a Memory-side consumer against that shape until it is accepted.
- A device publishing from off-box over a real LAN is unverified; that is spike adoption gate 1 and needs hardware.

## Related

- [Media Relay Contract](12-Media-Relay-Contract.md) — the wire format in full
- [Data Contract](06-Data-Contract.md) — the canonical observation format, which the relay is *not*
- [Privacy and Security](07-Privacy-and-Security.md) — the trust boundary
- [Team Split](05-Team-Split.md) — ownership and integration milestones
- [Engineering Standards](11-Engineering-Standards.md) — the mandatory service stack
