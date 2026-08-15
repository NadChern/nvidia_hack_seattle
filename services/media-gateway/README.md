# media-gateway

Repository-standard `worker` service for the Visual Memory Assistant.

Owns the WebRTC media boundary: it holds the only inference LiveKit subscription,
samples and dimension-guards decoded video, and relays media to the Vision and Speech
services over a local WebSocket. A read-only operator viewer is the sole subscription
carve-out. Deployed as the `media-worker` Compose service.

## Running it without LiveKit or hardware

Every command below must be run **from this directory**. `uv run` resolves the
project from the working directory, so running from the repository root picks
up whatever ambient virtualenv you happen to have and fails with
`ModuleNotFoundError: No module named 'media_gateway'`.

Terminal 1 — the gateway, driven by a scripted media source:

```text
VMA_MEDIA_SOURCE=scripted uv run uvicorn media_gateway.main:app --port 8080
```

Terminal 2 — watch the relay. `visual_memory_media_contract` is a dependency of
this service, so the tap runs from this directory too:

```text
uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/video --max 10
uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/audio --max 10
```

Expect `stream_hello`, an `epoch_started`, frames whose `sequence` restarts at
zero after each rejoin, then `epoch_ended`. A non-zero `dropped` count is the
latest-wins slot shedding stale frames and reporting it, not an error.

The default pace is deliberately slow. For a livelier stream:

```text
VMA_MEDIA_SOURCE=scripted VMA_SCRIPTED_FRAME_INTERVAL_S=0.05 VMA_SAMPLE_FPS=10 \
  uv run uvicorn media_gateway.main:app --port 8080
```

Consumers should use `MediaClient` rather than the tap; see
[packages/media-contract](../../packages/media-contract/README.md), which also
ships recorded fixtures so a consumer can be tested with no gateway running at
all.

## Running it with your laptop camera

Publishes real camera and microphone through LiveKit into the gateway, standing
in for the glasses. Three steps.

**1. Start LiveKit.** From the repository root:

```text
export VMA_LIVEKIT_API_KEY=vma-dev
export VMA_LIVEKIT_API_SECRET=$(openssl rand -hex 24)
docker compose -f compose.dev.yaml up -d livekit
python3 tools/dev-livekit/check_listeners.py
```

**2. Start the gateway** from this directory, with the same two variables
exported:

```text
VMA_MEDIA_SOURCE=livekit VMA_DIMENSION_GUARD_MODE=sustained \
  uv run uvicorn media_gateway.main:app --port 8080
```

`sustained` matters. The guard defaults to a strict 320x180 and a real
webcam is typically 1280x720, so under `strict` every frame is correctly
rejected and nothing reaches the relay.

**3. Start the console** and press Publish at <http://localhost:5173>:

```text
cd ../../apps/console && npm install && npm run dev
```

Or skip steps 1-3 entirely with `./scripts/dev_stack.sh` from the repository
root, which starts LiveKit, this gateway and the console together.

On WSL2 open the page in the **Windows** browser, not inside WSL2, because
that is where the camera is. `http://localhost` counts as a secure context, so
`getUserMedia` works without TLS; any other hostname would need HTTPS.

No browser at all is another option: `virtual-glasses` publishes synthetic
media or a file into the same room, which is what the integration suite uses.

Watch frames arrive with the tap:

```text
uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/video --max 20
```

Press Rejoin on the page and watch the track SIDs change while the participant
identity does not. That is the media-epoch boundary the vision pipeline resets
on.

If media never flows on WSL2, it is the UDP path: default NAT networking
forwards localhost for TCP only. `tools/dev-livekit/livekit.dev.yaml` ships
with `force_tcp: true` for exactly this reason. The alternative is
`networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`.

No Docker, or need TLS for a second device?
[`tools/dev-livekit/README.md`](../../tools/dev-livekit/README.md) covers the
pinned-binary path, the WSL2 networking options, and why tunnels are not an
option for media.

## Session control API

```text
POST /v1/pairing                          issue an internal-only, single-use code
POST /v1/pairing/claim                    exchange that code for a device credential
POST /v1/sessions                         create a publisher session and token
POST /v1/sessions/{session_id}/token      refresh that publisher token in place
POST /v1/sessions/{session_id}/viewer     mint a subscribe-only operator token
POST /v1/device/{session_id}/events       accept Agent transcript/reply events
WS   /v1/device/{session_id}/events       stream events to device/operator HUDs
POST /v1/device/{session_id}/manual-trigger arm one wake-prefix-free where turn
GET  /v1/sessions                         list sessions for the operator picker
DELETE /v1/sessions/{session_id}          end a session
```

Operator and Agent surfaces use the internal bearer. A device credential can create a
session only for its embedded `device_id`, refresh/delete only that device's session, open
only that session's event socket, and arm its consume-once manual trigger. It cannot list sessions, mint viewers, publish events,
or read status/relays. The viewer cannot publish media or data. Its client requests low
when simulcast is present, but SG-C proved subscription requests insufficient; production
glasses therefore publish one 720p layer with simulcast disabled.

## Development

```text
uv sync --frozen --all-groups
uv run uvicorn media_gateway.main:app --reload
```

## Checks

Everything at once, including a live round trip through a real LiveKit server:

```text
./scripts/verify_local.sh
```

`--quick` skips the live round trip; `--docker` adds the image build and the
emulated ARM64 packaging gate. `scripts/verify_local.ps1` is the Windows twin.
The script starts a LiveKit server if none is running and stops it again on the
way out.

The individual checks, if you want them one at a time:

```text
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

`pytest` deselects the integration suite by default (`-m 'not livekit'`), so
the standards CI loop stays green with no LiveKit server.

### Integration suite

`tests/integration/` re-runs the [S01 spike](../../docs/spikes/livekit-media-gateway/RESULTS.md)'s
ten assertions against the code that actually ships — three real join, publish
and rejoin cycles through a LiveKit server, including the privacy sweep that
fails if anything on the media path talks to a host that is not this machine.

```text
docker compose -f ../../compose.dev.yaml up -d livekit
VMA_TEST_LIVEKIT_URL=ws://127.0.0.1:7880 uv run pytest tests/integration -m livekit
```

Credentials come from `VMA_LIVEKIT_API_KEY` / `VMA_LIVEKIT_API_SECRET`, or from
`VMA_TEST_*` overrides if the server under test uses a different pair. Each run
takes about 25 seconds and uses its own room prefix, so a leftover room from a
crashed run cannot affect the next one. CI runs it in
`.github/workflows/media-gateway-integration.yml`, on changes to this service
or the shared contract only.

## Container build — context is the repository root

This service depends on `packages/media-contract` by relative path, so the
build context must be the repository root, not this directory:

```text
docker build -f services/media-gateway/Dockerfile -t vma/media-gateway:dev .
```

Building from inside this directory fails: the `COPY packages/media-contract`
layer has nothing to copy. Compose uses `build.context: .` with
`build.dockerfile: services/media-gateway/Dockerfile` for the same reason.

The ARM64 packaging gate (emulated; proves layering only, never CUDA or GN100
runtime behaviour):

```text
docker buildx build --platform linux/arm64 -f services/media-gateway/Dockerfile .
```

## Boundary decisions

- **The media plane stays on LiveKit/WebRTC.** Ingress, codecs, jitter
  buffering, ICE, and return audio are all LiveKit's. The FastAPI WebSocket
  relay carries only already-decoded, dimension-guarded, sampled frames between
  co-located containers, so it does not "replace the media plane" in the sense
  `docs/11-Engineering-Standards.md` prohibits. The alternative — every consumer
  holding its own LiveKit subscription — would multiply decode cost and token
  surface and duplicate the sampler in every service.
- **`--workers 1` is load-bearing.** A second uvicorn worker would open a second
  LiveKit subscription on the same room, doubling decode and splitting relay
  state. Never raise it.
- **`av` (PyAV) is in a non-default `publisher` dependency group** so
  `uv sync --no-dev` keeps it out of the runtime image. Only the
  `virtual-glasses` test publisher imports it; it is not on the GN100 path.
- **numpy is capped at `<2.5`** because numpy 2.5 dropped Python 3.11, which the
  repository standard mandates.
- **`packages/media-contract` is an editable path dependency.** Non-editable
  copies the package into the environment and `uv sync --frozen` then audits it
  as satisfied, so edits to the shared contract would silently not reach this
  service. Editable resolves through the source tree, in the container as well.

## Related

- Wire protocol: `docs/12-Media-Relay-Contract.md` (once published)
- Shared client: `packages/media-contract`
- Origin spike and its results: `docs/spikes/livekit-media-gateway/`
