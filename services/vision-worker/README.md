# Vision Service (`vision-worker`)

Consumes sampled glasses video, asks Cosmos 3 Nano what happened to registered personal objects in
a short frame window, verifies identity with C-RADIOv4-H, and writes only policy-approved events to
Application Memory.

The current production pipeline is deliberately **not** a per-frame detector/tracker:

```text
Media Gateway frames
  → bounded EvidenceRing
  → Cosmos window reasoning: {label, final-frame box, action, location}
  → box crop
  → C-RADIO personal-gallery match
  → placed-only promotion and cooldown gates
  → canonical Observation + evidence in Application Memory
```

Unregistered clutter is never written. A model-proposed location is not trusted unless the event
matches one durable personal-object identity and passes the promotion policy.

## Current behavior

### Window reasoning

Cosmos receives at most a few frames from a roughly six-second window. It reports only configured,
registered labels visible in the final frame and classifies each as:

- `placed`
- `picked_up`
- `carried`
- `nothing_happened`
- `unknown`

`nothing_happened` and `unknown` are honest non-events and never become observations. Calls run off
the ingest loop with one analysis in flight, so slow inference cannot block frame intake.

### Placement promotion policy

`VMA_PROMOTE_MOTION_EVENTS=false` is the default. Cosmos may still report `picked_up` or `carried`,
but they appear only as `suppressed_by_policy` diagnostics. Only identity-matched `placed` events
are written to Memory.

This is intentional for the current sparse glasses cadence: one false pickup would invalidate a
correct confirmed location. Enable motion promotion only after motion classification is validated
at the deployed frame rate.

### Personal-object identity

For each promotable event, the service crops Cosmos's final-frame box, embeds it with
`nvidia/C-RADIOv4-H`, and compares it with the durable gallery from Application Memory. The write
gate requires the configured cosine threshold. Outcomes are visible at `GET /v1/events`:

- `written`: identity and promotion gates passed;
- `skipped_no_identity`: no gallery object matched strongly enough;
- `suppressed_by_policy`: a motion event was intentionally withheld; or
- `deduped`: the same object/action was written within its cooldown.

The event feed is a bounded operational ring, not canonical history. Application Memory remains the
authority after a `written` result.

### Registration

Registration creates a durable object in Memory, captures a bounded frame window, and localizes the
chosen label with a strict single-frame Cosmos prompt. Every geometrically usable crop then passes a
second semantic localization, and the tighter second box—not the broad proposal—is what gets
embedded. A separate contrastive reference-quality check rejects sharp but wrong crops; for keys,
a ring, fob, or tag without a clearly visible metal key blade is explicitly invalid. C-RADIO embeds
the survivors, removes near-duplicates, and stores two to four diverse reference views. Fewer than
the configured minimum views fails explicitly, deletes the empty object, and stores no weak gallery.
The live identity path applies the same two-stage crop transform before gallery matching.

The registration allowlist is `VMA_DETECTION_LABELS`. `/v1/status` exposes it as
`config.registration_labels` so the Console uses the same server-enforced list rather than a
hard-coded menu.

## APIs

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Relay, reasoner, embedder, and owned dependency readiness |
| `GET /v1/status` | Reasoner cadence, registration allowlist, gallery state, queue health, and pipeline counters |
| `GET /v1/events` | Recent Cosmos/identity outcomes for the Console demo feed |
| `POST /v1/objects` · `GET /v1/objects` | Create/list personal objects through Memory's registry |
| `POST /v1/objects/{id}/capture` | Start bounded registration capture |
| `GET /v1/objects/{id}/status` | Poll capture, extraction, quality, and selected-view progress |

There is no public video-upload or Memory-query API here. Ordinary frames arrive only through the
Media Gateway relay; confirmed observations leave through `MemoryEmitter`.

## Important configuration

All settings use the `VMA_` prefix.

| Setting | Default | Meaning |
|---|---:|---|
| `GATEWAY_VIDEO_URL` | `ws://127.0.0.1:8080/v1/stream/video` | Sampled relay input |
| `MEMORY_BASE_URL` | `http://127.0.0.1:8081` | Registry and observation destination |
| `REASON_KIND` | `cosmos` | `cosmos` or deterministic `fixture` |
| `REASON_BASE_URL` | `http://127.0.0.1:8001/v1` | Cosmos OpenAI-compatible endpoint |
| `REASON_MODEL` | `nvidia/Cosmos3-Nano` | Multimodal reasoner |
| `REASON_WINDOW_SECONDS` | `6` | Evidence window duration |
| `REASON_INTERVAL_SECONDS` | `7` | Minimum analysis cadence |
| `REASON_MAX_FRAMES` | `4` | Frames sent per Cosmos call |
| `EVENT_COOLDOWN_SECONDS` | `20` | Same-object/action deduplication window |
| `PROMOTE_MOTION_EVENTS` | `false` | Whether pickup/carried may write Memory |
| `IDENTITY_KIND` | `none` | Use `radio` on the GN100 real-model profile |
| `IDENTITY_MIN_COSINE` | configurable | Personal-gallery write threshold |
| `DETECTION_LABELS` | empty | Registration allowlist, comma-separated in environment configuration |
| `REGISTRATION_CAPTURE_SECONDS` | `6` | Bounded enrollment capture |
| `REGISTRATION_TARGET_VIEWS` | `4` | Desired durable reference views |
| `REGISTRATION_MIN_VIEWS` | `2` | Minimum successful gallery size |

The GN100 deployment overrides identity and thresholds through its protected runtime environment.
Do not copy remote `.stack.env` values or secrets into source.

## Console demonstration

The Console shows raw glasses POV rather than nonexistent per-frame boxes. Its Vision tab polls
`/v1/status` and `/v1/events` to show four observable receipts:

```text
registered gallery → Cosmos event → personal identity → durable Memory write
```

For a live placement, leave the registered object at rest through a complete reasoner window. Wait
longer than `EVENT_COOLDOWN_SECONDS` before demonstrating a second placement of the same object.

## Run locally

Run `uv` inside this service directory:

```bash
cd services/vision-worker
uv sync --frozen --all-groups
VMA_REASON_KIND=fixture \
VMA_GATEWAY_VIDEO_URL=ws://127.0.0.1:8080/v1/stream/video \
VMA_MEMORY_BASE_URL=http://127.0.0.1:8081 \
  uv run uvicorn vision_worker.main:app --port 8082
```

The fixture reasoner is the CPU/CI path. It proves relay, gallery, policy, and Memory wiring without
pretending a real visual model is running. The GN100 profile uses Cosmos at port 8001 and the
C-RADIO CUDA embedder.

## Checks

```bash
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Key suites cover:

- Cosmos native grounding parsing and honest malformed-response behavior;
- window cadence, placed-only promotion, identity skips, cooldown, and Memory writes;
- registration quality, diversity, and gallery refresh;
- status/event API contracts and full application lifespan; and
- domain isolation from model runtimes.

Linux ARM64/CUDA compatibility and real Cosmos/C-RADIO behavior require the physical GN100 gate;
CPU tests and cross-builds do not prove them.
