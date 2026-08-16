# Console

The page you drive and demonstrate the system from: publish or watch glasses
video, follow Cosmos placement and C-RADIO identity receipts, ask Memory a
question, exercise Speech and the guarded Agent, review personal-object
enrollment, and operate Remote Assist.

Replaced `services/media-gateway/src/media_gateway/static/publisher.html`, which
had grown to 447 lines of inline HTML, CSS and JavaScript served as a Python
string. That page is gone — see *What it replaced* below.

## Run it

```bash
cd apps/console && npm ci && npm run dev
```

Then <http://localhost:5173>. Nothing else is required: every panel reports its
own service as unreachable and the rest of the page keeps working, so the
console is useful with only the gateway running, or only vision.

The services it talks to, and what each panel needs:

| panel | service | default |
|---|---|---|
| Glasses, video | media-gateway | `http://127.0.0.1:8080` |
| Vision receipts | vision-worker | `http://127.0.0.1:8082` |
| Memory | application-memory | `http://127.0.0.1:8081` |
| Speech | speech | `http://127.0.0.1:8085` |
| Assistant | agent | `http://127.0.0.1:8086` |
| Enroll | vision-worker + application-memory | `http://127.0.0.1:8082`, `:8081` |
| Assist | media-gateway | `http://127.0.0.1:8080` |

Override with `VMA_GATEWAY_URL`, `VMA_VISION_URL`, `VMA_MEMORY_URL`,
`VMA_SPEECH_URL`, or `VMA_AGENT_URL` when starting the dev server. For glasses
pairing, also set `VITE_VMA_GATEWAY_PUBLIC_URL=http://<trusted-lan-ip>:8080`;
the QR must contain an address the glasses can reach, not browser loopback.

`VITE_VMA_VIEWER_VIDEO_QUALITY` selects the simulcast layer the admin viewer
requests: `high` (default), `medium`, or `low`. High is right for the glasses,
which publish a single 720p layer. Use `low` when watching a *development*
publisher that still has simulcast on — SG-C measured that a viewer competing
for the high layer there collapsed the gateway's own frames to 320x180.

## Why it is proxied

Everything is fetched from same-origin `/api/<service>` paths that Vite proxies
(`vite.config.ts`). No panel knows a host or a port, so there is no CORS story
in development and a built console points at whatever serves it.

`ws: true` on the gateway, vision, and speech proxies is load-bearing. Without it the
WebSocket upgrade is answered with a 404, which in a browser reads as a
mysterious immediate disconnect rather than a routing mistake.

## Push to talk and hands-free events

The Agent owns one Speech STT WebSocket for every publisher-present session and pushes
completed transcripts and guarded hands-free replies to the Gateway. The console owns one
Gateway device-event WebSocket for the selected session above the tab contents. Speech
and Assistant consume that shared event stream, so switching tabs neither closes a
connecting socket nor starts another Parakeet consumer. The three media panels stay
mounted but hidden so an in-flight turn and the glasses return-audio attachment survive
tab changes.
Holding the Assistant button arms the next completed transcript; releasing does
not start another recorder or upload browser audio. Once Speech ends the
contiguous segment, the panel posts its transcript to Agent, sends the guarded
reply to Speech synthesis, and plays the returned WAV when real TTS is enabled.

The panel always renders `answer_status` and the guard verdict. An external
Agent backend gets a prominent warning because transcript text then crosses the
local trust boundary. See [Agent Laptop Testing](../../docs/14-Agent-Laptop-Testing.md)
for the MiniCPM and OpenRouter free-model profiles.

## Enrollment review

The Enroll tab uses Vision's configured registration allowlist as its picklist. With a published
video session, **Freeze current POV** copies the displayed frame locally in the browser. The
operator drags a tight crop around the physical object, stages two to six distinct angles, enlarges
or removes any staged crop, and then chooses **Confirm and register**. No object or identity view is
created before that explicit confirmation. Vision applies geometric image-quality checks and
C-RADIO diversity ranking to the operator-selected crops, then writes only the surviving references
to Memory. Cosmos does not choose or veto Console enrollment crops. Frozen full frames and pending
crops remain browser-local; only confirmed crops cross the Vision API and become durable registry
evidence.

## Cosmos pipeline receipts

The current Vision pipeline deliberately has no per-frame detector/tracker, motion-state boxes, or
live overlay. The left stage shows the raw first-person camera. The Vision tab polls the current
`/v1/status` and `/v1/events` surfaces and presents the proof chain the demo audience needs:

```text
registered gallery → Cosmos window event → C-RADIO identity → durable Memory write
```

It reports gallery object/view counts, windows analyzed, memory-worthy events, identity
matches/skips, observations written, queue health, registration results, and an activity stream.
Each activity row carries the action, label, stable object ID when resolved, identity cosine, and
one explicit outcome:

- `written`: placement and identity passed every write gate;
- `skipped_no_identity`: no registered object matched strongly enough;
- `suppressed_by_policy`: motion was diagnostic-only under placed-only promotion; or
- `deduped`: the same object/action was recorded inside its cooldown.

This replaces the old box UI with evidence that corresponds directly to the Cosmos window and
Memory semantics actually deployed on the GN100.

## Components

Built on [shadcn/ui](https://ui.shadcn.com) with the audio components from
[ElevenLabs UI](https://ui.elevenlabs.io) (MIT). Both are *registries*, not
packages: components are copied into `src/components/ui` and are ours after
that.

ElevenLabs UI's primitives are verbatim copies of shadcn's — the `button.tsx`
they ship is byte-identical to the canonical one — so mixing the two produces no
divergence. Add primitives from either:

```bash
npx shadcn@latest add card
npx shadcn@latest add https://ui.elevenlabs.io/r/audio-player.json
```

`live-waveform` and `bar-visualizer` were vendored from the ElevenLabs GitHub
repo directly, because `ui.elevenlabs.io` rate-limits by IP and returned 429 for
this machine. They import only React and `@/lib/utils`.

**Do not add `use-scribe` or `voice-picker`.** Both reach for `@elevenlabs/client`
and their cloud. Speech here is Parakeet and Kokoro, running on-prem.

## What it replaced

`publisher.html` and its vendored 562 kB LiveKit bundle are gone, along with
the `/dev/publisher` and `/dev/static/vendor/{name}` routes and the `dev`
router that served them. This page does everything that one did:

- publish camera and microphone to the gateway's LiveKit room
- **rejoin** — same session and identity, new track SIDs, which is what the
  gateway turns into a new media epoch and the pipeline resets on
- camera and microphone toggles
- return-audio tone, played back on the assistant track
- vision and memory status panels
- hand the session slot back on `pagehide`

The gateway therefore has no browser UI of its own. `scripts/dev_up.sh` still
starts it alone, but publishing a camera now needs this console alongside it —
or `scripts/dev_stack.sh`, which starts both. For a headless publisher there
is `virtual-glasses`, which the integration suite uses.

## Known gaps

- **No per-frame boxes or scene depth map.** Cosmos reasons over sparse windows and emits bounded
  event receipts; it is not a continuous visual tracker.
- **Transcripts need a published session and hands-free Agent listener.** The Agent opens
  Speech STT for publisher-present sessions and pushes completed segments to the Gateway.
  A line appears when you stop speaking; there is no interim result.
- **Types are hand-written** (`src/lib/contracts.ts`), narrowed to the fields
  actually read. The intended end state is generation from each service's
  `/openapi.json`.
- **One bundle, ~970 kB** (275 kB gzipped), mostly `livekit-client`. Fine over a
  LAN; worth code-splitting if this is ever served over anything slower.
