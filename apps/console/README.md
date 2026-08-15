# Console

The page you drive the system from: publish a camera into the gateway, watch
detections land on it in real time, ask memory a question, exercise speech,
and run the guarded push-to-talk assistant.

Replaced `services/media-gateway/src/media_gateway/static/publisher.html`, which
had grown to 447 lines of inline HTML, CSS and JavaScript served as a Python
string. That page is gone — see *What it replaced* below.

## Run it

```bash
cd apps/console && npm install && npm run dev
```

Then <http://localhost:5173>. Nothing else is required: every panel reports its
own service as unreachable and the rest of the page keeps working, so the
console is useful with only the gateway running, or only vision.

The services it talks to, and what each panel needs:

| panel | service | default |
|---|---|---|
| Glasses, video | media-gateway | `http://127.0.0.1:8080` |
| Vision, boxes | vision-worker | `http://127.0.0.1:8082` |
| Memory | application-memory | `http://127.0.0.1:8081` |
| Speech | speech | `http://127.0.0.1:8085` |
| Assistant | agent | `http://127.0.0.1:8086` |

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

## The boxes

`WS /v1/overlay?session_id=<selected-session>` sends coordinates, not pixels.
The Console already has the selected glasses video, so the vision worker sends
normalized boxes
and the browser draws them on a `<canvas>` over its own `<video>`. Kilobytes per
second instead of megabits, no second encode, crisp at any zoom.

It also means the latency is **real and visible**, which is the point. The
`latency` readout is `emitted_at - relayed_at`, both stamped inside the vision
worker, so it measures the pipeline rather than the gap between two machines'
clocks. A re-encoded annotated video track would look perfectly synchronised and
prove nothing.

Box colour is the object's `motion_state`, not decoration: watching a box go
amber → yellow → green as something is set down is the clearest evidence that a
state machine is running rather than a detector firing once per frame.

The session query is required by the Console even though the backend retains an
unscoped diagnostic mode. This prevents detections from another publisher or a
previous selection being drawn over the active glasses video.

`missed` counts gaps in the frame sequence — overlays the pipeline produced that
this browser was too slow to read. It costs nothing real (a stale overlay would
not have been drawn) but it distinguishes a slow laptop from a slow pipeline.

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

- **No scene depth map.** Depth reaches the console as a number per object,
  drawn on the box with its age — a full map would need MoGe on every frame,
  which is the cost the once-a-second cadence exists to avoid. Set
  `VMA_DEPTH_KIND=moge` on the vision worker to see the numbers at all.
- **Transcripts need a published session and hands-free Agent listener.** The Agent opens
  Speech STT for publisher-present sessions and pushes completed segments to the Gateway.
  A line appears when you stop speaking; there is no interim result.
- **Types are hand-written** (`src/lib/contracts.ts`), narrowed to the fields
  actually read. The intended end state is generation from each service's
  `/openapi.json`. Until then the overlay `schema_version` is checked at runtime
  and a mismatch is shown in the video panel.
- **One bundle, ~946 kB** (268 kB gzipped), mostly `livekit-client`. Fine over a
  LAN; worth code-splitting if this is ever served over anything slower.
