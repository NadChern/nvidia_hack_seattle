# Glasses Client Plan — RayNeo X3 Pro

The plan for `apps/glasses-x3`, a pure-Kotlin Android client that replaces the console's
"pretend to be the glasses" Publish button with the real device.

No Unity. The client needs no 6DoF pose, no spatial anchors, and no world mesh — it
captures first-person media, plays a spoken reply, and draws flat text. Everything Unity
would add is weight, an extra runtime, and a second rendering stack on a battery-limited
device. [Everyday](https://github.com/TheophileGaudin/Everyday) is our reference that a
plain Android app on this hardware is viable: it is a three-module Gradle project
(`_glasses`, `_phone`, `_shared`) installed with ordinary `adb install`, with no
proprietary RayNeo SDK in its dependency surface.

Five of this plan's assumptions have since been measured rather than asserted; three of them
were wrong. See [Spikes](#spikes) and [the results](spikes/glasses-client/RESULTS.md).

## Decisions

| Decision | Choice |
|---|---|
| Runtime | Pure Kotlin / Jetpack Compose, no Unity |
| Transport | LiveKit WebRTC, same room the console publishes to today |
| Console view of the glasses | Console joins the room as a **subscriber** with a viewer token |
| Client scope | Capture + HUD + spoken reply |
| Question trigger | Server-side wake prefix on the Parakeet transcript |
| Pairing | QR code rendered by the console, scanned by the glasses |
| Location | This monorepo, `apps/glasses-x3` |

## The half of this that already exists

Most of the loop is built. The client's job is smaller than it looks, and the plan is
mostly about **not** rebuilding these:

- **Token minting.** `POST /v1/sessions {device_id}` returns `room`, `livekit_url`,
  `identity`, `token`, `expires_at` — [sessions.py:77](../services/media-gateway/src/media_gateway/api/sessions.py:77).
  The Kotlin client speaks the same endpoint the console already does. No backend change
  for ingest.
- **Ingest, sampling, guarding.** The gateway subscribes, samples to 8 FPS with a
  latest-wins slot, and discards off-dimension frames before the sampler
  ([docs/12](12-Media-Relay-Contract.md#video)). The glasses publish; nothing downstream
  changes.
- **The spoken reply path.** The gateway joins each room and publishes an
  `assistant-tts` audio track (`return_audio_track_name`, [config.py:137](../services/media-gateway/src/media_gateway/config.py:137)).
  The glasses subscribe to it like any remote track. **No new backend work for reply audio.**
- **Wake word, server side.** `services/agent/listener.py` is already a
  "wake-prefix-gated hands-free transcript listener": wake prefix `"hey memory"`
  ([config.py:98](../services/agent/src/agent/config.py:98)), a bounded where-question
  shape, the guard, TTS, and PCM to the gateway's return-audio socket. Utterance
  boundaries come from Silero VAD in [speech/utterance.py](../services/speech/src/speech/utterance.py).
  **The chosen wake-word design is already implemented.** The glasses need to stream mic
  audio and nothing else. The matcher fix in [gap 5](#5-the-wake-matcher-was-too-strict-to-survive-a-room)
  is now implemented and regression-tested.
- **Live transcripts.** Speech exposes `WS /v1/stt/{session_id}`, which the Agent's
  `HandsFreeListener` and the console's `TranscriptProvider` consume today. The HUD does
  **not** add another STT consumer; it receives transcript and reply events through the
  gateway channel in gap 2.

## The five gaps

### 1. Viewer token and a console subscriber view

**Implemented; SG-C closed with a changed constraint.** The gateway now mints a
subscribe-only viewer grant, and the console lists publisher-present sessions, joins in
viewer mode, requests low when simulcast exists, and keeps `OverlayCanvas` over the remote
track. SG-C proved that gateway-high plus viewer-low does **not** hold under the demo
topology. The production glasses must publish one 1280×720 layer with simulcast disabled;
that admitted 120/120 gateway frames after the viewer joined.

The original gap was that `mint_access_token` issued one grant shape for every role:
`can_publish=True`,
`can_subscribe=True` ([tokens.py:68](../services/media-gateway/src/media_gateway/transport/tokens.py:68)).
A console that joins the room needs a genuinely read-only grant, and today it cannot get one.

- Add a `viewer` `GrantRole` whose grants set `can_publish=False`,
  `can_publish_data=False`, `can_subscribe=True`.
- Add `POST /v1/sessions/{session_id}/viewer` behind the existing `authorize_request`
  bearer check, returning the same `SessionTokenResponse` shape with a viewer identity.
- Console: subscribe to the remote video track and render it in `VideoStage`, so
  `OverlayCanvas` keeps drawing detections over it. Today `VideoStage` attaches only the
  **local** preview via `attachPreview` ([glasses.ts](../apps/console/src/store/glasses.ts)) —
  the store grows a second, subscribe-only mode beside `publish()`.
- Session selection is **already served**: `GET /v1/sessions`
  ([sessions.py:162](../services/media-gateway/src/media_gateway/api/sessions.py:162))
  returns every live session with `publisher_present`, which is exactly how the Agent's
  `HandsFreeListener` discovers sessions today. The console needs a picker, not an endpoint.

[SG-B](spikes/glasses-client/RESULTS.md) measured what happens when the viewer joins
mid-stream: before the viewer, the gateway received 69 of 87 frames at 1280×720; after,
only 26 of 91, with 64 frames at **320×180**. One loopback machine ran publisher, gateway,
and viewer, but the demo laptop uses the same topology.

**Subscription pinning was not the fix.** SG-C exercised the production gateway requesting
high and a viewer requesting low; after viewer join, only 25/121 gateway frames passed the
720p guard. Repeating with publisher simulcast disabled admitted 120/120. Publish one
720p/15 FPS layer from Android, and let no inference guarantee depend on the SFU honouring
per-subscriber quality.

**The viewer therefore asks for HIGH, not LOW.** Once the publisher sends a single layer
there is nothing to demote to, and operators need the detail Vision has. The request is
read from `VITE_VMA_VIEWER_VIDEO_QUALITY` (default `high`) so a development publisher that
still uses simulcast can be watched at `low` without a code change. This is the one place
where the demo default and the safe-under-simulcast default differ, which is exactly why
it is a setting.

A refused publish surfaces as a **timeout, not an error** (SG-B, B2): the server simply
never grants the track. The console must bound that call rather than await it.

**This amends [docs/12](12-Media-Relay-Contract.md#boundary), which states the gateway
holds the only LiveKit subscription and that consumers do not join the room.** The rule's
stated purpose is to keep decode cost and token surface at one for *inference* consumers;
an operator's admin view is a different class of consumer. The doc gets an explicit
carve-out — a viewer token is read-only, is minted per session behind the internal bearer
token, and no inference or memory path may depend on it. Rewrite that section rather than
letting code and doc silently disagree.

### 2. Reply text has no push channel

**Implemented.** The Agent now posts every completed transcript and every guarded
hands-free reply to the Gateway. `WS /v1/device/{session_id}/events` fans typed, bounded
JSON events to the owning glasses credential and internal console. The console's shared
provider consumes this channel instead of opening another Speech STT socket; the Android
HUD uses the same channel.

The original gap was that the console got an answer by *calling* `POST /v1/agent/query` and rendering the response
([AgentPanel.tsx:75](../apps/console/src/features/agent/AgentPanel.tsx:75)). The
hands-free listener path produces **audio only** — there is no channel that pushes reply
text anywhere. So a HUD cannot show what the assistant just said.

LiveKit data channels are deliberately disabled (`can_publish_data=False`, "an unaudited
side path between participants"), so this cannot ride the room.

**The gateway is the device's only counterparty.** The obvious shape — glasses open a
socket to Speech on :8085 for transcripts and another to the Agent on :8086 for replies —
creates three problems at once: the QR would have to carry three addresses, a device
credential would have to be understood by three services, and the STT socket would gain a
*third* consumer beside the Agent's listener and the console's `TranscriptProvider`, which
is explicitly written to avoid starting "a second Parakeet consumer for the same session"
([TranscriptProvider.tsx:10](../apps/console/src/hooks/TranscriptProvider.tsx:10)).

Route it through the service the device already talks to:

- The **Agent already owns an STT socket per session** (`HandsFreeListener`) and already
  posts to the gateway for return audio. Give it one more post: the transcript it just
  saw, and the reply it just produced.
- Gateway gains `WS /v1/device/{session_id}/events`, fanning those out to the glasses and
  to the console. Keep it separate from the `VMA1` relay — that framing is for decoded
  media, and mixing text into it would blur a contract [docs/12](12-Media-Relay-Contract.md)
  is careful about.
- Carry `guard` on the wire. [Guard rule 3 is fail-closed on purpose](../services/agent/src/agent/guard.py),
  and a vetoed reply must read on the HUD as a deliberate refusal, not as a failure.
- The console gains the hands-free answers it currently cannot see either.

This costs the gateway a fan-out responsibility. It buys one URL in the QR, one service
that validates the device credential, and no new Parakeet consumer.

### 3. Pairing

**Implemented through the buildable client; device scan remains G1 acceptance.** The
console renders a QR from `POST /v1/pairing`; the Android CameraX/ML Kit scanner claims it
once and persists the resulting device credential and Gateway URL in DataStore. The
Gateway stores only bounded hashes of pending codes. Device credentials are signed,
expiring, survive Gateway restart while the internal secret is stable, and authorize only
session create/own refresh/own events.

Glasses have no keyboard and the venue's IP will move. The console renders a QR; the
glasses scan it with CameraX + ML Kit barcode scanning and persist the result to
`DataStore`. The demo client deliberately reopens target selection on every new app
process, allowing the wearer to scan either the laptop or GN100 QR. The old pairing is
retained until the replacement claim succeeds and is exposed through RayNeo's documented
temple-touch focus model: swipe changes focus and a single tap activates the saved-target
action. Because of gap 2, the
payload needs exactly one address:
`{gateway_url, pairing_code, expires_at}`.

**Do not put the raw internal bearer token in the QR.** Anyone who photographs the console
screen keeps it, and it authorizes every internal surface on the gateway. The exchange,
all of it inside the gateway:

1. Console calls `POST /v1/pairing` with the internal bearer → short-TTL, single-use
   `pairing_code`. Rendered as the QR.
2. Glasses call `POST /v1/pairing/claim` with the code and their `device_id` → a
   device-scoped credential that is **not** the internal bearer.
3. That credential authorizes only what a device needs: create, refresh, and delete a session for
   its own `device_id`, open its own device-events socket, and arm its consume-once manual
   trigger. Not `/v1/stream/*`, not
   `/v1/status`.

If the schedule collapses, shipping the raw token is survivable for a demo on an isolated
network — but it is a knowing trade, and it must be written down in
[docs/07](07-Privacy-and-Security.md), not discovered later.

### 4. Token lifetime across a long session

**Implemented through the buildable client; five-minute/device reconnect remains G7
acceptance.** `POST /v1/sessions/{session_id}/token` re-mints the publisher grant without
changing the session, room, or identity. The owning device credential is accepted, and
the Android client schedules refresh at 60% of the returned expiry and retains that grant
for reconnect.

`token_ttl_s` defaults to 300 s ([config.py:81](../services/media-gateway/src/media_gateway/config.py:81)).
A JWT is checked at join, so a five-minute token does not end a running session — but a
reconnect after expiry cannot rejoin, and the only way to get a new token today is
`POST /v1/sessions`, which mints a **new session id**. That silently splits one wearing
into two sessions mid-demo.

Add `POST /v1/sessions/{session_id}/token` to re-mint for an existing session, and have
the client refresh at ~60% of `expires_at`.

[SG-B](spikes/glasses-client/RESULTS.md) checked both assumptions underneath that, and
one of them needed a second pass. A connected participant does survive its token expiring
— good, a five-minute token does not end a wearing. A join with an expired token first
appeared to succeed outright, which would have been a security-relevant surprise; decoding
the JWT confirmed the `exp` claim was real, and a sweep (SG-B2) bounded the actual
behaviour: LiveKit 1.13.4 admitted a token **28 s** past expiry and refused one **88 s**
past it. That is ordinary clock-skew leeway, almost certainly 60 s, not an unenforced
check. Expiry is enforced; the refresh endpoint is required; do not spend the leeway.

### 5. Intent routing belongs to the Agent model

**Implemented.** The server no longer uses a wake-prefix or question-shape regex before
calling the Agent. Every completed non-empty STT transcript from an authenticated live
glasses session reaches Nemotron. Its system instruction selects `where_is`,
`start_registration`, or no supported action, so a bare “where are my keys?” does not
silently stop between STT and the Agent.

This intentionally removes the earlier ambient-speech filter and its configurable wake
variants. The Gateway and Android HUD may still expose the consume-once manual-trigger
interaction as user feedback, but the Agent listener does not require or inspect it. The
trust boundary remains downstream: only Memory can authorize a location, and the
deterministic reply guard rejects an ungrounded model answer.

## Module layout

```text
apps/glasses-x3/
  settings.gradle.kts
  app/                    Compose HUD, the only installed artifact
    pairing/              QR scan, DataStore credential, gateway URL
    session/              POST /v1/sessions, token refresh, lifecycle
    media/                LiveKit Room, camera + mic publish, assistant-tts playback
    events/               gateway device-events socket: transcripts and replies
    hud/                  Compose overlay, safe area, live trigger state
```

One installed module. Everyday's `_phone` and `_shared` split solves a companion-app
problem we do not have; a second module now costs Gradle wiring and buys nothing. The
package boundaries above are where a module split would go if a phone companion ever
appears.

**No validator carve-out is needed.** I claimed one in the first draft and it was wrong:
`validate_repo.py` enforces its Python service rules only under `services/`, and its one
repository-wide check forbids `Pipfile`, `poetry.lock`, and `requirements*` files — none of
which an Android project produces. Adding `apps/glasses-x3` requires no change to the
validator and no exception list.

CI does need a new job: `gradle assembleDebug` plus unit tests, on JDK 17.

**The toolchain is already project-local**, in the gitignored `.tools/` beside the pinned
LiveKit server: `.tools/jdk17` and `.tools/android-sdk`. The system JDK is 8, which is
irrelevant and misleading — nothing needs installing. Build and install with:

```bash
JAVA_HOME=$PWD/.tools/jdk17 ANDROID_HOME=$PWD/.tools/android-sdk \
  apps/glasses-x3/gradlew -p apps/glasses-x3 testDebugUnitTest lintDebug assembleDebug
adb install -r apps/glasses-x3/app/build/outputs/apk/debug/app-debug.apk
```

## Milestones

Each one ends in something observable on the device or in the console. Nothing is "done"
until it has been run.

| # | Milestone | Done when |
|---|---|---|
| **S** | SG-D device preflight; JDK 17 toolchain; wake-matcher fix (gap 5) | **Done.** Gradle/JDK 17 build and SG-A2 tests pass; SG-D answers recorded in [RESULTS](spikes/glasses-client/RESULTS.md) |
| **G0** | Gradle skeleton, Compose HUD, `adb install` | Text renders on the X3 Pro display, legible at the real focal distance |
| **G1** | Console QR view + glasses scanner + persisted credential | After a cold restart, glasses default to QR selection and temple swipe/tap can activate the preserved saved target |
| **G2** | Session + LiveKit join, publish camera and mic | `GET /v1/status` on the gateway reports the publisher present with two epochs; frames pass the dimension guard |
| **G3** | Subscribe `assistant-tts`, play through the glasses speaker | `POST /v1/return-audio/{id}/tone` is audible on the device |
| **G4** | HUD v1: session state, live transcript from Speech STT | Speech into the glasses appears as text on the display |
| **G5** | Viewer token + console subscriber view + single-layer publishing | The console shows the wearer's live camera with detection overlays; SG-C confirms a non-simulcast publisher keeps gateway ingest at 720p |
| **G6** | Agent reply push + HUD answer + end-to-end | "Hey memory, where did I leave my keys?" → spoken reply in the glasses, text on the HUD and in the console |
| **G7** | Hardening | 20-minute continuous session with no thermal throttle, reconnect after Wi-Fi drop, token refresh crossing 5 minutes |

G5 is independent of G0–G4 and is TypeScript + Python work — it can run in parallel with
the Kotlin track rather than behind it.

### Implementation stop point

All work that can be validated without the X3 Pro is implemented:

- JDK 17/Gradle 8.9 build, twelve JVM tests, Android lint, and `assembleDebug` pass;
- QR issue/claim, persisted device credential, session create/refresh, LiveKit room setup,
  non-simulcast 720p/15 FPS capture configuration, explicit AEC/NS, HUD event socket, and
  Compose HUD compile;
- viewer/session picker, guarded reply display, SG-C single-layer gate, and Android CI are
  in place;
- SG-D's answers are folded back into the client: `minSdk` 26, the world camera resolved by
  enumeration rather than a hardcoded `"0"` (the device reports two back-facing cameras),
  and a `SessionForegroundService` with `camera|microphone` types so Android does not
  revoke capture the moment the Activity loses focus.

G0–G6 remain open only where their definition requires observing real display, camera,
microphone, speaker, touchpad, network, or lifecycle behaviour. **An APK that compiles is
not evidence for those hardware claims**, and neither is a probe: SG-D answered what the
device *reports*, while SG-E and SG-F need a session actually running on it.

The HUD shows whether the `assistant-tts` track is subscribed, because reply audio plays
through LiveKit's own routing and a broken return path would otherwise be indistinguishable
from an assistant with nothing to say.

## Spikes

Scripts and full results: [docs/spikes/glasses-client](spikes/glasses-client/README.md).
The point of the first three was that none of them needed the glasses, an Android
toolchain, or the GN100 — so they were worth running before writing the plan's
assumptions into code.

| | Question | Status |
|---|---|---|
| **SG-A** | Can reply audio self-trigger the wake word, and what does echo do to a real wake? | **Done.** No self-trigger. Recall 4/10 — see gap 5 |
| **SG-A2** | Does scanning for the prefix recover recall without buying false fires? | **Done.** 10/10 recall, 12/12 no-fire |
| **SG-B** | Viewer token: does it join, is publish refused, does it perturb ingest, does expiry bite? | **Done.** Joins; publish refused as a timeout; **ingest resolution collapsed**; expiry enforced after ordinary clock-skew leeway |
| **SG-C** | Does pinning the gateway high and viewer low restore 720p ingest? | **Done: no.** 25/121 admitted with simulcast; disabling publisher simulcast passed 120/120 |
| **SG-D** | Device preflight: API level, ABI, camera enumeration, touchpad `MotionEvent`s, display safe area, speaker/mic routing | **Done.** API 32, arm64-v8a only, 1280×480 display, standard `cyttsp` multitouch, **two** back cameras, **no hardware AEC** |
| **SG-E** | LiveKit Android SDK publishing from the X3 Pro: permissions, lifecycle, codecs, 30-minute stability | Needs device **and** toolchain. This is S01's outstanding gate |
| **SG-F** | Is on-device AEC enough to keep reply audio out of the transcript? | Needs the device. Gap 5's fix reduces what rides on the answer |

SG-E and SG-F are now the only gates left, and both need an end-to-end session on the
device rather than a probe. SG-F matters more than it did: SG-D found **no hardware echo
canceller**, so keeping the assistant's own reply out of the microphone rests entirely on
WebRTC's software APM, which the client enables explicitly.

## Risks

**The wake word failed silently, and echo was why — but not the way the first draft
claimed.** [SG-A](spikes/glasses-client/RESULTS.md) found no self-trigger loop: 0 of 4
realistic assistant replies fired because the matcher demands both a wake prefix and a
where-question shape after it, and guard-shaped replies have neither.

The measured failure was the opposite. The baseline required the transcript to start
with the prefix, so reply audio or a plain disfluency suppressed a legitimate trigger.
Measured recall was **4/10**. Gap 5 is now implemented: the matcher scans on word
boundaries, retains the question-shape gate, and accepts configured STT variants. Keep the
SG-A2 no-fire corpus in the Agent suite whenever this gate changes.

AEC is still worth enabling — LiveKit Android's WebRTC `AudioProcessing` (AEC + NS) at
`Room` construction — because keeping reply audio out of the transcript is better than
tolerating it. But it is now a second line of defence, not the fix.

**Wake word in a hackathon room.** Server-side matching inherits every STT error: SG-A
found that all four plausible mishearings of the prefix (`hay memory`, `he memory`,
`hey memories`, `hey mammary`) fired nothing in the baseline. They are now explicit,
configurable variants rather than fuzzy matches. A hardware manual trigger remains a useful fallback. The target chooser now proves the
vendor-supported interaction path—temple swipe changes focus and single-tap activates.
Single-tap now arms the existing consume-once manual trigger during a live session; it
retains the where-question gate. Accidental activation still needs soak testing.

**SG-D device answers:** the ARGF20 runs Android 12/API 32 and supports only
`arm64-v8a`; Camera2 exposes two back-facing cameras and CameraX analysis starts normally;
the two Cypress touch controllers surface as ordinary direct multi-touch inputs; and the
Android display is a 1280×480 side-by-side binocular surface. The app now ABI-filters to
`arm64-v8a` and duplicates its Compose HUD into two 640×480 eye buffers. Camera-ID
selection, gesture semantics, and comfortable optical safe area remain interactive checks.

**Thermals and battery.** Continuous 720p encode plus a WebRTC uplink on glasses is a real
power draw. Publish one non-simulcast 720p layer at 15 FPS — the gateway samples to 8 FPS regardless, so a higher
capture rate is spent entirely on heat. Set the gateway's dimension guard to match
whatever the device actually produces, and confirm with the guard's own reject counter
rather than by assumption.

**LiveKit Android SDK size and ABI.** The bundled WebRTC native libs are large. SG-D
measured `arm64-v8a` as the device's only ABI, and the Android build now filters to it.

## Related

- [Media Relay Contract](12-Media-Relay-Contract.md) — the subscription boundary this plan amends
- [Privacy and Security](07-Privacy-and-Security.md) — where the pairing-credential trade must be recorded
- [Data Contract](06-Data-Contract.md) — what a memory is; the client produces none
- [Team Split](05-Team-Split.md) — ownership and integration milestones
