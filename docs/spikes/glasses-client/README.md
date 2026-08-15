# Glasses client spikes

Unknowns behind [docs/15 — Glasses Client Plan](../../15-Glasses-Client-Plan.md),
reduced to measurements. Everything here runs on a developer machine with **no
glasses, no Android toolchain, and no GN100** — that is the point. What is left
after these run is the genuinely hardware-bound set, listed in the plan as
SG-D, SG-E, and SG-F.

Per [docs/09 § Required records](../../09-Spike-Plan.md), results and their
denominators live in [RESULTS.md](RESULTS.md).

## SG-A — wake-prefix robustness against echo and STT error

`agent.listener.triggered_question` is a pure function, so the entire
wake-word risk surface can be measured without audio, a model, or a device.

```bash
cd services/agent && uv run python ../../docs/spikes/glasses-client/sg_a_wake_echo.py
```

Corpora are hand-written and small, so treat the numbers as a shape rather than
a rate: the failures are categorical, not marginal.

## SG-A2 — does scanning for the wake prefix recover recall for free?

The accepted fix for what SG-A found, scored on the same corpora plus an adversarial
no-fire set. The script retains the pre-SG-A2 implementation locally so the baseline
remains reproducible after production adopts the scan, then scores production too.

```bash
cd services/agent && uv run python ../../docs/spikes/glasses-client/sg_a2_wake_scan.py
```

## SG-B — viewer token, ingest perturbation, and token expiry

Three participants against a real local LiveKit: a 720p simulcast publisher
standing in for the glasses, the gateway worker, and the proposed read-only
console viewer.

Answers four questions the plan asserted without evidence:

| | Question |
|---|---|
| B1 | Does a `can_publish=False` token join and receive video? |
| B2 | Does the server refuse a publish from that token? |
| B3 | Does the viewer joining mid-stream change the frame dimensions the gateway sees? [docs/12](../../12-Media-Relay-Contract.md#video) records transient 8×8 frames during simulcast adaptation, so a new subscriber forcing a layer change would land directly on the dimension guard. |
| B4 | Does a connected participant survive its token expiring, and is an expired token refused at join? |

```bash
export VMA_LIVEKIT_API_KEY=vma-dev
export VMA_LIVEKIT_API_SECRET=$(openssl rand -hex 24)
export LIVEKIT_KEYS="$VMA_LIVEKIT_API_KEY: $VMA_LIVEKIT_API_SECRET"
.tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml &
cd services/media-gateway && uv run python -u ../../docs/spikes/glasses-client/sg_b_viewer_token.py
```

Run it unbuffered and redirect to a file. The script holds several rooms open
for the better part of a minute, and a pipe that is killed on timeout takes the
buffered results with it. Results print as they are determined for the same
reason.

## SG-B2 — is token expiry enforced, and with how much leeway?

SG-B saw an expired token join. This separates clock-skew leeway from expiry not
being checked at all, which have very different consequences for the refresh
design. Same server as SG-B; takes about seven minutes, mostly sleeping.

```bash
cd services/media-gateway && uv run python -u ../../docs/spikes/glasses-client/sg_b2_token_expiry.py
```

## SG-C — production simulcast pinning

Runs the shipping gateway subscription (`high`) against a 720p simulcast publisher, then
joins a read-only viewer that explicitly requests `low`. A strict 1280×720 gateway guard
turns any gateway layer collapse into rejected-dimension counters.

Start LiveKit as for SG-B. In a second terminal, from `services/media-gateway`:

```bash
VMA_MEDIA_SOURCE=livekit \
VMA_DIMENSION_GUARD_MODE=strict \
VMA_EXPECTED_VIDEO_WIDTH=1280 \
VMA_EXPECTED_VIDEO_HEIGHT=720 \
uv run uvicorn media_gateway.main:app --port 8080
```

Then, from the repository root, run both publisher shapes:

```bash
cd services/media-gateway
uv run python -u ../../docs/spikes/glasses-client/sg_c_simulcast_pinning.py
SG_C_SIMULCAST=false \
  uv run python -u ../../docs/spikes/glasses-client/sg_c_simulcast_pinning.py
```

Pass requires at least 80% of frames reaching the gateway after viewer join to pass the
720p guard. Explicit high/low subscription requests failed that gate; a single-layer
publisher passed 120/120 after viewer join. Record the dimension counters and co-tenant
warnings; one loopback run is still topology evidence rather than isolated causal
attribution.

## SG-D — device preflight

The only spike here that needs the glasses. Ten minutes with a USB cable, `adb`
only — no Android Studio, no Gradle, no JDK. Run it before G0: every question it
asks can move a milestone, and none can be answered from a datasheet.

```bash
adb devices
./docs/spikes/glasses-client/sg_d_device_preflight.sh > sg_d_$(date +%Y%m%d).txt
```

Record the answers in [RESULTS.md](RESULTS.md) under SG-D.

## Watching the HUD without wearing the glasses

`scrcpy` mirrors the device framebuffer over `adb`, which makes every HUD claim
screenshottable and reviewable by someone who is not wearing the device:

```bash
scrcpy --serial <device> --window-title ARGF20
```

The mirror shows the **stereo framebuffer**: 1280×480 containing the two
640×480 eye views that `StereoLayout` renders, so every element appears twice.
That is the app working correctly, not a duplication bug.

**What this cannot close.** The framebuffer is not what the wearer perceives
through the optics. Legibility at the real focal distance, comfortable safe
area, and whether text sits where the eye expects it are still human checks on
the device — G0's acceptance is deliberately worded that way. Use scrcpy to
prove *content and state*, never comfort.
