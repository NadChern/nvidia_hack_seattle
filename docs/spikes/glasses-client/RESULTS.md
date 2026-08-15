# Glasses client spike results

- **Date:** 2026-08-12
- **Machine:** WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2), x86_64, no GPU used
- **Source commit:** `3c375e1` plus the untracked plan under `docs/15`
- **Components:** LiveKit server 1.13.4 (pinned, `.tools/`), `livekit` Python SDK from
  `services/media-gateway`, `agent.listener` from `services/agent`
- **Hardware used:** RayNeo X3 Pro ARGF20 added for SG-D; SG-A/B/C used no glasses or GN100.
- **Time box:** one session

Corpora in SG-A are hand-written and small (10, 12, and 4 items). They are diagnostic,
not a rate estimate: every failure below is categorical — a whole class of utterance
that cannot fire — so the denominators bound the shape, not the probability.

## SG-A — wake-prefix robustness

`agent.listener.triggered_question`, wake prefix `"hey memory"`.

| Corpus | n | Fired as expected |
|---|---|---|
| Clean wearer questions | 6 | 6/6 |
| Assistant replies alone — **must not** fire | 4 | 4/4 (none fired) |
| Reply audio prefixed to a real wake | 4 | **0/4** |
| Partial echo or disfluency before the wake | 4 | **0/4** |
| Misheard wake prefix | 6 | 1/6 (only the control) |

**There is no self-trigger loop.** Not one realistic assistant reply fires the matcher.
`triggered_question` demands the wake prefix *and* a where-question shape after it, and
guard-shaped replies (`"On the living-room coffee table at 10:42…"`, `"I have no record of
the keys."`) have neither. The echo-loop risk written into the plan's first draft was
overstated.

**The real failure is silent non-triggering.** The matcher requires the transcript to
*start with* the prefix, so anything in front of it suppresses the wake entirely:

```text
fire=False  uh hey memory where did i leave my keys
fire=False  um, hey memory, where are my keys
fire=False  i have no record of the keys. hey memory where did I leave my keys
fire=False  afterward hey memory where did i leave my keys
```

A disfluency is enough. Reply audio bleeding into the head of the next utterance is enough.

**Every plausible mishearing of the prefix fires nothing:** `hay memory`, `he memory`,
`hey memories`, `hey mammary` — 0/4.

## SG-A2 — scan for the prefix instead of anchoring to the start

Candidate: find the wake prefix anywhere in the utterance, keep the existing
question-shape gate on what follows, keep the alphanumeric boundary rule.

| Corpus | n | today | scan | scan + variant list |
|---|---|---|---|---|
| Should fire | 10 | 4 | **10** | **10** |
| Must not fire | 12 | 12 | **12** | **12** |
| Misheard prefix | 4 | 0 | 0 | **4** |

The no-fire corpus is the one that decides it, and it includes the cases that make
scanning look dangerous:

```text
hey memory is a cool name for the project      -> does not fire
the hey memory demo runs on the spark box      -> does not fire
hey, memory usage is climbing on the gpu       -> does not fire
```

They stay silent because the question-shape gate rejects them. **The double gate was
always what made this safe; `startswith` was never the part doing the work.** Scanning
recovered every miss at zero measured false-fire cost.

**Decision:** adopt the scan, add accepted prefix variants as configuration beside
`wake_prefix`. **Implemented:** `services/agent` now uses that scan and promotes these
corpora into regression tests. **Constraint:** the question-shape gate must not be
loosened at the same time — it is now carrying the whole false-fire budget.
**Fallback:** touchpad push-to-talk, which the plan already schedules at G4.

## SG-B — viewer token, ingest perturbation, token expiry

One local LiveKit, one room, three participants: a 720p simulcast publisher standing in
for the glasses, the gateway worker, and the proposed read-only console viewer. Publisher
and both subscribers on the same machine — the same arrangement as the demo laptop.

| | Result |
|---|---|
| **B1** Viewer with `can_publish=False` joins and receives video | **Yes** |
| **B2** Publish attempt from that token | **Refused — but as a `TimeoutError`**, not a clean error. The server never grants the track. A caller that awaits it hangs. |
| **B3** Effect on what the gateway receives | **Material degradation — see below** |
| **B4** Connected participant survives its token expiring | **Yes** |

### B3 — the console's arrival collapsed ingest resolution

Frames the gateway worker received, by dimension:

| Phase | 1280×720 | 640×360 | 320×180 | total |
|---|---|---|---|---|
| Before the viewer joined | 69 | 0 | 18 | 87 |
| After the viewer joined | 26 | 1 | **64** | 91 |

The 18 low-resolution frames in the first phase are ordinary simulcast ramp-up. The 64 in
the second are not: the viewer joining moved the gateway's own subscription down a layer,
so Vision would have been detecting on 320×180 instead of 720p, silently, for as long as
an operator had the console open.

**Attribution is not fully isolated.** Publisher, gateway, and viewer shared one loopback
machine, and libwebrtc logged `native video stream queue overflow` during the run, so
local decode contention plausibly drove some of the downgrade rather than the extra
subscriber alone. That does not make it ignorable: the demo laptop also runs the console
and the gateway together, so the topology that produced this is the topology we ship.

**Decision:** pin both subscriptions explicitly — gateway to the high layer, console
viewer to a low one — rather than letting adaptation choose. **Follow-up:** SG-C re-runs
this after the change, with the dimension guard's reject counter as the acceptance signal.

### B4 — token expiry, and a correction

The first run reported an expired token joining successfully, which would have been a
security-relevant defect in a dependency. It was not reported until it was checked.

Decoding the JWT confirmed the claim was genuine — `exp = iat + 2s`, verified against
wall clock. A sweep (SG-B2) then bounded the behaviour:

| Token age at join | Result |
|---|---|
| expired 0 s ago | joined |
| expired 8 s ago | joined |
| expired 28 s ago | joined |
| expired 88 s ago | **refused** |
| expired 298 s ago | **refused** |

Expiry **is** enforced. The window is ordinary JWT clock-skew leeway, almost certainly
60 s. LiveKit 1.13.4 is behaving correctly and the plan's refresh endpoint is still
required — the leeway is not a budget to spend.

## SG-C — explicit subscription quality is not sufficient

SG-C exercised the shipping `RoomWorker`, whose configured subscription quality is
`high`, then joined a read-only viewer that explicitly requested `low`. The gateway used
a strict 1280×720 dimension guard.

| Publisher | Gateway after viewer joined | Viewer | Result |
|---|---|---|---|
| 720p simulcast | 25/121 admitted; **96/121 rejected at lower dimensions** | 95 frames at 320×180, 25 at 1280×720 | **Fail** |
| Single 720p layer | **120/120 admitted; 0 rejected** | 720p track | **Pass** |

The explicit high/low requests did not isolate the subscriptions on the co-tenant setup.
The failure closely reproduced SG-B despite using the production gateway quality request.
A publisher-side single layer did isolate correctness because the SFU had no lower layer
to select. This matches `RoomWorker._request_quality`'s existing warning that a quality
request is not a guarantee.

**Decision:** the glasses publish one 1280×720, 15 FPS layer with simulcast disabled.
The console may still request low when a simulcast publisher is used in development, but
production correctness does not depend on that request. The measured cost is that the
operator also decodes 720p; include it in G7 thermal/load testing.

## SG-D — X3 Pro device preflight

The RayNeo X3 Pro was attached to WSL2 using `usbipd`, authorized with Linux ADB, and
probed with `sg_d_device_preflight.sh`. Raw output is in the repository-root
`sg_d_20260812.txt` working artifact.

| Question | Measured answer |
|---|---|
| Product | RayNeo ARGF20 / `MercuryLiteXR` |
| Android | Android 12, API 32 |
| ABI | `arm64-v8a` only |
| Display | one Android display, 1280×480 at 160 dpi |
| Eye layout | side-by-side 640×480 buffers; verified by `screencap` |
| Cameras | two Camera2 back-facing cameras; CameraX analysis starts normally |
| Touch surfaces | `cyttsp5_mt` and `cyttsp6_mt`, ordinary direct multi-touch inputs |
| Audio preflight | no AEC effect reported by `dumpsys audio`; routing still needs SG-F |
| Baseline thermal status | 0 |
| Baseline battery | 100% |

The app installed and launched without a fatal exception. Camera and microphone runtime
permissions were granted, and the app remained the resumed activity. The original flat
HUD rendered once at the center seam, proving it was not usable as a binocular surface.
The client now renders identical content into each 640×480 half; an ADB screenshot
confirmed matching left/right pairing prompts. The camera preview was removed from the
HUD, while headless CameraX `ImageAnalysis` continues scanning QR codes.

**Decision:** enable the `arm64-v8a` ABI filter and use a stereo Compose root. Plain
Compose gesture handling is viable because both touch controllers are standard direct
multi-touch devices, but semantic mapping of temple gestures to actions still needs an
interactive app pass. SG-D closes the API, ABI, basic CameraX, display-layout, and input
transport unknowns; it does not close camera-ID selection, safe-area comfort, audio/AEC,
or sustained operation.

## Consequences for the plan

1. Gap 5 added: the wake matcher is fixed before anything is built against it.
2. Gap 1 requires publisher-side single-layer video; subscription pinning alone failed.
3. Gap 4 keeps its refresh endpoint, for the right reason.
4. The plan's echo-loop risk is rewritten; the plan's "wake word may be brittle" risk is
   promoted from speculation to a measured number.

## Not answered here

SG-E (LiveKit Android SDK on the X3 Pro — S01's outstanding gate) and SG-F (on-device
AEC/routing) now have hardware available but still require an end-to-end session and
interactive observation. SG-D does not substitute for either gate.
