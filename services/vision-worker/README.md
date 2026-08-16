# Vision Service (`vision-worker`)

Watches the media relay and decides whether an object is at rest somewhere or
moving through the world — the one distinction the Memory Service's whole
truthfulness story depends on.

> Keys carried from the kitchen to the front door and pocketed must never
> resolve to "the front hall" just because the last sighting was there.

The distinction that decides a location was never *"is a hand touching it"* —
it is *"is this object at rest, or moving."* A sighting only updates the
confirmed location once the object has held its position for a dwell period;
a sighting while moving never overwrites a confirmed placement. Producing that
distinction — not detecting objects, which any detector does — is this
service's actual job.

## See it work

No camera, no model, no GPU:

```bash
cd services/vision-worker && uv run pytest tests/test_pipeline.py tests/test_fixture_scenarios.py -v
```

`test_pipeline.py` drives the entire stack — relay consumer, detector,
tracker, background-motion estimator, the stability machine, the evidence
ring, and the verifier — through hand-built frames with no mocks below the
relay boundary, and proves the two claims that matter: an object carried in
and set down produces a confirmed `placed` candidate, and an object that
never stops moving never does, no matter how long it is watched.

`test_fixture_scenarios.py` replays the eight golden scenarios in
[`packages/vision-contract/fixtures.py`](../../packages/vision-contract/src/visual_memory_vision_contract/fixtures.py)
— including the demo case itself — against the real production thresholds,
not a shortened test config.

## Run it

Run `uv` from **inside this directory**; from the repository root it resolves
a different environment and fails with `ModuleNotFoundError`.

```bash
cd services/vision-worker
VMA_GATEWAY_VIDEO_URL=ws://127.0.0.1:8080/v1/stream/video \
VMA_MEMORY_BASE_URL=http://127.0.0.1:8081 \
  uv run uvicorn vision_worker.main:app --port 8082
```

Pair it with [media-gateway](../media-gateway/README.md)'s scripted or
laptop-camera path and [application-memory](../application-memory/README.md)
running locally, then watch `/v1/status` for `frames_processed` to confirm
the relay connection is live.

| Endpoint | Purpose |
|---|---|
| `GET /health/live` · `GET /health/ready` | Process health; readiness reflects whether the relay task is alive, never whether an object is in view |
| `GET /v1/status` | Reasoner/promotion configuration, identity/registration state, and pipeline counters |
| `POST /v1/objects` · `GET /v1/objects` | Create/list personal objects through Memory's registry |
| `POST /v1/objects/{id}/capture` · `GET /v1/objects/{id}/status` | Arm and poll bounded registration capture |

There is no video ingestion or memory-query endpoint here. The registration
routes are bounded control operations; ordinary observations are still posted
to `application-memory`'s `/v1/observations` via `MemoryEmitter`.

### Placement promotion policy

`VMA_PROMOTE_MOTION_EVENTS=false` is the default. Cosmos can still report
`picked_up` and `carried`, and those labels remain visible as
`suppressed_by_policy` diagnostics, but only identity-matched `placed` events
are written to Memory. This prevents one false motion label from invalidating a
confirmed placement on the current sparse glasses uplink. `/v1/status` exposes
the active toggle and `metrics.motion_events_suppressed`.

Set `VMA_PROMOTE_MOTION_EVENTS=true` to restore the canonical full timeline once
motion classification has been validated at the deployed frame rate. The
toggle changes Vision promotion only; it does not change the observation or
Memory contracts.

### The whole stack, with the real models

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

Plain `compose.yaml` runs `VMA_DETECTOR_KIND=fixture` and the rule verifier, so
it is a fully wired pipeline that cannot recognise anything — useful precisely
because it tests the wiring on a machine with no GPU. `compose.gpu.yaml`
swaps in YOLOE and the VLM verifier, and documents what it needs (an nvidia
container runtime, and an Ollama on the host serving `VMA_VLM_MODEL`).

### Verification does not run on the frame loop

A VLM call takes ~20s. The relay hands frames in one at a time and the
gateway's hub keeps a *single* latest-frame slot per subscriber, so a verifier
awaited inline would not slow the stream — it would make the gateway discard
20s of frames and leave the stability machine reading a gap it cannot see.

So `video_frame` proposes a candidate and returns; `verify/pending.py` runs the
verifier on a worker task. Two things follow. `/v1/status` reports
`verification.pending` and `verification.dropped` — the second must be zero,
because each drop is a real event that was seen, proposed, and never recorded.
And anything that needs the pipeline to have finished thinking must
`await pipeline.drain()`; `scripts/replay_clip.py` does this before printing,
or it would reach the end of a clip and report the in-flight candidate as
never confirmed.

## Current status: the pipeline and the detector are both real

Every stage between the relay and Memory is built, wired, and tested:
relay consumer → detector → tracker → background-motion estimator → the
stability machine → the evidence ring → the verifier → `MemoryEmitter`.

**`VMA_DETECTOR_KIND` selects the detector: `fixture` (default) or `yoloe`.**
`fixture` runs an empty, looping script against the live relay — finding
nothing, but proving every other piece of plumbing works, which is an honest
state to ship rather than a placeholder lie; it needs no GPU and no extra
install, and is what `ci` runs. `yoloe` (`detect/yoloe.py`) is the real
detector — two warm YOLOE checkpoints, text-prompt for known targets and
prompt-free for open vocabulary — and needs the `models` extra (`uv sync
--extra models`); see `model-manifest.toml` for the pinned checkpoint,
source, and runtime. The checkpoint's in-process masks also feed the optional
personal-object identity path without changing the `Detection` wire contract.

**The device is detected, not assumed:** CUDA, then Apple's MPS, then CPU
(`_select_device`). `dev-macos` therefore runs the real detector too — PyPI
publishes macOS arm64 wheels for the stable torch and torchvision pair.
Linux ARM64 is separately pinned to the exact cu130 nightlies proven on the
GN100 GB10; `standards-exception.toml` records that hardware-only exception.
Expect Apple Silicon to be slower than CUDA; the pipeline drops
stale frames rather than queueing them, so it stays live and simply shows
fewer boxes. `VMA_YOLOE_DEVICE` forces a device when detection picks wrong —
`cpu` is the way back if Metal falls over on an operation it has no kernel
for, which is the failure mode nobody here can rule out.

Running the YOLOE path locally:

```bash
uv sync --frozen --all-groups --extra models
VMA_DETECTOR_KIND=yoloe VMA_DETECTION_LABELS="keys,wallet" uv run uvicorn vision_worker.main:app --port 8082
```

Its own test suite is opt-in (`pytest -m models`) since it needs the extra,
a cached checkpoint, and ideally a GPU — see `tests/test_detect_yoloe.py`.

**`VMA_IDENTITY_KIND` selects personal identity: `none` (default), `fixture`,
or `radio`.** Identity resolves once from three quality track frames, caches the
result for that track, and only annotates events; an unavailable or unmatched
gallery never suppresses an ordinary observation. `radio` uses the pinned 653M
C-RADIOv4-H masked summary/spatial vectors. Its model revision is part of the
`embedder_id`, so objects enrolled with SO400M need fresh H-model views before
they can match. Gallery snapshots refresh from Memory every 30 seconds
and retain their last-known-good version across a temporary outage. Registration
arms a six-second `EvidenceRing` window, rejects weak footage relative to that
window's own sharpness median, and stores 2–4 farthest-point-selected views.
No clip endpoint or ffmpeg process is involved. The `identity` block at
`/v1/status` reports resolved/ambiguous/unmatched/escalated, latency, and gallery
counts; `registration` reports attempts and terminal outcomes.

```bash
VMA_DETECTOR_KIND=yoloe VMA_IDENTITY_KIND=radio \
VMA_DETECTION_LABELS="keys" uv run uvicorn vision_worker.main:app --port 8082
```

**`VMA_DEPTH_KIND` selects the depth adapter: `none` (default), `fixture`,
`yolo`, or `moge`.** `yolo` uses the pinned YOLO26 metric-depth checkpoint and
is the constrained-laptop live-overlay profile: it shares the Ultralytics
runtime already loaded for YOLOE and was measured coexisting with Speech on
the 8 GB development GPU. `moge` remains the higher-quality geometry adapter. `none` is not a degraded state — it is this pipeline's original
shape, image-space stability with `depth_m=None` on every candidate. `moge`
(`depth/moge.py`) is the real adapter — one warm MoGe-2 ViT-L checkpoint,
`Ruicheng/moge-2-vitl-normal` — and, unlike the detector, its `initialize()`
catches its own load failure and degrades to `none` rather than crashing
startup: `domain/stability.py`'s image-space path works with no depth at
all. It runs at low cadence, once per candidate about to be proposed, not
once per frame — see `pipeline.py`'s and `depth/moge.py`'s module
docstrings for why. `fixture` scripts a constant range for exercising the
wiring with no GPU. Its test suite is likewise opt-in (`pytest -m models`)
— see `tests/test_depth_moge.py`.

For the current 8 GB WSL2 laptop, enable real boxes and per-object metric depth
in the full launcher with:

```bash
VMA_ENABLE_CONSTRAINED_VISION=true ./scripts/dev_stack.sh
```

The Admin Console scopes `WS /v1/overlay` to the selected Gateway session, so
boxes from another publisher can never be drawn over the glasses video.

## How it works

**Detection and identity assignment are separate stages.** `detect/` decides
"is there a `keys`-shaped box in this frame"; `track/` decides "is this the
same object as the last frame's." The default tracker
(`track/greedy_iou.py`) is plain Python — greedy IoU matching across
consecutive frames, no model, no `numpy`, no `scipy` — deliberately: coupling
tracking to a detector's own `.track()` call (the common pattern) would leave
the no-model path with a detector but no way to turn detections into
identity. It has no motion model, so it is not the answer for a head-worn
camera, where a stationary object can jump most of the frame because the head
moved; `docs/spikes/tracker-benchmark/RESULTS.md` records why BoT-SORT is
still worth building for the YOLOE path and why this benchmark cannot decide
it.

**Motion is read from the frame, not assumed from a hand.** This service
never detects hands. `pose/image_motion.py` estimates background ego-motion
via phase correlation — an FFT-based global-translation estimate between
consecutive frames — and the stability machine compares an object's own
screen motion against it: a held object stays roughly fixed in the frame
while the background sweeps past; a resting object's motion tracks the
background's, since both come from head movement alone. No OpenCV, no torch.

**A track never promotes to `placed` on its first stable sighting.** An
object that has always been sitting somewhere looks, for exactly one frame,
identical to an object that was just placed. A track that visibly moved and
then settled needs only a short dwell period to promote — motion-then-settle
is strong evidence a placement genuinely happened — but a track that is
stable from its very first sample needs a much longer sustained stillness
before this service is willing to claim it, precisely because it never saw
the placement itself.

**A sighting is not an observation.** The state machine reports every first
sighting as `observed`, because describing what it saw is its job. The
pipeline then logs it and drops it: `observed` is never verified, never
encoded into a clip, and never written to Memory. The reducer over there
does not create a placement from one, so promoting it would spend an encode
and two uploads to move a pointer — and with prompt-free detection, once for
every object that enters frame. Plan criterion: "object visible, never
touched" and "walking past an object" must produce **zero** observations.
They are visible in `/v1/events` with outcome `not_promoted`.

**Thresholds are durations, converted to frame counts at a configured
rate.** `VMA_DWELL_SECONDS` and friends are what an operator sets;
`StabilityConfig.from_durations` turns them into the frame counts the state
machine counts, at `VMA_SOURCE_FPS`. That rate is **the gateway's
`VMA_SAMPLE_FPS`, not the glasses' capture rate** — the relay delivers a
sampled stream, at 8fps by default rather than the 24 the glasses capture.
Setting it wrong does not fail: every threshold silently comes to mean a
different duration. So the
pipeline measures the rate it is really being fed, `/v1/status` reports
`configured_fps` and `observed_fps` side by side, and a material disagreement
logs a warning. `compose.yaml` drives both services from one variable.

**Evidence is a shared, time-scoped ring, not a per-object recording.**
`evidence/ring.py` buffers already-sampled frames — never raw media, the
Media Gateway's `raw_buffer_seconds = 0` commitment is about what *it*
retains, not what it relays — and a candidate's window slices a temporal
range out of it by timestamp. On confirmation, `evidence/clip.py` encodes
that window into a short mp4 via PyAV (CPU, no GPU) alongside a still-frame
fallback, so a clip-encode failure degrades gracefully rather than losing
the observation.

**"Did it move?" is answered in the room, not on the screen.** The image-space
signal above cannot separate a carried object from a panning head on handheld
footage — measured on `media/clips`, no threshold does, and the failure is a
`picked_up` fired at keys that never left the desk. `VMA_VERIFIER_KIND=world_
motion` adds `verify/world_motion.py`, which reconstructs the candidate's own
window through `pose/da3.py` (Depth Anything 3: depth *and* camera pose),
back-projects the object through `domain/geometry.py`, and vetoes a verdict
its world trajectory contradicts. On the same footage that produced three
false pickups, the world position drifts 0.4% of scene scale.

Two constraints shape it. DA3 reconstructs a *window* jointly — ~283ms per
view, and its scale is fixed per inference call, so world points from two
calls cannot be compared — which is why this is a candidate-window check
rather than a per-frame pose source. And `depth-anything-3` requires
`numpy<2` while this workspace pins `numpy>=2.4.6`, so it is **not a declared
dependency**: install it in a separate environment with the same
platform's locked torch pin. Wherever it is absent, the verifier degrades to the rules
alone and `/v1/status` reports which one is really running.

**Verification is deterministic by default, and that default is not a
placeholder.** `docs/09-Spike-Plan.md`'s S04 spike names exactly this as its
own stop condition: if a model-based verifier fails, "substitute... a
conservative rule-based candidate generator." `verify/rules.py` is that
fallback, built first rather than as a contingency, so a model setback costs
a config swap. Person 2 replaces this module and nothing else changes.

## Boundary decisions

- **`Location` may have null `room`/`surface` for a `placed` observation.**
  A confirmed candidate now carries `depth_m` (and optionally `box3d`) when
  `VMA_DEPTH_KIND=moge` — real metric geometry, in *camera* space. That is
  not yet a room or a surface: turning a camera-space range into a *world*
  position needs a capture pose (`domain/geometry.py`'s `compute_world_
  point`), and nothing in this service can produce one until task #46's
  `DevicePose` reads real ARDK 6DoF pose off the relay's data channel. Until
  then, `Location` with all fields null is still a valid, honest object per
  memory-contract's own contract — an "I confirmed it was placed, but not
  exactly where" rather than a wrong or a withheld answer.
- **One `Tracker`/`PoseSource`/`TrackRegistry`/`EvidenceRing` per process, not
  per epoch.** All reset together on every `epoch_started`. This does not
  support two sessions publishing simultaneously; a demo runs one glasses
  stream at a time.
- **`--workers 1` is load-bearing** (see the Dockerfile): `Pipeline` holds
  one relay connection and all of one epoch's state as a singleton. A second
  worker would independently process the same stream and double-write
  Memory.
