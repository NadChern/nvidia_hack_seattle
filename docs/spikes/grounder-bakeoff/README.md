# Grounder bake-off — which open-source model should ship?

**Question.** The hackathon is over. In customer discovery the grounder choice is
no longer "what did we demo with" but "what can we ship, and is it fast enough
for realtime video". Those are two different filters and the incumbent passes
neither cleanly.

**The constraint that decides it:** the model must be **open source**, because
the privacy-first story is the product. That single requirement removes the two
best-grounding candidates from contention.

| Model | Params | Licence | Shippable |
|---|---|---|---|
| Qwen3-VL-4B-Instruct | 4B | Apache-2.0 | ✅ |
| Qwen3-VL-8B-Instruct | 9B | Apache-2.0 | ✅ |
| MOSS-VL-Realtime | 11.3B | Apache-2.0 | ✅ |
| Cosmos3-Nano *(incumbent)* | 16B | NVIDIA Open Model License | ~ permissive, not Apache |
| LFM2.5-VL-3B | 3B | LFM Open License — free below $10M revenue | ✅ capped |
| ~~Moondream 3~~ | 9B MoE | **BSL 1.1**, "No Third-Party Service" | ❌ excluded |

LFM2.5-VL-3B is the current measured best (IoU 0.889, **0.92** with the extent
rule, spike 12c) and the smallest arm at 6.02 GiB resident — if it wins, the
deployment card drops with it, potentially to a 48 GB L40S at ~$1.55/hr.

Its licence caps commercial use above $10M annual revenue. **Owner decision,
2026-09-02: contend anyway.** The threshold is distant, and a grounder is a
replaceable component on roughly a six-month cycle — by the time the cap binds,
its replacement probably has not been published yet, and switching cost is a
bake-off re-run rather than a rewrite. What the choice costs is the *claim*, not
the code: "fully open source" becomes "open weights, commercially licensed above
$10M" if a customer asks. Due for review when revenue is within a year of the cap.

**Moondream 3 remains excluded, and it is not the same case.** BSL 1.1's "No
Third-Party Service" grant is not a revenue threshold to grow into — it forbids
offering the model as a service at *any* size, which is exactly what this product
is. No amount of growth makes it legal, so unlike LFM there is no trade to weigh.
[vLLM also does not support the architecture](https://github.com/vllm-project/vllm/issues/25215).

## Two tasks, because the reasoner does two things

The pipeline asks one model both *where is it* and *what just happened to it*,
and an arm can be excellent at one and useless at the other. Both are scored.

```bash
--task grounding   # one still image, one box       (36 annotated images)
--task events      # a 20 s window, 8 frames        (annotated recordings)
```

### Grounding axes

Grounding IoU alone already picked the wrong model once here. In spike 3c
containment sat at ~100% while identity F1 was 0.776: the box was on the right
object at the wrong *extent*, and IoU could not say so. Every arm reports:

| Axis | Measures | Gate |
|---|---|---|
| **IoU** | box vs 36 human annotations | ≥ 0.5 for a usable C-RADIO crop |
| **Containment + area ratio** | wrong object *vs* wrong extent | containment ≈ 1 with low IoU ⇒ prompt problem, not model problem |
| **Latency p50/p95** | per grounding call | **< 7 s** = `reason_interval_seconds`, or the pipeline backs up |
| **No-box rate** | silent declines | a decline reads downstream as "object absent" |

### Event axes

Only `placed` reaches memory today: `promote_motion_events` is `False` because
"at ~1fps Cosmos hallucinates handling on a resting object, and a single false
pickup flips a confirmed placement to *moved afterward*". That policy has never
been measured — it rests on an impression from the hackathon. This axis is what
turns it into a number.

| Axis | Measures | Gate |
|---|---|---|
| **Placement recall (per event)** | did *any* window covering a real placement call it `placed` | the product's number: windows overlap, one hit is enough |
| **False-placement rate** | `placed` on a window where nothing happened | each one writes a **wrong location** into memory — the worst failure the product has |
| **False-handling rate** | `picked_up`/`carried` on a resting object | the specific number that decides whether `promote_motion_events` can be turned on |
| **Detection delay** | window end − event time | how long after a placement the memory exists |
| **Phantom-box rate** | a box on a window where the object has left the frame | the model inventing the object it was told to look for |
| **Latency p50/p95** | per **8-frame** call | the real realtime figure; single-image latency flatters every arm |
| Window accuracy / macro-F1 | all five actions | diagnostic, not a gate — see below |

### A placement is only catchable while the object is still in frame

Boxes go in the window's **last** frame, and `CosmosReasoner._parse` returns no
events at all when the reply carries no box. So a window whose last frame no
longer shows the object cannot report what happened in it — however clearly its
earlier frames showed the placement. Recall is therefore computed over
*catchable* placements only, and the harness prints how many were excluded,
because that number is a fact about the corpus and the window geometry rather
than about any model.

On the first annotated clip (a wallet placed on a desk, then the wearer walks
away) it is **one window out of four**. If that holds across the corpus it is a
finding in its own right: at a 7 s interval, the pipeline gets roughly one shot
at each placement, and `reason_interval_seconds` matters far more than it looks.

Per-window accuracy is deliberately *not* a gate. Windows overlap by 13 s, so a
single placement is offered to three of them and the pipeline needs only the
first (`event_cooldown_seconds` suppresses the rest). An arm that reports the
placement once and then goes quiet scores badly per-window and is behaving
exactly as wanted.

**Location phrases are recorded and never scored.** Judging "on the kitchen
table next to a mug" needs a human who watched the clip; scoring it
automatically once marked a correct phrase as a hallucination, because the wide
frame really did show that surface. Read them out of `runs/*.json` by eye.

Latency is the axis "realtime" actually lives on, and it is the one the
incumbent is weakest on (~5 s warm on a *single* image, against a 7 s window
interval — the event axis sends eight).

## Data

**Grounding:**

- **Images:** `clips/identity-probe/` — 36 HEIC stills, keys_1 (15), keys_2 (14),
  keys_3 (7), across `reference` / `query_clean` / `query_realistic`.
- **Ground truth:** `clips/spike1-arms/_cache/` — 36 human-annotated boxes, 1:1
  with the images. Not model output; these are the annotations spike 1 built.

Do **not** score against `clips/extent-*/_boxes/` — those are *predicted* boxes
saved from an earlier run, and scoring a model against another model's output
measures agreement, not accuracy.

**Events:**

- **Footage:** `clips/recordings/` — twelve files, six distinct scenes. The four
  `_hi` / `_relay` / `_win` files are the same two scenes re-encoded at what the
  gateway and relay actually deliver (720p/10 fps, 360p/6 fps), which makes them
  the *more* representative ones to annotate: the deployment never sees the 4 K
  original.
- **Ground truth:** `docs/spikes/grounder-bakeoff/truth/` — **does not exist
  until someone annotates it.** See below; the schema is documented in
  `truth/README.md`.

## Annotating the recordings

The event axis has no data until this is done. `clips/` is gitignored, so
annotations are written into `docs/` and committed.

```bash
cd services/vision-worker
uv run python scripts/annotate_placement.py annotate \
  --clip ../../clips/recordings/VID_20260819_120802_hi.mp4
```

It decodes the clip to JPEGs at ~2 fps, serves a browser annotator on
`127.0.0.1:8770`, and writes `docs/spikes/grounder-bakeoff/truth/<clip>.json` on
save. Drag to box, arrow keys to step, `1`–`5`/the dropdown to mark an event at
the current frame, `s` to save.

**Frames are rotated 90° clockwise on decode (`--rotate`, default 90).** PyAV
ignores rotation metadata, so these portrait recordings decode on their side in
every `av.open` path in this repo — this tool, the event axis, and
`media-gateway`'s virtual-glasses `--file` publisher, which means replaying one
of these files into the pipeline feeds the reasoner a sideways world. The
annotator records what it applied and the scorer applies the same, so the two
can never disagree.

**Box the object on the first and last frame it is visible.** That span *is*
the visibility span: outside it the scorer expects no box and counts one as a
phantom.

Two things it shows that matter more than they look:

- **Interpolated boxes are drawn dashed.** Annotate two anchors, scrub, and stop
  as soon as the dashed box tracks the object — the scorer interpolates the same
  way, so anything more is unpaid work.
- **Pixels on target, live, at capture resolution.** Below ~128 px identity is at
  its floor and by ~48 px it is at chance (docs/spikes/capture-resolution). A
  clip whose keyring never clears the floor should be discovered *now*, not
  after a model scores badly on it for reasons that are not the model's.

Annotate the quiet clips too, marking `nothing_happened`. Half of what this axis
measures — the false-positive rate — is measured only on windows where nothing
happened.

```bash
uv run python scripts/annotate_placement.py check   # validate every annotation
```

## Prompt parity

Every arm gets the **production** enrollment-localize prompt from
`reason/cosmos.py`, extent rule included — not a prompt tuned per model. Per-arm
tuning would measure prompt-engineering effort rather than model quality, and
spike 12c showed one sentence moves the worst noun from IoU 0.12 to 0.92, an
effect big enough to swamp the differences we are looking for.

The event axis gets the production **window** prompt the same way, and
`--check-prompt-drift` now covers three things: the extent rule, the window
prompt, and the action vocabulary. An event score against a stale prompt is
worse than no event score — it looks like a model result and is actually a diff.

The harness inlines those prompts (it must run under `uv --isolated`, where the
service package is not importable) and `--check-prompt-drift` re-reads
`reason/cosmos.py` to prove the copies are current. A drift is fatal, not a
warning.

Parity extends to **block order**: text first, then images, exactly as
`CosmosReasoner` builds its requests. Block order is part of what a VLM sees,
and the harness previously sent the image first on the grounding task — fixed
when the event axis landed, so grounding numbers taken before 2026-09-02 are not
comparable with ones taken after.

## Running it

One arm at a time, one server at a time — so `nvidia-smi` attribution is
unambiguous and no two models contend for the card.

```bash
# 0. once per box
pip install "vllm>=0.23"        # >=0.13 covers Qwen3-VL FP8 + Cosmos; LFM needs >=0.23
pip install "sglang[all]"       # MOSS-VL only
```

### Qwen3-VL 4B / 8B — Apache-2.0, vLLM

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}'
```

### Cosmos3-Nano — the incumbent baseline

```bash
vllm serve nvidia/Cosmos3-Nano --port 8001 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}'
```

BF16 only — there is no official FP8/FP4 path, which is why this arm alone costs
~32 GiB of weights.

### MOSS-VL-Realtime — SGLang, not vLLM

```bash
python -m sglang.launch_server --model-path OpenMOSS-Team/MOSS-VL-Realtime \
  --port 8001 --mem-fraction-static 0.85
```

**Count the runtime against it.** Nemotron already runs on vLLM; an SGLang winner
means the deployment maintains two inference runtimes. Not disqualifying, but it
should have to win clearly, not narrowly.

### Score any HTTP arm — grounding

```bash
cd services/vision-worker && uv run --isolated \
  --with pillow --with pillow-heif --with numpy \
  python scripts/spike_grounder_bakeoff.py --arm qwen3-vl-8b --check-prompt-drift \
    --dataset ../../clips/identity-probe --cache ../../clips/spike1-arms/_cache \
    --out ../../docs/spikes/grounder-bakeoff/runs/qwen3-vl-8b.json
```

`pillow-heif` is required, not optional — all 36 probe images are HEIC.

### Score any HTTP arm — events

Same server, same arm, second task. `av` replaces `pillow-heif` here (video, not
HEIC), and the window geometry defaults to the production values from
`config.py`.

```bash
cd services/vision-worker && uv run --isolated \
  --with pillow --with av --with numpy \
  python scripts/spike_grounder_bakeoff.py --arm qwen3-vl-8b --task events \
    --check-prompt-drift \
    --out ../../docs/spikes/grounder-bakeoff/runs/qwen3-vl-8b.events.json
```

Both `--truth-dir` and `--clips` default to the right places. The serve commands
above all cap images at **8**, which is `reason_max_frames`; a lower cap makes
every window call fail, and a higher one measures a request the pipeline never
sends.

`--window-seconds` / `--interval-seconds` / `--max-frames` default to 20 / 7 / 8,
the shipped values. They are exposed because "is the window wide enough" is
itself unresolved — spike 5b only bracketed the minimum between a 10 s window
that failed and a 28 s clip that passed — and re-running this axis at a
different width is now the cheapest way to answer it. Change them only
deliberately, and never between two arms being compared.

### LFM2.5-VL-3B — vLLM, like the others

```bash
vllm serve LiquidAI/LFM2.5-VL-3B --port 8001 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":8}'
```

**Needs vLLM ≥ 0.23.0** — LFM2.5-VL is native (`Lfm2VlForConditionalGeneration`,
no `--trust-remote-code`) but landed later than the Qwen arms, which only need
≥0.13. Use one vLLM ≥0.23 for every arm so the latency axis is a comparison
rather than a mix of serving stacks.

Earlier grounding spikes ran LFM under `uv --isolated` with `transformers>=5.0`,
because the service pins 4.57.6 for C-RADIOv4. That is no longer necessary here:
serving it puts the dependency split behind an HTTP boundary, which is also how
the deployment would run it.

## Reading the result

The winner is **not** simply the highest IoU, and it is not decided on grounding
alone — an arm that grounds best and hallucinates a pickup on every resting
object is not shippable, because the product's promise is knowing where things
are.

1. **Shippable arms only** — every arm in the table now qualifies except
   Moondream 3, which is not scored. If LFM wins, note in RESULTS.md what its
   licence costs the pitch, so the trade stays a decision rather than drifting
   into an unexamined default.
2. **Latency p50 must be under 7 s.** An arm that grounds beautifully at 12 s
   cannot hold the window interval and is disqualified for realtime video.
3. **Containment ≈ 1 with low IoU is a prompt result, not a model result** — that
   arm is grounding correctly and cropping wrong, and the fix is wording.
4. **Then** VRAM, because it picks the card and the card picks the bill.

A per-arm JSON lands in `runs/`. Fill the summary table in `RESULTS.md` from
those files rather than from terminal scrollback.

## Coordinate conventions are measured, never assumed

Every arm defaults to `--coord-order auto`, which scores the *same* replies under
both xyxy and yxyx and reports which the model actually uses. Guessing this wrong
once already made Cosmos look far worse than it was. Cosmos is pinned to `xyxy`
because it self-reported the convention and it was confirmed.
