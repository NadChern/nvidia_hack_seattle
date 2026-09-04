# Grounder bake-off — results

**Status: not yet run.** The event half is now *runnable* — four recordings are
annotated, three of them machine drafts awaiting review (see Annotation status).
The decision rules below are written *before* the numbers exist, on purpose. A
rule chosen after seeing the results is not a rule.

## Decision rule (fixed in advance)

There are two tasks — grounding (*where is it*) and events (*what happened to
it*) — and the pipeline uses one model for both. An arm must clear the
grounding gates to be a candidate at all, then clear the event gates to be the
pick.

### Grounding gates

An arm is a candidate only if it passes all four, in order:

1. **Shippable licence** — open weights we may actually deploy. Apache-2.0
   qualifies; so does LFM's revenue-capped licence, by owner decision
   (2026-09-02): the cap is distant and a grounder is a replaceable component,
   so the lock-in is cheap. Moondream 3 still fails — BSL 1.1's "No Third-Party
   Service" grant forbids offering it as a service at any size, which is the
   product, so no amount of growth fixes it.
2. **Latency p50 < 7 s** at 768 px, one image. That is
   `reason_interval_seconds`; a slower arm backs the window queue up and is
   disqualified for realtime video regardless of accuracy.
3. **Mean IoU ≥ 0.55**, i.e. at least the incumbent Cosmos measured on real
   footage. An arm that grounds worse than what we already ship is not a
   candidate.
4. **Then** lowest VRAM, and — as a tie-breaker only — vLLM over SGLang, since
   Nemotron already runs on vLLM and a second runtime is a standing maintenance
   cost.

If no arm passes 2 and 3, that is a real result: it means the product keeps
Cosmos and pays 32 GiB, or the pipeline changes shape. Record it, don't retune
until something passes.

**If LFM wins, record what the licence costs**, so the trade stays visible rather
than becoming an unexamined default: the pitch says "open weights, commercially
licensed above $10M" instead of "fully open source", and the decision is due for
review when revenue comes within a year of the cap. If a clean-licence arm lands
within ~0.03 mean IoU of it, prefer the clean one — that gap is inside the noise
this 36-image set can resolve, and it buys back the claim for nothing.

### Event gates

Also fixed before the numbers exist. A candidate wins on events if:

1. **Placement recall ≥ 0.8 per event.** Not per window: windows overlap by 13 s
   and `event_cooldown_seconds` means the pipeline writes the placement once, so
   one hit among the covering windows is a success. Missing one placement in
   five is the most a "where did I put my keys" product can absorb.
2. **False-placement rate ≤ 0.05** on windows where truth says nothing
   happened. This is the asymmetric one and it dominates the others: a missed
   placement leaves memory silent, but a false placement tells the user
   something confidently wrong, and confidently wrong is what loses a customer
   in discovery. An arm above 0.05 here loses to an arm with worse recall.
3. **Latency p50 < 7 s on the eight-frame call.** The single-image number from
   the grounding axis does not transfer — eight images is roughly eight times
   the prefill, and this is the call the pipeline actually makes every 7 s.
4. **Then** false-handling rate, which does not gate the winner but decides a
   separate question: if it is below ~0.05, `promote_motion_events` can be
   turned on and the product gains a movement timeline instead of only
   last-known-location. That switch has been off since the hackathon on an
   impression; this is the number that settles it.

Window accuracy and macro-F1 are **diagnostics, not gates**. An arm that reports
a placement in the first covering window and stays quiet in the next two scores
poorly on both and is behaving exactly as designed.

If every arm fails gate 2, the honest conclusion is that no open grounder can be
trusted to classify events at ~1 fps, and the pipeline needs a different shape —
motion cues from the tracker, or a higher frame rate into the reasoner — rather
than a different model. Record that; it is a more useful finding than a winner.

## Summary — grounding

Fill from `runs/<arm>.json`, not from terminal scrollback.

| Arm | Licence | Ship? | mean IoU | median | ≥0.5 | no box | containment | area ratio | p50 s | p95 s | VRAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl-4b | Apache-2.0 | ✅ | | | | | | | | | |
| qwen3-vl-8b | Apache-2.0 | ✅ | | | | | | | | | |
| moss-vl-realtime | Apache-2.0 | ✅ | | | | | | | | | |
| cosmos3-nano *(incumbent)* | NVIDIA OML | ~ | | | | | | | | | |
| lfm2.5-vl-3b | LFM, capped >$10M | ✅ | | | | | | | | | |

Prior knowledge to compare against, both measured on this same probe set:
LFM2.5-VL-3B **0.889** (0.92 with the extent rule, spike 12c); Cosmos3-Nano
**~0.55** with excursions to 0.05 on real footage.

Numbers taken before 2026-09-02 used an image-first request; the harness now
sends text first, matching `CosmosReasoner`. Re-run rather than mixing the two.

## Summary — events

Fill from `runs/<arm>.events.json`. **Blocked on annotation** — see below.

| Arm | placements found | recall | false placed | false handling | delay p50 s | window acc | macro-F1 | p50 s (8 frames) | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-vl-4b | | | | | | | | | |
| qwen3-vl-8b | | | | | | | | | |
| moss-vl-realtime | | | | | | | | | |
| cosmos3-nano *(incumbent)* | | | | | | | | | |
| lfm2.5-vl-3b | | | | | | | | | |

Read the `locations` array in each run file by eye while filling this in. A
model that grounds well, times the event right, and describes the wrong surface
is not shippable either, and no column here would catch it.

## Annotation status

**Four annotated, one of them confirmed by hand.** `clips/recordings/` holds
twelve files but only six distinct scenes, two of them also present as
gateway-quality (`_hi`, 720p/10 fps) and relay-quality (`_relay`, 360p/6 fps)
re-encodes. Prefer the re-encodes: the deployment never sees the 4 K original,
and an event score on pristine footage measures a call that is never made.

| Clip | Duration | Contents | Truth |
|---|---|---|---|
| VID_20260819_120633 | 19.8 s | **enrollment** — keys rotated in hand, no placement | not wanted |
| VID_20260819_120701 | 18.5 s | **enrollment** — wallet rotated in hand, no placement | not wanted |
| VID_20260819_120727_hi | 27.8 s | keys placed on the desk t≈7.1, walk to kitchen, return t≈20 | **draft** |
| VID_20260819_120802_hi | 28.1 s | wallet placed on the desk t≈4.5, walk to kitchen, return t≈19.5 | **confirmed** |
| VID_20260819_160630_hi | 46.5 s | keys placed on a desk holding a second keyring and a lanyard, left there | **draft** |
| VID_20260819_162151_hi | 61.0 s | keys placed on the desk t≈11.5, retrieved t≈42.5, placed on the hall console t≈49 | **draft** |
| `_relay` / `_win` / 4 K variants | — | re-encodes of scenes already covered | later |

`120802_hi` was machine-drafted and then **corrected by hand in the annotator**
(2026-09-02), which is where the visibility bug below was found. The other three
were drafted the same way on 2026-09-03 and have **not** been through the
annotator yet; each one's `notes` field names the specific calls a reviewer
should check, and every one of them is a visibility boundary or an object-identity
question rather than a box.

### What the corpus can now measure

24 windows across the four clips, at the production schedule (20 s span every
7 s, 8 frames):

| Clip | Windows | Object absent | Carried | Catchable placement | **Quiet** |
|---|---|---|---|---|---|
| 120727_hi | 4 | 2 | 0 | 2 | 0 |
| 120802_hi | 4 | 2 | 0 | 2 | 0 |
| 160630_hi | 7 | 2 | 1 | 1 | **3** |
| 162151_hi | 9 | 2 | 1 | 4 | **2** |
| **total** | **24** | **8** | **2** | **9** | **5** |

Five distinct placements, every one of them catchable in at least one window, so
gate 1 (placement recall ≥ 0.8 *per event*) has five events to score — thin, but
it is a number. Gate 3's latency comes free from the same 24 calls per arm, 120
across five arms.

One of the nine catchable windows is a coin-flip: `120727_hi` has `t_start` 0.1,
which puts its windows on 7.1/14.1/21.1 and its placement at *exactly* t=7.1,
inside the first window by the width of the `start < t <= end` comparison. The
keys are at rest on the desk at 7.1 and the hand is clear by 7.6; annotating the
later moment would drop that window. Whichever a reviewer picks, pick it the same
way in every file — the rule the drafts use is **`placed` is the first frame the
object rests on the surface**, hand still there or not.

The eight *absent* windows are not filler either: they are the only place a
phantom box is scored, and the wallet clip showed an arm can score better by
hallucinating in them if visibility is annotated wrong.

### Two of the twelve are the wrong kind of footage

`120633` and `120701` are **enrollment** clips — the object held up and rotated
in front of the camera for the whole runtime. They belong to
`eval_registration.py`; they contain no placement and cannot contribute a single
event. The `_relay`, `_win` and 4 K variants of `120727`/`120802` are re-encodes
of scenes already covered, so annotating them by hand is not new evidence about
a model, it is a resolution ablation — and one that should be done by
transferring truth across encodes on the time axis, which is a tool feature
nobody has written, rather than by annotating the same scene three times.

### Gate 2 is measurable, and cannot be passed

An earlier version of this file said gate 2 needed footage that does not exist —
"at least 45 seconds" of the object sitting in frame untouched. **That was
wrong**, and annotating `160630` and `162151` is what showed it. A window is
quiet if **no event falls in its 20 s span** *and* the object is in shot **in its
last frame**. The object does not have to be visible for the whole span. So
every time the wearer sets something down, does other things for 20 s, and
happens to look back at it, that is a quiet window — and looking at it for one
more frame 7 s later is another.

`160630` yields three that way and `162151` two. The two 28 s clips yield none,
for a different reason: they are too short for any window to get clear of their
own placement.

Five is measurable and it is still not enough, but the arithmetic is worth being
precise about, because it decides how much footage to shoot:

- Gate 2 is a **rate ≤ 0.05**. With five samples the finest rate the corpus can
  express is 0.2. One false placement fails the gate outright.
- Zero false placements in five proves nothing: the 95% upper bound on 0/5 is
  **≈0.45**, nine times the gate.
- To put the upper bound *under* 0.05 on a clean run takes **≈60 quiet
  windows** (1 − 0.05^(1/60) = 0.049).

So gate 2 as written can be **failed** on today's corpus but not **passed**, and
an arm that comes through clean should be reported as "no false placements in 5
windows", never as "passes gate 2".

The recording brief changes shape accordingly. It is not one long static shot:

> Put objects down where they stay, then **keep coming back to them**. Sit at
> the desk and work, glance at the shelf, walk past the console, cook — with two
> or three enrolled objects visible somewhere in the flat and nothing being
> placed or picked up. Every 7 s in which an object is in frame and untouched is
> a window; ~8 minutes of that kind of footage across a few sessions is worth
> more than one long clip, because the frames differ.

Ordinary "living in the flat" recordings do it, and they are much cheaper to
shoot than staged placements — which is the opposite of what this corpus is
made of, and the reason gate 2 is the number missing. The shot list, the
re-encode recipe and the two annotation decisions it forces are in
[RECORDING.md](RECORDING.md).

## What the first annotated clip already shows

Four things fell out of annotating one 28 s clip, before any model has been
run. All four are about the corpus and the harness, not about any arm.

1. **Everything decodes sideways.** The originals carry `rotation=-90` and PyAV
   ignores rotation metadata, so every `av.open` path here — the annotator, the
   event axis, and `media-gateway`'s virtual-glasses `--file` publisher — reads
   these portrait recordings on their side. Replaying a recording into the
   pipeline therefore hands the reasoner a rotated world. The annotator now
   applies and records `rotate` (90 for these files) so the harness matches it,
   but **`publisher/sources.py` still does not**, and that is a live bug in the
   dev/demo path rather than a harness detail. The gateway's own log line says
   the live relay arrives portrait, so this is likely replay-only — worth
   confirming from a session log before assuming it.
2. **Two windows out of four could catch the placement**, and which two is not
   obvious. Boxes go in a window's last frame and no box means no event, so a
   window whose last frame no longer shows the object cannot report it. The
   wearer turns away ~3 s after setting the wallet down (so the window ending at
   t=14 is blind) but walks back to the same desk, which makes the window ending
   at t=21 catchable again — 16 s after the event, and only because a 20 s span
   still reaches it. Recall is scored over catchable placements for that reason.
   If this pattern holds the pipeline gets one or two shots per placement, which
   makes `reason_interval_seconds` and `reason_window_seconds` sharper knobs
   than they look.
3. **Visibility has to be annotated, not derived.** The first version of the
   scorer took the object's in-shot span to be its first boxed frame through its
   last, which is right for a clip where the object appears once and wrong for
   this one: the wearer leaves for eleven seconds and returns. Under the derived
   rule the whole kitchen stretch counted as "the object was there", and truth
   interpolated a box along the straight line between the two desk sightings —
   so a scripted arm that hallucinated a placement in the kitchen scored
   **window accuracy 0.75 and mean IoU 0.37 with 1/1 phantom**, where against
   the corrected file it scores 0.50 and 1.00 with 2/2 phantom. It was being
   rewarded for hallucinating. Truth files now carry an explicit `visibility`
   list of frame ranges and the annotator has a key for it. Worth stating
   plainly: a wrong visibility model does not make the score noisier, it makes
   it answer a different question, and nothing in the output looks wrong.

4. **A placed object sits below the identity floor at gateway resolution.** The
   wallet measures 241–255 px on target while held up to the camera and
   **63–100 px** once it is on the desk, against a ~128 px floor and ~48 px
   chance level (docs/spikes/capture-resolution). Grounding it is one problem;
   confirming *whose* wallet it is from the placed frame is a different and
   harder one, and no arm in this bake-off can fix it.

## How to read containment against IoU

| containment | IoU | Meaning | Fix |
|---|---|---|---|
| ~1.0 | low | Right object, box too tight — the extent failure | prompt wording |
| ~1.0 | high | Correct | — |
| low | low | Wrong object | model, not prompt |

Spike 3c is why this column exists: containment was ~100% while identity F1 sat
at 0.776, and IoU alone could not tell those apart. An arm failing only on
extent is not a bad model, and swapping it out would be the wrong conclusion.

## Open questions this run does NOT answer

- **Location-phrase quality.** Recorded per window, scored by nobody. Judging
  "on the kitchen table next to a mug" needs a human who watched the clip, and
  an earlier attempt to score it automatically marked a *correct* phrase as a
  hallucination because the wide frame really did show that surface. It flows
  straight into `Location.surface` in memory, so a wrong phrase is user-visible;
  read the `locations` arrays before signing off on a winner.
- **Whether 20 s is the right window.** The event axis takes
  `--window-seconds` / `--max-frames`, and spike 5b only bracketed the minimum
  between a 10 s window that failed and a 28 s clip that passed. Re-running one
  arm at 10/14/20/28 s is now cheap and would replace a bracket with a curve —
  but do it *after* the arm is chosen, or the comparison stops being one.
- **MOSS-VL's native streaming mode.** Scored here on the same single-image
  protocol as everyone else for comparability, which under-sells it — its
  cross-attention design keeps watching while generating, and it ships a
  WebSocket interface taking external JPEG frames. If it places well on the
  single-image axis, the continuous-mode run is worth doing before deciding.
- **Identity F1 downstream.** IoU is a proxy. The thing that matters is whether
  the crop feeds C-RADIO a recognisable object; `spike_extent.py` +
  `probe_identity.py` close that loop.
