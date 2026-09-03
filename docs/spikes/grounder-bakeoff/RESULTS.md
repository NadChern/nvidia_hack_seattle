# Grounder bake-off — results

**Status: not yet run**, and the event half is not yet *runnable* — it needs the
recordings annotated first (see Annotation status). The decision rules below are
written *before* the numbers exist, on purpose. A rule chosen after seeing the
results is not a rule.

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

**One of twelve done, three worth doing, and one clip still missing.**
`clips/recordings/` holds twelve files but only six distinct scenes, two of them
also present as gateway-quality (`_hi`, 720p/10 fps) and relay-quality
(`_relay`, 360p/6 fps) re-encodes. Prefer the re-encodes: the deployment never
sees the 4 K original, and an event score on pristine footage measures a call
that is never made.

| Clip | Duration | Contents | Annotate? |
|---|---|---|---|
| VID_20260819_120633 | 19.8 s | **enrollment** — keys rotated in hand, no placement | no |
| VID_20260819_120701 | 18.5 s | **enrollment** — wallet rotated in hand, no placement | no |
| VID_20260819_120727 (+ `_hi` `_relay` `_win`) | 27.8 s | keys placed on the desk t≈4.7, walk to kitchen, return t≈20 | **yes** (`_hi`) |
| **VID_20260819_120802_hi** | 28.1 s | wallet placed on the desk t≈4.5, walk to kitchen, return t≈19.5 | **done** |
| VID_20260819_120802 (+ `_relay` `_win`) | 28.5 s | same scene, other encodes | later |
| VID_20260819_160630 | 47.0 s | multi-object — keys and wallet, several placements and pickups across two rooms | **yes** |
| VID_20260819_162151 | 61.7 s | keys placed on the desk, then on a hall console t≈49.4, held in shot to the end | **yes** |

The one finished annotation was machine-drafted and then **corrected by hand in
the annotator** (2026-09-02), which is where the visibility bug below was found.
Its two remaining soft spots: the second in-shot range has boxes only from frame
220, so the window ending at t=21 can be scored for its event but not for IoU;
and its range boundaries sit on the 5-frame annotation grid, so each is accurate
to ±0.5 s.

At 20 s windows firing every 7 s, that is roughly 4 windows per short clip and 8
for the longest — call it 35 model calls per arm, 175 across five arms. Cheap
enough to re-run when a prompt changes, which is the point.

### Two of the twelve are the wrong kind of footage

`120633` and `120701` are **enrollment** clips — the object held up and rotated
in front of the camera for the whole runtime. They belong to
`eval_registration.py`; they contain no placement and cannot contribute a single
event. The `_relay`, `_win` and 4 K variants of `120727`/`120802` are re-encodes
of scenes already covered, so annotating them by hand is not new evidence about
a model, it is a resolution ablation — and one that should be done by
transferring truth across encodes on the time axis, which is a tool feature
nobody has written, rather than by annotating the same scene three times.

That leaves **three** worth the afternoon: `120727_hi`, `160630`, `162151`.

### Gate 2 needs footage that does not exist yet

Gate 2 — the one that decides the winner — is measured **only** on windows where
the object is present and truth says nothing happened. A window's span is
`reason_window_seconds` long, so it is quiet only if **no event falls in the
preceding 20 s** and the object is still in its last frame. That is a much
stronger requirement than "a stretch where nothing happens", and it is the
reason the wallet clip yields zero quiet windows even after a careful
annotation: the wallet is back in shot from t=19.5, but the window ending at
t=21 reaches back to t=1 and still contains the placement, and the next ends at
t=28 with the wearer facing the window.

Measured across the corpus, the longest event-free stretch with the object in
shot is **≈12 s** — `162151`, keys on the hall console from t≈49.4 to the end at
61.7. `160630`'s best is ≈4 s. Every recording was shot as "here is a thing
happening", which is the natural way to shoot a demo and the wrong way to
measure a false-positive rate.

So annotating all three remaining clips raises the placement count from 1 to
perhaps six or eight — enough for gate 1 to mean something — and leaves gate 2
at **0/0**, unmeasurable. That takes one new recording, and the brief is
specific:

> Put the object down somewhere it stays. Then keep it in frame, doing nothing
> to it, for **at least 45 seconds** — sit at the desk, read, look around the
> room, let the object drift in and out of the centre. No hands on it, no other
> objects placed or picked up. Two or three minutes of that yields five or six
> quiet windows; a 30 s clip yields one.

Until that exists, an arm can clear gates 1 and 3 and the bake-off still cannot
say whether it hallucinates placements — which is the failure the product
cannot absorb, so "we ran it and it looked fine" would be the wrong conclusion
to draw from a run on today's corpus.

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
