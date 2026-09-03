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

**Nothing is annotated yet.** `clips/recordings/` holds twelve files — six
distinct scenes, two of them also present as gateway-quality (`_hi`, 720p/10 fps)
and relay-quality (`_relay`, 360p/6 fps) re-encodes. Prefer the re-encodes: the
deployment never sees the 4 K original, and an event score on pristine footage
measures a call that is never made.

| Clip | Duration | Resolution | Annotated |
|---|---|---|---|
| VID_20260819_120633 | 19.8 s | 3840×2160 @30 | ☐ |
| VID_20260819_120701 | 18.5 s | 3840×2160 @30 | ☐ |
| VID_20260819_120727 (+ `_hi` `_relay` `_win`) | 27.8 s | 3840×2160 @30 | ☐ |
| VID_20260819_120802 (+ `_hi` `_relay` `_win`) | 28.5 s | 3840×2160 @30 | ☐ |
| VID_20260819_160630 | 47.0 s | 3840×2160 @120 | ☐ |
| VID_20260819_162151 | 61.7 s | 3840×2160 @30 | ☐ |

At 20 s windows firing every 7 s, that is roughly 4 windows per short clip and 8
for the longest — call it 35 model calls per arm, 175 across five arms. Cheap
enough to re-run when a prompt changes, which is the point.

Annotate the quiet stretches too. Gate 2 — the one that decides the winner — is
measured **only** on windows where truth says nothing happened, so a corpus of
nothing but placements cannot answer the question this axis exists to ask.

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
