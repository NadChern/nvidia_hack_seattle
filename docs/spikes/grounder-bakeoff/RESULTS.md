# Grounder bake-off — results

**Status: not yet run.** The decision rule below is written *before* the numbers
exist, on purpose. A rule chosen after seeing the results is not a rule.

## Decision rule (fixed in advance)

An arm wins only if it passes all four, in order:

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

## Summary

Fill from `runs/*.json`, not from terminal scrollback.

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

- **Event/action classification.** Grounding is one of two things the reasoner
  does; `placed` / `picked_up` / `nothing_happened` is the other, and a model can
  ground well while hallucinating handling on a resting object (which is why
  `promote_motion_events` is off by default). Score with `eval_placement.py` on
  the recorded clips before switching the production reasoner.
- **MOSS-VL's native streaming mode.** Scored here on the same single-image
  protocol as everyone else for comparability, which under-sells it — its
  cross-attention design keeps watching while generating, and it ships a
  WebSocket interface taking external JPEG frames. If it places well on the
  single-image axis, the continuous-mode run is worth doing before deciding.
- **Identity F1 downstream.** IoU is a proxy. The thing that matters is whether
  the crop feeds C-RADIO a recognisable object; `spike_extent.py` +
  `probe_identity.py` close that loop.
