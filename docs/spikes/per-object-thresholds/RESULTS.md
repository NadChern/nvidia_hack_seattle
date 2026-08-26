# Per-object identity thresholds (Spike 4)

## Decision

**Adopt, and graduate into the service.** Gallery-derived per-object thresholds replace the single
global `identity_min_cosine=0.8334`. On the three-keyring probe they lift correct identifications
from 19/25 to 24/25 (76% -> 96%) at **zero** misidentifications, cutting false rejections from 6 to
1. The mechanism degrades safely: an object with no same-label sibling falls back to a floor, which
is exactly why one global cosine sufficed for the one-object demo and would not survive a second
same-class instance. Shipped defaults: floor `identity_min_cosine=0.75`, confusion margin
`identity_min_margin=0.02`.

This spike was scoped in `docs/spikes/enrollment-redesign` (the enrollment-redesign plan, Spike 4)
as a ~2-hour experiment folding into Spike 1's embeddings. It passed its gate and, at the user's
direction, graduated the same session into `services/vision-worker/src` — the gallery scorer, the
pipeline gate, and config wiring — rather than waiting for the aggregate enrollment gate. The
remaining enrollment spikes (0, 1, 2) are independent of this result.

## Record

- Date: 2026-08-21
- Owner: Visual Memory Assistant maintainers
- Time box: Spike 4 (~2 hours), plus same-session graduation into `services/*/src`
- Source baseline: `3946a61b9168ea1a80cde20b5d07600c9d41f830` on `feat/register-button`
- Machine: WSL2 Linux x86_64, NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB
- Runtime: Python 3.11.14, Torch 2.6.0+cu126, Transformers 4.57.6
- Embedder: `nvidia/C-RADIOv4-H` revision `0057b339059c0b9e1b4ba996f975410ebbfdfcc8`, the same
  adapter the identity probe used
- Inputs: local, gitignored `clips/identity-probe`; 3 physical keyrings (distinct fobs), 11
  reference views, 25 queries (`--split all`: 8 clean + 17 realistic)
- Harness: `services/vision-worker/scripts/validate_object_thresholds.py`, which wraps the probe
  vectors as production `GalleryView`s and runs the **actual** `score_gallery` / `object_thresholds`
  code (summary_weight=1.0, since the probe embedder emits only the summary channel)

## Method

The threshold for each registered object is derived from the gallery rather than hand-tuned:

```
threshold(obj) = max(floor, max_cross_object_cosine(obj) + confusion_margin)
```

where `max_cross_object_cosine(obj)` is the highest reference-vs-reference cosine between `obj`'s
views and those of any *other same-label* object. An object with no same-label sibling has no cross
term and degrades to `floor`. A query is accepted for the winning object only if its gallery score
clears that object's threshold.

Each keyring is independently the enrolled target; its own queries are positives, and the accept
bar is the winner's derived threshold (for the GLOBAL baseline, the flat `0.8334`). Outcomes are
counted three ways: **correct** (accepted as the true keyring), **misidentified** (accepted as the
wrong keyring), **rejected** (fell below the bar). The sweep runs floor in {0.60, 0.70, 0.75, 0.80}
x margin in {0.02, 0.04, 0.06}.

Reproduce:

```bash
cd services/vision-worker
uv run python scripts/validate_object_thresholds.py --split all
```

## Results

Baseline to beat is the global constant that the identity-probe RESULTS warned "won't survive a
second same-class instance":

| Gate | Correct | Misidentified | Rejected | Derived bars |
|---|---|---|---|---|
| GLOBAL `0.8334` | 19/25 (76%) | 0 | **6 (24%)** | — |
| floor 0.60, margin 0.02 | **24/25 (96%)** | 0 | 1 (4%) | keys_1 0.791 · keys_2 0.742 · keys_3 0.791 |
| floor 0.70, margin 0.02 | **24/25 (96%)** | 0 | 1 (4%) | keys_1 0.791 · keys_2 0.742 · keys_3 0.791 |
| **floor 0.75, margin 0.02 (shipped)** | **24/25 (96%)** | 0 | 1 (4%) | keys_1 0.791 · keys_2 0.750 · keys_3 0.791 |
| floor 0.80, margin 0.02 | 23/25 (92%) | 0 | 2 (8%) | keys_1 0.800 · keys_2 0.800 · keys_3 0.800 |

Full margin sweep (all at 0 misidentified): raising the margin monotonically trades recall away for
no precision gain — 0.04 -> 23/25, 0.06 -> 22/25 at every floor — because misidentifications are
already zero, so the margin only lifts the bar into true positives.

Three findings drove the shipped defaults:

1. **The global constant's failure mode here is over-rejection, not confusion.** It never
   misidentifies (0/25), but it is set so conservatively that it discards a quarter of valid
   queries. The per-object bars sit *below* 0.8334 for every keyring, which is precisely the recall
   the global number was giving up.

2. **Per-object bars self-calibrate to confusability.** keys_2 is less similar to the others and
   earns a lower bar (cross-cosine 0.722), while keys_1 and keys_3 are mutually confusable and hold
   a higher one (0.771). One global constant cannot express that spread; it must sit at the max and
   over-reject everyone else.

3. **0.75 is the safe floor.** Floors of 0.60–0.75 all reach the best 24/25, because the derived
   cross-object bars (0.742–0.791) exceed them and do the gating. At floor 0.80 the floor starts
   *binding* — it clips keys_2's natural 0.742 bar up to 0.800 and re-introduces a false reject. So
   0.75 is the highest floor that still protects a lone registered object (no sibling -> bar 0.75)
   without clipping a legitimately-low per-object bar.

## Failures and constraints

- **The misidentification claim is not stress-tested by this data.** The three keyrings carry
  visually distinct fobs, so there is 0 misidentification even at the global 0.8334 — the "second
  same-class instance the global threshold could not survive" does not actually manifest as a
  confusion on this set. This spike therefore validates the mechanism's *safety* (never worse than
  global on precision; degrades to floor for a lone object) and its *recall* benefit (6 -> 1 false
  rejects), but the precision-under-genuine-confusion claim awaits ≥2 near-identical same-class
  pairs. The enrollment-redesign plan already defers the full 20–30 capture set with ≥3 same-class
  instance pairs for exactly this; **these thresholds are exploratory keys defaults, not a frozen
  or general claim.**
- **summary_weight=1.0 in the harness, 0.5 in production.** The probe embedder emits only the
  summary channel, so the harness cannot exercise the blended summary+spatial score the live
  pipeline uses. The threshold *derivation* is channel-agnostic and unit-tested against the blended
  path (`test_identity_gallery.py`), but the specific numbers above are summary-only.
- **Single embedder, single machine.** C-RADIOv4-H on one 4070; no cross-checkpoint or GN100
  validation.

## Follow-up

- Graduated to `services/*/src` this session: `identity/gallery.py` (`object_thresholds`,
  `score_gallery(..., floor, confusion_margin)`, `GalleryScore.threshold`), `pipeline.py` (gate now
  `score.score < score.threshold`, wires `identity_min_margin`), `main.py`, and `config.py`
  defaults (floor 0.75, margin 0.02; `identity_escalation_low` lowered to 0.73 to keep its
  `<= floor` invariant). 169 vision-worker tests pass, ruff clean.
- **Freeze-blocked on data:** do not treat 0.75/0.02 as final until re-measured on a capture set
  with genuinely near-identical same-class instances; that set's protocol is written at the
  enrollment gate.
- The `identity_min_margin` knob, defined in config but never wired into `pipeline.py`, is now
  live — closing the "margin gate configured but inert" gap the plan flagged.
