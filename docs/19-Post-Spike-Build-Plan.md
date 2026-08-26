# Post-Spike Build Plan — Enrollment Redesign

**Purpose.** The enrollment-redesign spikes reached a gate on 2026-08-20. Its decision and the
ordered list of what to build live in `docs/spikes/enrollment-redesign/RESULTS.md` — **on the
`spike/enrollment-redesign` branch, which never merges** (it is a lab notebook, 56 commits / 67
experiments ahead of `main`). Because that gate is invisible from `main`, the graduates list has
already been re-derived wrong once. This file is the graduates list, on the merge path, so it is not
forgotten again. When in doubt, the gate on the spike branch is the authority; this is its shadow.

## The decision (verbatim intent)

> **Build — but not the system the plan describes.** Enrol from the stream while the object is in
> the hand. Rank against the registry instead of gating on a constant. Decide once per sighting,
> not once per frame.

Every load-bearing part of the original masked-crop redesign is **falsified** (see "Do not build").
The root cause of the production identity gap was **pixels on target** (64 px reproduces the 0.690
failure exactly), not background, not box quality, not augmentation.

## Already shipped (on `main` / in PR #6 / PR #4)

| Item | What | Ships as |
|---|---|---|
| 1a | Per-object floor: `threshold = max(floor, max_cross_object_cosine + margin)`, replacing global `VMA_IDENTITY_MIN_COSINE` | PR #6 (`gallery.py`, `pipeline.py`) |
| 2 (partial) | Enrol from the stream — grounder-free center-anchor register button (SAM2 tracker → gallery) | PR #6 (`register_listener.py`, `enroll.py`, `track.py`) |
| 0 (partial) | Frame supply — bitrate floor breaks the estimator collapse (1→15 fps) | PR #4 (`LiveKitController.kt`) |
| 7 (code) | `box_to_mask` returns the padded rectangle — this is the **measured-best** choice, not a shortcut | already the production path |

## To build, in order

Each item carries the spike number that justifies it. "Graduating" means writing fresh against
`main` as a normal feature branch.

1. **Pool the identity verdict per sighting (median over the sighting's frames).** *(gate item 1b,
   spike 9b)* — Highest remaining value: "a bigger win than any threshold tuning." Lifts the intruder
   gap +0.011 → +0.054 and gets 8/8 events right. Composes directly with the per-object floor already
   shipped. Decide once per sighting, not once per frame.
2. **Finish the media path.** *(gate item 5, spikes 14e/14f/14j/15b/15d)*
   - Publish **H.265** (15 fps at 16.8% device CPU) instead of VP8 (9 fps at 38.1%). The bitrate
     floor shipped; this codec switch is the other half of the frame fix.
   - Set **`backupCodec = h264`** — load-bearing: `liblivekit_ffi.so` has no software H.265 decoder,
     so a CPU-only subscriber sees **silence, not an error**.
   - Move the gateway's JPEG encode **off the asyncio event loop** (15b).
   - Make `rgba_raw` **fail loudly** rather than dropping two frames in three (15d).
3. **State the extent in the grounding prompt.** *(gate item 3, spike 12c)* — One sentence naming
   what to include takes the worst noun from IoU **0.12 → 0.92** and kills a temperature-0 bimodal
   swing. Containment was always 100%; the label was silently choosing extent.
4. **Condition box padding on where the box came from.** *(gate item 4, spikes 3e/2e)* — 0.75 padding
   repairs a mis-scoped grounder box (F1 0.776 → 0.909) but **hurts** a tracker box (separation
   +0.340 → +0.255). Padding is right for grounder boxes, wrong for tracker boxes.
5. **Delete the separate registration capture.** *(gate item 2 remainder, spike 13)* — A sharp 4K
   gallery feeding 720p queries scores *below* a matched 720p one; the object is 472–674 px in the
   first seconds of an ordinary recording. Registration becomes a place to ask for a **label**, not a
   source of pixels. Kills `register_video` / `registration_capture_seconds`.
6. **Fix the placement window.** *(gate item 6, spike 5b)* — Placement detects 2/2 on real clips; the
   ±5 s window breaks it to 0/2.
7. **Document `box_to_mask` as a decision, not a shortcut.** *(gate item 7, spikes 1/8)* — It is the
   strongest option measured. The docstring must say so, or the next reader "finishes" it with a
   segmenter and regresses identity. (This has happened.)

## Do NOT build (falsified — do not rebuild from the proposal doc)

| Thing | Why not | Number |
|---|---|---|
| **Any segmenter, for masking** | Makes identity worse; on a same-class twin drops ranking 5/5 → 1/5 | spikes 1, 8 |
| Reference augmentation | Moves cosine 0.690 → 0.804 with F1 **identical** at 0.723 | 1d |
| `identity_min_margin` **as a global gate** | Configured, never on the frontier over 40 decisions; the floor catches what it aimed at. *(Note: PR #6's per-object confusion-margin reuse of this knob is a different, valid mechanism — keep the comment that says so.)* | 9 |
| Glasses-side burst capture | The stream fix made it unnecessary | 0k, 14g |
| Separate 4K registration clip | 720p → 4K moves own cosine 0.008 | 13 |
| DAM4SAM tracker | No licence file; re-entry needs no tracker (4/4 as fresh detection) | 2f |
| Zero-shot exemplar conditioning | Halves accuracy 0.850 → 0.487; IPLoc-ID fine-tunes for it, it does not transfer | 3b |
| Small local VLMs as the front end | IoU 0.000 on 12/12; quantised, stops emitting boxes | 3f, 3g |

## Standing decision

**All-local, target an H200-class cloud GPU (141 GiB).** Nothing runs through a third-party API
(Gemini in spike 12 was a capability probe only). The 8 GiB budget is no longer binding — C-RADIOv4-H
(~2.5 GiB), a 3B grounder (~6 GiB) and SAM2.1-tiny (~0.2 GiB) are all resident at once, so the 4.3 s
per-sighting model swap is deleted, not optimised. Front-runner grounder: LFM2.5-VL-3B (IoU 0.889,
0.92 with the extent prompt).

## Provenance

Full experimental record, all 67 spikes and 17 falsifications:
`docs/spikes/enrollment-redesign/RESULTS.md` on branch `spike/enrollment-redesign`. Original (now
largely falsified) redesign: `docs/18-Enrollment-Redesign-Proposal.md`.
