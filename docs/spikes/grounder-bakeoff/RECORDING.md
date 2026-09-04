# Recording brief — the ambient corpus

[RESULTS.md](RESULTS.md) fixed the event gates before any number existed, and
one of them cannot be reached with the footage we have. Gate 2 is a
**false-placement rate ≤ 0.05** on quiet windows; the corpus has **five**. Five
samples can *fail* that gate — one false placement is 0.2 — but they cannot
pass it: the 95% upper bound on 0/5 is ≈0.45, nine times the gate. Roughly
**60 quiet windows** puts the bound under 0.05 on a clean run.

This is the brief for shooting them. It is deliberately not more staged
placements: the corpus already has five of those and they are the expensive
kind of footage. What is missing is ordinary time with an object sitting there.

## What a quiet window actually costs to make

The scorer fires a window every 7 s spanning the previous 20 s, and a window is
**quiet** when no event falls in its span *and* the object is in shot **in the
window's last frame**. Not throughout — the last frame, because boxes are read
there and `CosmosReasoner._parse` returns no event without a box.

Two consequences, and they set the whole shape of this:

1. **An object resting in view generates a quiet window every 7 seconds.**
   90 seconds of "keys on the desk while I work" is ~13 of them. Getting to 60
   is minutes of recording, not hours.
2. **Quiet windows need no boxes.** `expected()` reads only `visibility` and
   `events`; `box_at` feeds the window-IoU *diagnostic*, and returns `None`
   harmlessly when there is nothing annotated. So a 90 s ambient clip is one
   visibility span and zero events — the cheapest annotation in the corpus.

   (The annotator does not yet believe this — see *Two decisions*, below.)

**Do not read 60 windows as 60 independent samples.** Consecutive windows
overlap by 13 of their 20 seconds, so on one static scene they are close to the
same observation counted three times. Spreading the same total across many
short recordings, in different rooms and different light, is what makes the
count mean something — which is the real reason for the shot list below rather
than one long take.

## The shot list

Eight recordings, ~90 s each, ~12 minutes total. That is ~100 windows, of which
~75 should come out quiet — comfortably over 60 with room for clips that turn
out unusable.

**A — resting in view, four clips.** The object sits somewhere it plausibly
lives (desk corner, shelf, kitchen counter) and you do something else in the
same room for 90 s. Work at the laptop, wash up, read. The object stays in
frame most of the time and you never touch it. This is the bulk of the corpus
and the easiest to shoot.

**B — hands busy next to it, two clips.** Same setup, but your hands are
working *within a few inches of the untouched object*: moving a mug past the
keys, tidying the desk around them, unpacking a bag beside them. **This is
where a false placement actually comes from**, and the current corpus contains
not one window of it — every existing quiet window has the object alone and
still. If gate 2 is going to be failed by a real arm, it will be failed here,
so these two clips carry more weight than the four in A.

**C — in and out of shot, two clips.** Move around the flat with the object
staying put: leave the room, come back, glance at it, walk past. Besides
varying the background, these supply **absent** windows, which are the only
place a phantom box is scored.

Across all eight, vary what is cheap to vary: room, time of day, daylight
versus lamps, cluttered versus clear surface. Two of the clips should include a
**distractor** — a second keyring, a similar-looking object — sitting near the
enrolled one, the way `160630` does.

### While recording

- **Do not touch the enrolled object** in A and B. If you do, note roughly
  when; one handled moment does not spoil a clip, an unrecorded one does.
- **Keep it recognisable.** Identity holds to ~128 px on target and reaches
  chance by ~48 px (`docs/spikes/capture-resolution`). A keyring across the
  room is not a fair test of anything; the annotator prints pixels on target if
  you want to check a framing before committing 90 s to it.
- **Leave two or three enrolled objects in shot** whenever it is convenient —
  keys and wallet on the same desk. It costs nothing at record time and each
  one can become its own truth file later, if we decide the annotation is worth
  it (see *Two decisions*).
- Keep each take to ~90 s. Longer takes buy correlated windows, not new ones.

## Getting the files into the corpus

Recordings come off the glasses as 4 K HEVC with a `rotation=-90` display
matrix and an audio track. The scorer never sees that: it sees the
gateway-quality re-encode, **1280×720, 10 fps, H.264, video only, and the
display matrix dropped** — which is why every truth file carries `rotate: 90`
and applies it on decode itself.

That last part is a trap. `-noautorotate` stops ffmpeg *applying* the rotation
but still copies the matrix into the output, and the annotator would then
rotate an already-rotated frame. `-display_rotation 0` as an **input** option
is what actually clears it:

```bash
cd clips/recordings
for f in VID_2026*.mp4; do
  case "$f" in *_hi.mp4|*_relay.mp4|*_win.mp4) continue;; esac
  ffmpeg -y -display_rotation 0 -i "$f" -map 0:v:0 -an \
    -vf "fps=10,scale=1280:720" \
    -c:v libx264 -profile:v high -pix_fmt yuv420p -b:v 4.6M \
    -movflags +faststart "${f%.mp4}_hi.mp4"
done
```

Verify one before annotating eight — this prints the dimensions and should
print nothing about a display matrix:

```bash
ffprobe -v error -select_streams v:0 -show_streams clips/recordings/<new>_hi.mp4 | grep -iE "^width|^height|displaymatrix|rotation"
```

Keep the camera's own `VID_YYYYMMDD_HHMMSS` names. They sort chronologically,
they are how every existing truth file names its clip, and re-deriving which
take was which from a renamed file is not worth the tidiness.

## Two decisions to make when the footage lands

Neither is worth doing before we know what was actually shot, and both are
small.

**1. Box-free visibility spans.** `annotate_placement.py:621` reports a
visibility range containing no box as a problem, and `validate` gates the save
as well as `check` — so the annotator will currently refuse to save exactly the
cheap ambient annotation this brief is built around. The message itself already
says "fine if the object is only distantly visible; say so in notes", so the
intent was advisory and the implementation is stricter than the scorer. Relax
it and ~12 minutes of footage is an hour of annotation; leave it and every span
needs a box.

**2. Two objects on one recording collide.** `spike_grounder_bakeoff.py:1111`
builds its truth map as `truths[truth.clip] = truth`, keyed by the **clip
filename**. Two truth files naming the same recording — keys and wallet on the
same desk — silently overwrite each other, and the window-IoU diagnostic is
then scored against the wrong object's boxes. The action scoring is unaffected
(each record carries its own label), so this is a diagnostic bug rather than a
gate bug, but it must be fixed before a second truth file names a clip that is
already annotated. Key the map by truth file, not by clip.
