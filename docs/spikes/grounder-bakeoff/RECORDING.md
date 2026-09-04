# Recording brief — filming the "nothing happened" footage

**In one line:** we need ~12 minutes of ordinary video where an object (keys,
wallet) sits somewhere untouched while you go about your day. Eight short clips,
about 90 seconds each. That's the whole job. The rest of this page explains why,
and what to do with the files afterwards.

## Why we're filming this

We're choosing which AI model to ship. One of its jobs is to watch your camera
feed and notice **"you just put your keys down here"** — we call that a
*placement*, and it's how the product remembers where your things are.

The danger is a model that **invents** placements — tells you it saw you set the
keys on the shelf when you never did. That stores a wrong location and sends you
to the wrong place, which is worse than saying nothing. So before we ship a
model, we have to measure how often it does this.

To measure it, we need lots of footage where **an object is just sitting there
and nothing is happening to it** — so we can count how often the model wrongly
shouts "placed!" over a perfectly still object.

**Right now we have almost none of it.** Our existing clips are all *staged
placements* — someone deliberately putting a thing down. We have only five short
stretches of "nothing happening," and five is too few to prove a model is safe:
it could get all five right by luck. We need roughly **sixty**. This footage is
those sixty.

## How little it takes (the nice surprise)

The AI reviews your video in 20-second chunks, and it starts a fresh chunk every
7 seconds. So a single 90-second clip of your keys sitting on the desk gives us
about **thirteen** testable "nothing happened" moments — not one. Getting to
sixty is a matter of minutes of filming, not hours.

And these clips are the *cheapest* kind to prepare afterwards: because nothing
happens in them, there's nothing to label frame-by-frame. Just note "the keys
are in shot from here to here" and that's it.

One catch worth knowing: those 20-second chunks overlap heavily, so sixty
moments from *one* long static shot aren't really sixty independent tests —
they're almost the same shot counted over and over. That's why the plan below is
**eight short clips in different places**, not one long one. Different rooms,
different light, different clutter — that's what makes the count actually mean
something.

## What to film — eight clips, ~90 seconds each

Total ~12 minutes. That yields well over the sixty moments we need, with room
for a clip or two that doesn't come out.

**Group A — object resting in view (four clips).** Put the keys (or wallet)
somewhere it would naturally sit — desk corner, shelf, kitchen counter — and
just *do something else in the same room* for 90 seconds. Work at the laptop,
wash up, read. The object stays in frame most of the time and **you never touch
it.** This is the bulk of the footage and the easiest to shoot.

**Group B — hands busy right next to it (two clips).** Same idea, but now your
hands are working *within a few inches of the untouched object* — moving a mug
past the keys, tidying the desk around them, unpacking a bag beside them.
**These two matter most.** A model is most tempted to invent a placement when
there's hand movement near the object, and none of our current footage has that.
If a model is going to fail this test, it'll fail here.

**Group C — object goes in and out of view (two clips).** Leave the object where
it is and move around — walk out of the room, come back, glance at it, pass by.
Besides giving us different backgrounds, these give us stretches where the object
is *out of shot*, which is the only place we can catch a model hallucinating an
object that isn't even there.

Across all eight, change whatever's easy to change: room, time of day, daylight
vs. lamplight, tidy surface vs. cluttered one. On **two** of the clips, leave a
lookalike nearby — a second keyring, a similar object — sitting next to the real
one, to see whether the model confuses them.

## While you're filming

- **Don't touch the object** in groups A and B. If you slip and pick it up,
  just note roughly when (e.g. "grabbed the keys around 0:40") — one handled
  moment doesn't ruin a clip, but an *unrecorded* one quietly corrupts the test.
- **Keep the object big enough to recognise.** If it's across the room it's a
  tiny blur and tests nothing. Rule of thumb: it should be clearly identifiable,
  not a distant speck. (There's a tool that can show you the exact pixel size if
  you want to check a framing before committing 90 seconds to it.)
- **Leave a couple of enrolled objects in shot** when it's convenient — keys
  *and* wallet on the same desk. It costs nothing while filming and gives us two
  tests from one clip later.
- **Keep each clip to ~90 seconds.** Longer clips don't add new moments, they
  just repeat the same one — better to start a fresh clip somewhere else.

---

# After the footage lands (technical — for processing, not filming)

Everything below is for whoever prepares the files; skip it if you're just
recording.

## Convert each recording before annotating it

Clips come off the glasses as 4K video, rotated, with an audio track. The
scoring tool doesn't use that — it uses the same downscaled version the real
product would see: **1280×720, 10 fps, H.264, no audio.** Convert each new
recording with:

```bash
cd clips/recordings
for f in VID_2026*.mp4; do
  case "$f" in *_hi.mp4|*_relay.mp4|*_win.mp4) continue;; esac   # skip already-converted
  ffmpeg -y -display_rotation 0 -i "$f" -map 0:v:0 -an \
    -vf "fps=10,scale=1280:720" \
    -c:v libx264 -profile:v high -pix_fmt yuv420p -b:v 4.6M \
    -movflags +faststart "${f%.mp4}_hi.mp4"
done
```

**The one non-obvious flag is `-display_rotation 0`.** The footage carries a
"rotate me 90°" tag, and our annotation files *also* apply that rotation
themselves. If the converted file keeps the tag, the frame gets rotated twice
and everything is sideways. `-display_rotation 0` strips the tag so rotation
happens exactly once, in our tooling. (`-noautorotate` is *not* enough — it stops
ffmpeg rotating but still copies the tag through. This recipe was verified
against the existing clips.)

Check one file before converting all eight — this should print the size and
**nothing** about a display matrix or rotation:

```bash
ffprobe -v error -select_streams v:0 -show_streams clips/recordings/<new>_hi.mp4 | grep -iE "^width|^height|displaymatrix|rotation"
```

Keep the camera's original `VID_YYYYMMDD_HHMMSS` filenames — they sort by time
and every annotation file refers to its clip by that name.

## Two code fixes the new footage will force

Neither is worth doing before we see what was actually shot; both are small.

**1. Let the annotator save a clip that has no boxes.**
`annotate_placement.py:621` currently refuses to save an annotation where an
object is marked visible but has no box drawn on it. But the whole point of this
footage is cheap "object is over there, untouched" spans with *no* boxes — so as
it stands, the tool won't save exactly what we're shooting. (Its own warning
even says the box is optional "if the object is only distantly visible", so the
intent was advisory — the code is just stricter than it meant to be.) Relax it
and ~12 minutes of footage is about an hour of annotation; leave it and every
span needs a hand-drawn box.

**2. Don't let two objects on one clip overwrite each other.**
`spike_grounder_bakeoff.py:1111` stores ground truth keyed by *clip filename*.
If we annotate keys *and* wallet on the same recording as two files, the second
silently replaces the first, and one location-accuracy number gets scored
against the wrong object. (The main pass/fail scoring is unaffected — each entry
knows its own object — so this only corrupts a secondary diagnostic.) Fix:
store truth keyed by the annotation file, not by the clip. Only needed once a
second object is annotated on an already-annotated clip.
