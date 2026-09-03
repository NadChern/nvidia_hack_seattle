# Placement ground truth

One JSON file per annotated recording, written by
`services/vision-worker/scripts/annotate_placement.py`. These live in `docs/`
rather than beside the footage because **`clips/` is gitignored in full** and
hand annotation is the most expensive, least reproducible artifact in this
repository — losing it costs an afternoon per clip, losing a clip costs a
re-record.

## Schema

`eval_harness.GroundTruth`'s schema, plus five keys that place a frame index on
a time axis. `GroundTruth.load` reads only the keys it knows, so
`eval_placement.py` loads these files unchanged.

```jsonc
{
  "object_id": "keys_01",
  "label": "keys",                       // the reasoner's label vocabulary
  "distractor_ids": ["keys_02"],         // without one, identity is untestable
  "boxes": {"0": [0.40, 0.40, 0.60, 0.62]},  // frame index -> normalized xyxy
  "events": [
    {"t": 12.5, "action": "placed", "location": "on the desk beside a mug"}
  ],
  "visibility": [[0, 80], [195, 250]],   // inclusive frame ranges it is in shot
  "notes": "",

  "clip": "VID_20260819_120802_hi.mp4",  // resolved against clips/recordings/
  "fps": 10.0,
  "frame_stride": 5,                     // decode stride the indices are in
  "t_start": 0.0,                        // these recordings' PTS need not start at 0
  "duration_s": 28.0,                    // the last annotatable timestamp
  "rotate": 90,                          // clockwise degrees applied on decode
  "width": 720, "height": 1280           // AFTER rotation, for pixels on target
}
```

## `rotate` is not optional here

**PyAV ignores rotation metadata.** These recordings are shot portrait and the
originals carry `rotation=-90`, but every `av.open` path in this repository —
this tool, the event axis, and `media-gateway`'s virtual-glasses `--file`
publisher — decodes them lying on their side unless told otherwise. The
annotator applies `--rotate 90` by default and records it; the scorer reads the
recorded value and applies the same. A truth file without the key is assumed to
have been annotated sideways (`rotate: 0`), because that is what the tooling did
before the key existed.

`action` must be one of `placed`, `picked_up`, `carried`, `nothing_happened`,
`unknown` — the model's own vocabulary (`reason/cosmos.py::_ACTIONS`). Anything
else is unscoreable and `annotate_placement.py check` fails on it.

## `visibility` is annotated, not inferred

Mark every stretch the object is **in shot** — `visibility` is a list of
inclusive `[first, last]` frame ranges, and an object that leaves and comes back
gets two of them. Outside those ranges the scorer treats it as **absent**, and
absence is scored, not skipped: the expected answer is no box, and a box there
counts as a *phantom*.

The first version of this derived the ranges from the boxes — first boxed frame
to last — and that is wrong in both directions, which the wallet clip showed the
moment a human annotated it properly. The wearer sets the wallet down, walks to
the kitchen for eleven seconds, and comes back to the same desk. `min..max`
swallows the absence whole, so:

- every kitchen window was scored as *the object was there and the model failed
  to box it*, and
- `box_at` cheerfully interpolated a ground-truth box along the straight line
  between the two desk sightings, putting truth on a kitchen worktop.

Scored against the corrected file a phantom-happy arm reports 2/2 phantom boxes
and mean IoU 1.00 on the windows that can be scored. Scored against the derived
version the same arm reported 1/1 phantom, **window accuracy 0.75** and mean IoU
0.37 — it was rewarded for hallucinating in the kitchen because truth had a box
there too. A visibility model that is wrong does not degrade the score; it
measures a different question.

Sparse boxes cannot recover this on their own, either: an annotator boxes the
frames worth boxing, not every frame the object is in shot for, so the last box
before an absence is not the last sighting. Hence two fields. Boxes stay sparse;
`visibility` is the one thing worth marking exhaustively.

## Only some windows can catch an event

Boxes go in the window's **last** frame and `CosmosReasoner._parse` returns no
events at all without a box. A window whose last frame no longer shows the
object cannot report what happened in it, however clearly its earlier frames
showed it. So a placement is only *catchable* by windows that still see the
object at their end, and `placement_recall` is scored over those.

## Annotate sparsely, on purpose — but anchor the frames that get read

Boxes between two annotated frames are linearly interpolated, and the tool draws
that interpolation dashed so you can see whether it is good enough before
labelling another frame. Annotation burden that will not get done buys nothing.

Where to spend the few boxes you do draw is not arbitrary. The event axis reads
truth at exactly one frame per window — the **last** one — so those are the
frames worth getting right. At the production schedule (20 s span every 7 s,
`t_start` 0) they are the frames at t = 7, 14, 21, 28 … plus the clip's final
frame, which the trailing window always ends on. Box every one of them the
object is in shot for and the event score never touches an interpolated box at
all; interpolation across a hand-to-surface transition is meaningless, but it
costs nothing if no window ends there. Everything else you box is a sample for
the *grounding* axis, which scores each annotated frame on its own.

## A clip with no event still needs a file

Annotate `nothing_happened` explicitly. A clip with no entry cannot be told from
one nobody has labelled yet, and the false-positive rate — the number that
decides whether `promote_motion_events` can be turned on — is measured entirely
on windows where truth says nothing happened. Quiet footage is not filler here;
it is half the measurement.

## Check before trusting a run

```bash
cd services/vision-worker
uv run python scripts/annotate_placement.py check
```

It refuses an action outside the vocabulary, an event outside the clip, a
degenerate box, a box sitting outside every visibility range (and a file with
boxes but no ranges at all), and a clip whose smallest annotated box is below
~48 px on target — beneath that, identity is at chance (docs/spikes/capture-resolution)
and any event score from it will be confusing for a reason that has nothing to
do with the model.
