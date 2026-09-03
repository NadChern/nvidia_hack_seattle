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
  "notes": "",

  "clip": "VID_20260819_120802_hi.mp4",  // resolved against clips/recordings/
  "fps": 10.0,
  "frame_stride": 5,                     // decode stride the indices are in
  "t_start": 0.0,                        // these recordings' PTS need not start at 0
  "duration_s": 28.0,                    // the last annotatable timestamp
  "width": 1280, "height": 720           // capture size, for pixels on target
}
```

`action` must be one of `placed`, `picked_up`, `carried`, `nothing_happened`,
`unknown` — the model's own vocabulary (`reason/cosmos.py::_ACTIONS`). Anything
else is unscoreable and `annotate_placement.py check` fails on it.

## Annotate sparsely, on purpose

Boxes between two annotated frames are linearly interpolated, and the tool draws
that interpolation dashed so you can see whether it is good enough before
labelling another frame. Two or three anchors per clip is usually the whole job.
Annotation burden that will not get done buys nothing.

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
degenerate box, and a clip whose smallest annotated box is below ~48 px on
target — beneath that, identity is at chance (docs/spikes/capture-resolution)
and any event score from it will be confusing for a reason that has nothing to
do with the model.
