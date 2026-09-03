#!/usr/bin/env python3
"""Annotate placement ground truth on recorded clips, in a browser.

Twelve clips sit in `clips/recordings/` and not one of them is annotated, so
the event half of the reasoner -- `placed` / `picked_up` / `carried` /
`nothing_happened` -- has never been scored on real footage. Grounding has
36 human-annotated boxes behind it; events have nothing, which is why
`promote_motion_events` is off by default on a hunch ("at ~1fps Cosmos
hallucinates handling on a resting object") rather than on a number. This tool
produces the missing numbers' input.

## Why a browser and not a cv2 window

Development runs under WSL2, where an OpenCV or matplotlib window means an X
server. A local HTTP server and the browser already on the machine need
nothing installed and no display forwarding, and video scrubbing is something
browsers are genuinely good at.

## Why extracted frames and not the video element

The truth schema keys boxes by **frame index**, and `<video>.currentTime` is a
float that does not reliably land on a frame boundary. Frames are therefore
decoded once with PyAV, at the same stride the scorer will later use, and the
annotator sees exactly the images that will be scored -- there is no
seek-precision gap between what was labelled and what a model is shown.

## What "enough annotation" means here

Deliberately sparse, matching `eval_harness.GroundTruth`: a few boxes and the
events. Boxes between two annotated frames are linearly interpolated, and the
tool draws that interpolation dashed so the annotator can see whether it is
good enough before labelling another frame. The spikes established that a
tracker carries boxes between sparse anchors; annotation burden that will not
get done buys nothing.

Every frame view also prints **pixels on target** for the current box, because
identity holds to ~128 px and reaches chance by ~48 px
(docs/spikes/capture-resolution). A clip whose object never clears the floor is
worth discovering before it is annotated, not after it produces a confusing
score.

## Output

One JSON file per clip under `docs/spikes/grounder-bakeoff/truth/`, tracked in
git -- `clips/` is gitignored in full, and hand annotation is the most
expensive and least reproducible artifact in the repository. The schema is
`eval_harness.GroundTruth`'s, so `eval_placement.py` loads these files
unchanged; the extra `clip` / `fps` / `frame_stride` / `duration_s` keys are
what let a frame index be read back as a timestamp, which is what the event
axis scores against.

Usage::

    uv run python scripts/annotate_placement.py annotate \\
        --clip ../../clips/recordings/VID_20260819_120727_hi.mp4

    uv run python scripts/annotate_placement.py check
"""

from __future__ import annotations

import argparse
import json
import math
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

#: The reasoner's action vocabulary, verbatim from `reason/cosmos.py`. An
#: annotation outside this set is unscoreable, so `check` refuses it.
ACTIONS = ("placed", "picked_up", "carried", "nothing_happened", "unknown")

#: From docs/spikes/capture-resolution/RESULTS.md, via `eval_harness`. Printed
#: beside every box so a clip that cannot support identity is caught during
#: annotation rather than after a model scores badly on it.
PIXEL_FLOOR = 128.0
PIXEL_CHANCE = 48.0

#: Annotate at roughly this rate. The reasoner sees 8 frames across a 20 s
#: window (~2.5 s apart), so labelling faster than this buys nothing the
#: scorer can use; labelling much slower loses the moment of release.
TARGET_ANNOTATE_FPS = 2.0

#: Frames are downscaled for the browser only. The manifest keeps the source
#: dimensions, so a normalized box still reports pixels on target at capture
#: resolution -- the number that governs identity.
PREVIEW_LONG_EDGE = 1280

#: Clockwise degrees applied to every decoded frame, and recorded in the truth
#: file so the scorer applies exactly the same.
#:
#: These recordings need 90. The camera is mounted portrait and the container
#: carries `rotation=-90`, but **PyAV does not apply rotation metadata** -- so
#: every `av.open` path in this repository decodes them lying on their side,
#: including `media-gateway`'s virtual-glasses `--file` publisher. Replaying a
#: recording into the pipeline therefore feeds the reasoner a sideways world,
#: which is worth knowing before blaming a model for missing an object.
#: (The live relay is reportedly portrait already; see the rotation log in
#: `media_gateway/transport/room_worker.py`.)
DEFAULT_ROTATE = 90


# --- Frame extraction --------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """Where a clip's extracted frames and manifest ended up."""

    root: Path
    manifest: Path
    frames: int


def _rotated(image, clockwise: int):
    """Rotate a PIL image by whole clockwise degrees."""
    from PIL import Image

    if clockwise % 360 == 0:
        return image
    return image.transpose(
        {
            90: Image.Transpose.ROTATE_270,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,
        }[clockwise % 360]
    )


def prepare(
    clip: Path, work_root: Path, annotate_fps: float, long_edge: int, rotate: int
) -> Prepared:
    """Decode `clip` to JPEGs at a stride, keeping source frame indices.

    The index written here is the frame's position in the decode stream, which
    is what `eval_harness.Clip.open(..., stride=...)` reports as `Frame.index`.
    Keeping them identical is what makes a truth file usable by the scorer
    without a second, drift-prone mapping.
    """
    import av
    from PIL import Image

    root = work_root / clip.stem
    manifest_path = root / "frames.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("long_edge") == long_edge
            and existing.get("annotate_fps") == annotate_fps
            and existing.get("rotate") == rotate
        ):
            return Prepared(root, manifest_path, len(existing["frames"]))

    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        time_base = stream.time_base
        stride = max(1, round(fps / annotate_fps))
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        if rotate % 180:
            width, height = height, width
        for position, decoded in enumerate(container.decode(stream)):
            if position % stride:
                continue
            # A container without a time base, or a frame without a
            # presentation timestamp, still has to land on the time axis --
            # every event is annotated at one of these values.
            usable = decoded.pts is not None and time_base is not None
            t = float(decoded.pts * time_base) if usable else position / fps
            image = _rotated(Image.fromarray(decoded.to_ndarray(format="rgb24")), rotate)
            image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
            name = f"{position:06d}.jpg"
            image.save(root / name, format="JPEG", quality=88)
            entries.append({"index": position, "t": round(t, 4), "file": name})
    if not entries:
        raise SystemExit(f"no frames decoded from {clip}")

    manifest = {
        "clip": clip.name,
        "clip_path": str(clip),
        "fps": fps,
        "frame_stride": stride,
        "annotate_fps": annotate_fps,
        "long_edge": long_edge,
        "rotate": rotate,
        "width": width,
        "height": height,
        #: The last annotatable timestamp, not the wall duration: presentation
        #: timestamps on these recordings do not start at zero, and an event's
        #: `t` is one of these frame timestamps, so this is the bound that
        #: matters when validating one.
        "t_start": entries[0]["t"],
        "duration_s": round(entries[-1]["t"], 4),
        "frames": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Prepared(root, manifest_path, len(entries))


# --- The annotator page ------------------------------------------------------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>placement annotator</title>
<style>
 :root { color-scheme: dark; --line:#2c3140; --ink:#e6e9f0; --dim:#8b93a7; --hot:#ffb347; }
 body { margin:0; background:#14171f; color:var(--ink);
        font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
 header { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
          padding:8px 12px; border-bottom:1px solid var(--line); }
 header input { background:#1c2029; border:1px solid var(--line); color:var(--ink);
                padding:3px 6px; font:inherit; border-radius:3px; }
 main { display:flex; gap:12px; padding:12px; align-items:flex-start; }
 #stage { position:relative; line-height:0; flex:1 1 auto; max-width:70vw; }
 #frame { width:100%; height:auto; display:block; border-radius:4px; }
 #overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; }
 aside { width:330px; flex:0 0 auto; }
 h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
      color:var(--dim); margin:14px 0 6px; }
 .row { display:flex; justify-content:space-between; gap:8px; padding:2px 0;
        border-bottom:1px solid #1e2230; }
 .row button { background:none; border:none; color:var(--dim); cursor:pointer; font:inherit; }
 .row button:hover { color:#ff7b7b; }
 .row.here { color:var(--hot); }
 button.go { background:#2a3040; border:1px solid var(--line); color:var(--ink);
             padding:5px 10px; font:inherit; border-radius:3px; cursor:pointer; }
 button.go:hover { background:#343c50; }
 select, textarea { background:#1c2029; border:1px solid var(--line); color:var(--ink);
                    font:inherit; border-radius:3px; padding:3px 6px; width:100%; }
 #status { color:var(--dim); padding:0 12px 10px; }
 #rule { color:var(--dim); }
 kbd { background:#242936; border:1px solid var(--line); border-radius:3px; padding:0 4px; }
 .warn { color:#ff7b7b; } .ok { color:#7bd88f; } .mid { color:var(--hot); }
</style></head><body>
<header>
  <b id="clipname">-</b>
  <label>object_id <input id="object_id" size="10" value="keys_01"></label>
  <label>label <input id="label" size="8" value="keys"></label>
  <label>distractors <input id="distractors" size="12" placeholder="keys_02"></label>
  <button class="go" id="save">save</button>
  <span id="saved"></span>
</header>
<main>
  <div id="stage"><img id="frame" alt=""><canvas id="overlay"></canvas></div>
  <aside>
    <div id="where">-</div>
    <div id="pixels">-</div>
    <h2>boxes</h2><div id="boxes"></div>
    <h2>events</h2>
    <select id="action">
      <option>placed</option><option>picked_up</option><option>carried</option>
      <option>nothing_happened</option><option>unknown</option>
    </select>
    <input id="location" placeholder="on the kitchen table next to a mug" style="width:100%">
    <button class="go" id="addevent">add event at this frame</button>
    <div id="events"></div>
    <h2>notes</h2><textarea id="notes" rows="3"></textarea>
    <h2>the visibility rule</h2>
    <div id="rule">Box the object on the <b>first</b> and <b>last</b> frame it is visible.
      Outside that range the scorer treats it as <b>absent</b> &mdash; it expects the model to
      return no box there, and counts one that does as a phantom.</div>
    <h2>keys</h2>
    <div><kbd>&larr;</kbd><kbd>&rarr;</kbd> step &middot; <kbd>&darr;</kbd><kbd>&uarr;</kbd> x10
      &middot; drag to box &middot; <kbd>d</kbd> drop box &middot; <kbd>p</kbd> play
      &middot; <kbd>s</kbd> save</div>
  </aside>
</main>
<div id="status">loading...</div>
<script>
let M = null, pos = 0, playing = null;
let truth = {object_id:"keys_01", label:"keys", distractor_ids:[], boxes:{}, events:[], notes:""};
const $ = (id) => document.getElementById(id);
const img = $("frame"), cv = $("overlay"), ctx = cv.getContext("2d");

function frameNow() { return M.frames[pos]; }

function boxAt(index) {
  if (truth.boxes[index]) return truth.boxes[index];
  const keys = Object.keys(truth.boxes).map(Number).sort((a,b) => a-b);
  const lo = keys.filter(k => k < index).pop(), hi = keys.filter(k => k > index)[0];
  if (lo === undefined || hi === undefined) return null;
  const w = (index - lo) / (hi - lo), a = truth.boxes[lo], b = truth.boxes[hi];
  return [0,1,2,3].map(i => a[i] + (b[i] - a[i]) * w);
}

function verdict(px) {
  if (px >= 256) return ["comfortable", "ok"];
  if (px >= 128) return ["at floor", "mid"];
  if (px >= 48) return ["DEGRADED", "warn"];
  return ["CHANCE", "warn"];
}

function draw() {
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  ctx.clearRect(0, 0, cv.width, cv.height);
  const f = frameNow(), exact = truth.boxes[f.index], b = exact || boxAt(f.index);
  if (!b) { $("pixels").textContent = "no box here"; return; }
  ctx.setLineDash(exact ? [] : [6, 5]);
  ctx.strokeStyle = exact ? "#ffb347" : "#6f7ea8";
  ctx.lineWidth = 2;
  ctx.strokeRect(b[0]*cv.width, b[1]*cv.height, (b[2]-b[0])*cv.width, (b[3]-b[1])*cv.height);
  const px = Math.sqrt(Math.max(0,(b[2]-b[0])*M.width) * Math.max(0,(b[3]-b[1])*M.height));
  const [word, cls] = verdict(px);
  $("pixels").innerHTML = (exact ? "annotated" : "interpolated") +
    " &middot; " + px.toFixed(0) + " px on target <span class=" + cls + ">" + word + "</span>";
}

function render() {
  const f = frameNow();
  img.src = "frames/" + f.file;
  $("where").textContent = "frame " + f.index + "  t=" + f.t.toFixed(2) + "s   (" +
    (pos+1) + "/" + M.frames.length + ")";
  const keys = Object.keys(truth.boxes).map(Number).sort((a,b) => a-b);
  $("boxes").innerHTML = keys.map(k =>
    "<div class='row" + (k === f.index ? " here" : "") + "'><span>frame " + k +
    "</span><button onclick='dropBox(" + k + ")'>drop</button></div>").join("") ||
    "<div class=row><span style='color:#8b93a7'>none yet</span></div>";
  $("events").innerHTML = truth.events.map((e, i) =>
    "<div class=row><span>" + e.t.toFixed(2) + "s " + e.action +
    (e.location ? " &middot; " + e.location : "") +
    "</span><button onclick='dropEvent(" + i + ")'>drop</button></div>").join("") ||
    "<div class=row><span style='color:#8b93a7'>none yet</span></div>";
  draw();
}

function dropBox(k) { delete truth.boxes[k]; render(); }
function dropEvent(i) { truth.events.splice(i, 1); render(); }

let anchor = null;
cv.addEventListener("mousedown", (e) => {
  const r = cv.getBoundingClientRect();
  anchor = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
});
cv.addEventListener("mousemove", (e) => {
  if (!anchor) return;
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  draw();
  ctx.setLineDash([]); ctx.strokeStyle = "#ffb347"; ctx.lineWidth = 2;
  ctx.strokeRect(anchor[0]*cv.width, anchor[1]*cv.height,
                 (x-anchor[0])*cv.width, (y-anchor[1])*cv.height);
});
cv.addEventListener("mouseup", (e) => {
  if (!anchor) return;
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const box = [Math.min(anchor[0],x), Math.min(anchor[1],y),
               Math.max(anchor[0],x), Math.max(anchor[1],y)];
  anchor = null;
  if (box[2]-box[0] < 0.004 || box[3]-box[1] < 0.004) { render(); return; }
  truth.boxes[frameNow().index] = box.map(v => Math.round(Math.min(1, Math.max(0, v)) * 1e4) / 1e4);
  render();
});

function step(n) { pos = Math.min(M.frames.length-1, Math.max(0, pos + n)); render(); }

$("addevent").onclick = () => {
  truth.events.push({t: frameNow().t, action: $("action").value,
                     location: $("location").value || null});
  truth.events.sort((a,b) => a.t - b.t);
  render();
};

function collect() {
  truth.object_id = $("object_id").value.trim();
  truth.label = $("label").value.trim();
  truth.distractor_ids = $("distractors").value.split(",").map(s => s.trim()).filter(Boolean);
  truth.notes = $("notes").value;
  return truth;
}

async function save() {
  const r = await fetch("truth", {method: "POST", headers: {"content-type": "application/json"},
                                  body: JSON.stringify(collect())});
  const body = await r.json();
  $("saved").innerHTML = r.ok ? "<span class=ok>saved " + body.path + "</span>"
                              : "<span class=warn>" + body.error + "</span>";
}
$("save").onclick = save;

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const go = {ArrowLeft:-1, ArrowRight:1, ArrowDown:-10, ArrowUp:10}[e.key];
  if (go !== undefined) { e.preventDefault(); step(go); }
  else if (e.key === "d") { dropBox(frameNow().index); }
  else if (e.key === "s") { save(); }
  else if (e.key === "p") {
    if (playing) { clearInterval(playing); playing = null; }
    else { playing = setInterval(() => {
      if (pos >= M.frames.length-1) { clearInterval(playing); playing = null; } else step(1);
    }, 250); }
  }
});
window.addEventListener("resize", draw);
img.addEventListener("load", draw);

fetch("manifest").then(r => r.json()).then(data => {
  M = data.manifest;
  if (data.truth) {
    truth = data.truth;
    truth.boxes = truth.boxes || {}; truth.events = truth.events || [];
    $("object_id").value = truth.object_id || ""; $("label").value = truth.label || "";
    $("distractors").value = (truth.distractor_ids || []).join(", ");
    $("notes").value = truth.notes || "";
  }
  $("clipname").textContent = M.clip;
  $("status").textContent = M.frames.length + " frames at " + M.annotate_fps +
    " fps (stride " + M.frame_stride + " of " + M.fps.toFixed(2) + " fps source, " +
    M.width + "x" + M.height + ", " + M.duration_s.toFixed(1) + "s, rotated " +
    M.rotate + " deg clockwise on decode)";
  render();
});
</script></body></html>
"""


# --- Truth files -------------------------------------------------------------


def truth_path(truth_dir: Path, clip: Path) -> Path:
    return truth_dir / f"{clip.stem}.json"


def validate(payload: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Every reason this annotation could not be scored, or an empty list.

    Runs on save as well as in `check`, because an annotation that fails
    validation an hour later has already cost the annotation session.
    """
    problems: list[str] = []
    if not str(payload.get("object_id", "")).strip():
        problems.append("object_id is empty")
    if not str(payload.get("label", "")).strip():
        problems.append("label is empty")
    boxes = payload.get("boxes") or {}
    for key, box in boxes.items():
        if len(box) != 4 or not all(0.0 <= float(v) <= 1.0 for v in box):
            problems.append(f"box at frame {key} is not four values in 0..1")
        elif float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            problems.append(f"box at frame {key} is degenerate")
    events = payload.get("events") or []
    duration = float(manifest.get("duration_s", 0.0))
    for event in events:
        if event.get("action") not in ACTIONS:
            problems.append(f"action {event.get('action')!r} is outside the model's vocabulary")
        if not 0.0 <= float(event.get("t", -1)) <= duration + 1.0:
            problems.append(f"event at t={event.get('t')} falls outside the clip ({duration:.1f}s)")
    if not events:
        problems.append(
            "no events -- annotate `nothing_happened` explicitly if that is the answer, "
            "since a clip with no entry cannot be told from one nobody labelled"
        )
    return problems


def to_truth_file(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The `eval_harness.GroundTruth` schema plus what makes it time-addressable.

    `GroundTruth.load` reads only the keys it knows, so the four extra ones are
    invisible to it -- and they are what let the event axis convert an
    annotated frame index into the timestamp a window is scored against.
    """
    return {
        "object_id": str(payload["object_id"]).strip(),
        "label": str(payload["label"]).strip(),
        "distractor_ids": [str(d) for d in payload.get("distractor_ids", [])],
        "boxes": {str(k): [round(float(v), 4) for v in box] for k, box in payload["boxes"].items()},
        "events": [
            {
                "t": round(float(e["t"]), 3),
                "action": e["action"],
                "location": e.get("location") or None,
            }
            for e in sorted(payload.get("events", []), key=lambda e: float(e["t"]))
        ],
        "notes": payload.get("notes", ""),
        "clip": manifest["clip"],
        "fps": manifest["fps"],
        "frame_stride": manifest["frame_stride"],
        "t_start": manifest["t_start"],
        "duration_s": manifest["duration_s"],
        #: Clockwise degrees the annotator applied. The scorer must apply the
        #: same or every box is 90 degrees wrong.
        "rotate": manifest["rotate"],
        #: Capture resolution, not the preview's. A normalized box means
        #: nothing for identity without the pixels it covers at capture.
        "width": manifest["width"],
        "height": manifest["height"],
    }


# --- Serving -----------------------------------------------------------------


def serve(prepared: Prepared, truth_dir: Path, port: int, open_browser: bool) -> None:
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    out = truth_dir / f"{Path(manifest['clip']).stem}.json"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
            route = unquote(urlparse(self.path).path)
            if route in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif route == "/manifest":
                existing = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
                body = json.dumps({"manifest": manifest, "truth": existing}).encode()
                self._send(200, body, "application/json")
            elif route.startswith("/frames/"):
                name = Path(route).name
                path = prepared.root / name
                # Resolved and compared, so a crafted path cannot walk out of
                # the frame directory -- this listens on localhost, but a
                # server that reads arbitrary files is not worth shipping.
                if path.resolve().parent != prepared.root.resolve() or not path.is_file():
                    self._send(404, b"no such frame", "text/plain")
                    return
                self._send(200, path.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
            if unquote(urlparse(self.path).path) != "/truth":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            problems = validate(payload, manifest)
            if problems:
                self._send(
                    400, json.dumps({"error": "; ".join(problems)}).encode(), "application/json"
                )
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(to_truth_file(payload, manifest), indent=2) + "\n", encoding="utf-8"
            )
            self._send(200, json.dumps({"path": str(out)}).encode(), "application/json")

        def log_message(self, *_: Any) -> None:
            """Silence per-request logging; one line per frame drowns the console."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"clip     {manifest['clip']}  ({prepared.frames} frames)")
    print(f"truth    {out}")
    print(f"open     {url}   (ctrl-c to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


# --- check -------------------------------------------------------------------


def check(truth_dir: Path, clips: Path) -> int:
    """Summarize every annotation, and fail on any that cannot be scored.

    Prints pixels on target for the smallest annotated box, because a clip
    whose object never clears the identity floor will produce a confusing
    event score for a reason that has nothing to do with the model.
    """
    files = sorted(truth_dir.glob("*.json")) if truth_dir.is_dir() else []
    if not files:
        print(f"no annotations under {truth_dir}")
        return 1
    bad = 0
    print(f"{'clip':<34}{'label':<9}{'boxes':>6}{'events':>8}  {'min px':>7}  actions")
    print("-" * 96)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # A saved truth file carries `duration_s` itself, so it is its own
        # manifest for validation -- the same rules that gated the save.
        problems = validate(payload, payload)
        source = clips / payload.get("clip", "")
        if payload.get("clip") and not source.is_file():
            problems.append(f"clip {payload['clip']} not found under {clips}")
        boxes = payload.get("boxes") or {}
        actions = [e["action"] for e in payload.get("events", [])]
        smallest = ""
        if boxes and payload.get("width") and payload.get("height"):
            width, height = int(payload["width"]), int(payload["height"])
            floor = min(_pixels(b, width, height) for b in boxes.values())
            smallest = f"{floor:.0f}{'!' if floor < PIXEL_FLOOR else ''}"
            if floor < PIXEL_CHANCE:
                problems.append(
                    f"smallest box is {floor:.0f} px on target -- below the ~{PIXEL_CHANCE:.0f} px "
                    "at which identity reaches chance, so this clip cannot carry an identity score"
                )
        print(
            f"{path.stem:<34}{payload.get('label', '?'):<9}{len(boxes):>6}{len(actions):>8}"
            f"  {smallest:>7}  {', '.join(sorted(set(actions))) or '-'}"
        )
        for problem in problems:
            bad += 1
            print(f"    FAIL {problem}")
    print("-" * 96)
    print(f"{len(files)} annotated clip(s), {bad} problem(s)")
    return 1 if bad else 0


def _pixels(box: list[float], width: int, height: int) -> float:
    return math.sqrt(max(0.0, (box[2] - box[0]) * width) * max(0.0, (box[3] - box[1]) * height))


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("annotate", "prepare", "check"), help="what to do")
    parser.add_argument("--clip", type=Path, help="the recording to annotate")
    parser.add_argument(
        "--clips", type=Path, default=repo_root / "clips/recordings", help="where recordings live"
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=repo_root / "docs/spikes/grounder-bakeoff/truth",
        help="where annotations are written (tracked in git; clips/ is not)",
    )
    parser.add_argument(
        "--work",
        type=Path,
        default=repo_root / "clips/_annotate",
        help="extracted-frame cache (gitignored with the rest of clips/)",
    )
    parser.add_argument("--annotate-fps", type=float, default=TARGET_ANNOTATE_FPS)
    parser.add_argument("--long-edge", type=int, default=PREVIEW_LONG_EDGE)
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=DEFAULT_ROTATE,
        help="clockwise degrees applied on decode; these recordings need 90 (see DEFAULT_ROTATE)",
    )
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true", help="do not launch a browser")
    args = parser.parse_args()

    if args.command == "check":
        return check(args.truth_dir, args.clips)

    if args.clip is None:
        available = sorted(p.name for p in args.clips.glob("*.mp4"))
        parser.error("--clip is required; available:\n  " + "\n  ".join(available))
    clip = args.clip if args.clip.is_file() else args.clips / args.clip.name
    if not clip.is_file():
        parser.error(f"no such clip: {args.clip}")

    prepared = prepare(clip, args.work, args.annotate_fps, args.long_edge, args.rotate)
    print(f"prepared {prepared.frames} frames in {prepared.root}")
    if args.command == "prepare":
        return 0
    serve(prepared, args.truth_dir, args.port, not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
