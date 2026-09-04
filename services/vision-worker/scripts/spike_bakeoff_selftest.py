#!/usr/bin/env python3
"""Known-answer rehearsal for the bake-off harness — no GPU, no model, no money.

`spike_grounder_bakeoff.py` decides which grounder ships. It had never been run
against a server of any kind, and the first time it would have been was on a
rented H200 at $4.50/hour with 156 model calls queued behind it. A crash forty
minutes in costs a re-rent; a *silent* scoring bug costs the decision.

So this serves a scripted OpenAI-shaped endpoint whose answers are derived from
the ground truth itself, and asserts the numbers the harness reports back. An
arm that replies with the exact truth box must score IoU 1.0; one that replies
with the truth box transposed must be *detected* as yxyx rather than scored as
bad; one that claims a placement without emitting a box must score as silence,
because that is what `CosmosReasoner._parse` does in production.

What it covers: both axes end to end, the documented `uv run --isolated`
command lines, box parsing in all three accepted formats, `NO_OBJECT`, the
coordinate-convention auto-detection, the containment-vs-IoU diagnostic, the
event scorer's five rates, the message you get when the server is down, and
the guard that abandons a run rather than paying a timeout per remaining call.

What it cannot cover: whether a real model answers well. That is the bake-off.

    cd services/vision-worker
    uv run --with pillow --with pillow-heif --with av --with numpy \\
      python scripts/spike_bakeoff_selftest.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
sys.path.insert(0, str(SCRIPTS))

import spike_grounder_bakeoff as S  # noqa: E402

TRUTH_DIR = REPO / "docs/spikes/grounder-bakeoff/truth"
CLIPS = REPO / "clips/recordings"
DATASET = REPO / "clips/identity-probe"
CACHE = REPO / "clips/spike1-arms/_cache"

#: The production schedule, and the only one the expectations below are true for.
WINDOW_S, INTERVAL_S, MAX_FRAMES = 20.0, 7.0, 8


# --- What the scripted arm knows ---------------------------------------------


#: Somewhere plausible, for a scenario that must emit a box it has no truth for.
FALLBACK_BOX = (0.30, 0.30, 0.70, 0.70)


@dataclass
class Answer:
    """Everything the fake needs to answer one call in ground-truth terms.

    `label` is per-clip, not per-run: the corpus is three keys clips and one
    wallet clip, and the harness prompts each one with its own noun. An arm
    that answers with a different noun than it was asked scores `unknown`,
    which is correct and is its own scenario below.
    """

    box: tuple[float, float, float, float] | None
    action: str = "nothing_happened"
    absent: bool = False
    label: str = "keys"


@dataclass
class Script:
    """The current scenario: a mode, an answer queue, and where we are in it."""

    mode: str = "perfect"
    answers: list[Answer] = field(default_factory=list)
    served: int = 0
    overrun: int = 0

    def next(self) -> Answer:
        if self.served >= len(self.answers):
            # The harness asked for more calls than the scenario was built for,
            # which means this file's model of the run has drifted from the
            # harness's. Loud, because a wrong sequence would silently score
            # every answer against the wrong image.
            self.overrun += 1
            return Answer(None)
        answer = self.answers[self.served]
        self.served += 1
        return answer

    def fails(self) -> bool:
        """Is this call a scripted transport failure?

        `http_error` is a server that has died and is not coming back, which
        the harness must abandon; `flaky` alternates, which it must ride out.
        Both are 500s rather than a refused connection, because that is the
        shape the harness cannot distinguish from a model that cannot answer.
        """
        if self.mode == "http_error":
            return True
        return self.mode == "flaky" and self.served % 2 == 1


def _quad(box: tuple[float, float, float, float], order: str) -> str:
    x0, y0, x1, y1 = (round(v * S.COORD_SCALE) for v in box)
    a, b, c, d = (y0, x0, y1, x1) if order == "yxyx" else (x0, y0, x1, y1)
    return f"[{a}, {b}, {c}, {d}]"


def _shrink(box: tuple[float, float, float, float], factor: float) -> tuple:
    """Same centre, `factor` of the linear extent — the extent failure, exactly."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w, half_h = (box[2] - box[0]) * factor / 2, (box[3] - box[1]) * factor / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def grounding_reply(mode: str, answer: Answer, label: str) -> str:
    box = answer.box
    assert box is not None
    if mode == "perfect":
        return f"<ref>{label}</ref><box>{_quad(box, 'xyxy')}</box>"
    if mode == "tight":
        return f"<ref>{label}</ref><box>{_quad(_shrink(box, 0.5), 'xyxy')}</box>"
    if mode == "yxyx":
        return f"<ref>{label}</ref><box>{_quad(box, 'yxyx')}</box>"
    if mode == "bbox_json":
        x0, y0, x1, y1 = (round(v * S.COORD_SCALE) for v in box)
        return f'Here is what I found.\n{{"label": "{label}", "bbox_2d": [{x0}, {y0}, {x1}, {y1}]}}'
    if mode == "decline":
        return "NO_OBJECT"
    if mode == "prose":
        return "I can see the object resting on a wooden surface near the window."
    raise SystemExit(f"unknown grounding mode {mode}")


def event_reply(mode: str, answer: Answer, label: str) -> str:
    """A window reply: grounding tags first, then the JSON action tail.

    Same shape as production — `_parse_action_tail` reads the last balanced
    `[...]`, so the tail must come after the box and must not nest.
    """
    tail = '[{{"label": "{label}", "action": "{action}", "location": {location}}}]'

    def compose(box, action, location='"on the white desk"'):
        head = f"<ref>{label}</ref><box>{_quad(box, 'xyxy')}</box>\n" if box else ""
        return head + tail.format(label=label, action=action, location=location)

    if mode == "honest":
        if answer.absent:
            # Out of frame: no box, and the pipeline records nothing. Saying so
            # in the tail as well is what an honest model does.
            return compose(None, "nothing_happened", "null")
        # Visible but unboxed in truth is a sparse-annotation fact, not an
        # absence: a real model would still box it, and no IoU is scored there.
        return compose(answer.box or FALLBACK_BOX, answer.action)
    if mode == "phantom":
        return compose(FALLBACK_BOX, "placed")
    if mode == "silent":
        # Claims the placement and forgets the box. Production drops the whole
        # window, so this must score as silence, not as a placement.
        return compose(None, "placed")
    if mode == "handling":
        return compose(answer.box or FALLBACK_BOX, "picked_up", "null")
    raise SystemExit(f"unknown event mode {mode}")


# --- The server --------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    script: Script

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        content = payload["messages"][0]["content"]
        images = sum(1 for block in content if block.get("type") == "image_url")
        answer = self.script.next()
        if self.script.fails():
            self.send_error(500, "scripted failure")
            return
        # "wrong-label" answers about keys whatever it was asked, which is the
        # one mode that must ignore the answer's own label.
        label = "keys" if self.script.mode == "wrong_label" else answer.label
        mode = "honest" if self.script.mode in ("wrong_label", "flaky") else self.script.mode
        body = (
            grounding_reply(mode, answer, label)
            if images == 1
            else event_reply(mode, answer, label)
        )
        reply = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": body}}]}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *_args) -> None:
        """Silence — the harness's own output is what we are reading."""


# --- Building the answer queues ----------------------------------------------


def grounding_answers() -> list[Answer]:
    """Truth boxes in `collect()`'s order, normalised the way the harness does.

    Order, not content, is what keys the fake to the image: `collect()` is
    documented as stable precisely so arms run on different days see the same
    images in the same sequence, and that is what makes a positional script
    valid here.
    """
    import numpy as np

    answers: list[Answer] = []
    for _name, image_path, cached in S.collect(DATASET, CACHE, None):
        image = S.load_rgb(image_path)  # exif_transpose'd, as the harness does
        px = np.load(cached)["box"]
        answers.append(
            Answer(
                box=(
                    float(px[0]) / image.width,
                    float(px[1]) / image.height,
                    float(px[2]) / image.width,
                    float(px[3]) / image.height,
                )
            )
        )
    return answers


def event_answers() -> tuple[list[Answer], dict]:
    """One answer per window, in `run_events`'s order, plus the corpus shape.

    Uses the harness's own `decode_clip` and `build_windows` rather than
    reimplementing the schedule — a self-test that models the run differently
    from the run is testing its own model.
    """
    answers: list[Answer] = []
    shape = {
        "windows": 0,
        "absent": 0,
        "quiet": 0,
        "placed": 0,
        "carried": 0,
        "clips": 0,
        "placements": 0,
    }
    for path in sorted(TRUTH_DIR.glob("*.json")):
        truth = S.load_truth(path)
        clip = CLIPS / truth.clip
        if not clip.is_file():
            continue
        shape["clips"] += 1
        shape["placements"] += sum(1 for e in truth.events if e["action"] == "placed")
        frames = S.decode_clip(clip, truth.frame_stride, 768, truth.rotate)
        for window in S.build_windows(
            frames, window_s=WINDOW_S, interval_s=INTERVAL_S, max_frames=MAX_FRAMES
        ):
            last = window.frames[-1].index
            action, absent = truth.expected(window.start, window.end, last)
            answers.append(
                Answer(box=truth.box_at(last), action=action, absent=absent, label=truth.label)
            )
            shape["windows"] += 1
            if absent:
                shape["absent"] += 1
            elif action == "nothing_happened":
                shape["quiet"] += 1
            elif action == "placed":
                shape["placed"] += 1
            elif action == "carried":
                shape["carried"] += 1
    return answers, shape


# --- Running one scenario ----------------------------------------------------


def run_harness(port: int, task: str, out: Path, extra: list[str]) -> tuple[int, str]:
    """Invoke the harness exactly as the README tells a human to.

    Including `uv run --isolated`, because "the deps the docs name are enough"
    is one of the things being tested — the event axis needs `av` and the
    grounding axis needs `pillow-heif`, and neither is a project dependency.
    """
    deps = ["--with", "pillow", "--with", "numpy"]
    deps += ["--with", "av"] if task == "events" else ["--with", "pillow-heif"]
    data = [] if task == "events" else ["--dataset", str(DATASET), "--cache", str(CACHE)]
    command = [
        "uv", "run", "--isolated", *deps,
        "python", str(SCRIPTS / "spike_grounder_bakeoff.py"),
        "--arm", "qwen3-vl-4b",
        "--task", task,
        "--check-prompt-drift",
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--out", str(out),
        *data,
        *extra,
    ]  # fmt: skip
    done = subprocess.run(command, cwd=SCRIPTS.parent, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


def closed_port() -> int:
    """A port nothing is listening on, that will *refuse* rather than drop."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class Case:
    """One scenario.

    `expect` is read from the written report, so it only applies to a run that
    finishes; `exit_code`, `in_output` and `calls` are how the failure
    scenarios -- which deliberately write nothing -- are asserted instead.
    """

    name: str
    task: str
    mode: str
    expect: dict
    extra: list[str] = field(default_factory=list)
    exit_code: int = 0
    in_output: str = ""
    calls: int | None = None


def check(report: dict, expect: dict) -> list[str]:
    """Compare with a tolerance, and say what was got — not just that it failed."""
    problems = []
    for key, wanted in expect.items():
        got = report.get(key)
        if isinstance(wanted, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - wanted) < 5e-3
        else:
            ok = got == wanted
        if not ok:
            problems.append(f"{key}: expected {wanted!r}, got {got!r}")
    return problems


def main() -> int:
    for needed in (TRUTH_DIR, CLIPS, DATASET, CACHE):
        if not needed.exists():
            print(f"missing {needed} -- this rehearsal needs the local corpus")
            return 2

    print("building the answer queues from ground truth ...")
    ground = grounding_answers()
    events, shape = event_answers()
    print(
        f"  grounding: {len(ground)} images\n"
        f"  events:    {shape['windows']} windows over {shape['clips']} clips "
        f"({shape['absent']} absent, {shape['quiet']} quiet, "
        f"{shape['placed']} placed, {shape['carried']} carried)"
    )

    n_img, n_win = len(ground), shape["windows"]
    quiet, absent = shape["quiet"], shape["absent"]
    # An arm that always answers `nothing_happened` is right on exactly the
    # windows where truth says nothing happened -- absent ones included.
    silent_accuracy = round((quiet + absent) / n_win, 4)
    # Every placement is catchable in this corpus, and exactly one clip is the
    # wallet -- so the wrong-label arm loses exactly its placement.
    placements_catchable = shape["placements"]

    # fmt: off
    cases = [
        Case("grounding/perfect", "grounding", "perfect", {
            "scored": n_img, "no_box": 0, "errors": 0,
            "mean_iou": 1.0, "median_iou": 1.0,
            "iou_ge_0p5": n_img, "mean_containment": 1.0, "mean_area_ratio": 1.0,
            "coord_order_used": "xyxy",
        }),
        Case("grounding/tight-box", "grounding", "tight", {
            "scored": n_img, "no_box": 0,
            # Right object, half the extent: containment stays 1, IoU is the
            # area ratio. This is the pair the extent rule exists to separate.
            "mean_containment": 1.0, "mean_area_ratio": 0.25, "mean_iou": 0.25,
            "iou_ge_0p5": 0,
        }),
        Case("grounding/transposed", "grounding", "yxyx", {
            "coord_order_used": "yxyx", "mean_iou": 1.0, "no_box": 0,
        }),
        Case("grounding/bbox-json", "grounding", "bbox_json", {
            "mean_iou": 1.0, "no_box": 0, "scored": n_img,
        }),
        Case("grounding/NO_OBJECT", "grounding", "decline", {
            "no_box": n_img, "scored": 0, "mean_iou": 0.0, "errors": 0,
        }),
        Case("grounding/prose-only", "grounding", "prose", {
            "no_box": n_img, "scored": 0,
        }),
        Case("grounding/--limit", "grounding", "perfect", {
            "images": 5, "answered": 5, "scored": 5, "mean_iou": 1.0,
        }, extra=["--limit", "5"]),
        Case("events/honest", "events", "honest", {
            "windows": n_win, "errors": 0,
            "window_accuracy": 1.0, "macro_f1": 1.0,
            "quiet_windows": quiet, "false_placed": 0, "false_handling": 0,
            "absent_windows": absent, "phantom_boxes": 0,
            # It boxes whenever the object is in frame and stays silent when it
            # is not, so silence and absence coincide exactly.
            "no_box_windows": absent,
            "placement_recall": 1.0, "mean_window_iou": 1.0,
            "coord_order_used": "xyxy",
        }),
        Case("events/wrong-label", "events", "wrong_label", {
            # Perfect answers about "keys" on every clip, including the wallet
            # one. The tail is keyed by the label the window asked for, so that
            # clip scores `unknown` throughout and its placement is never found
            # -- one noun costs a whole recording.
            "windows": n_win,
            "placement_recall": round(1 - 1 / max(1, placements_catchable), 4),
        }),
        Case("events/phantom", "events", "phantom", {
            "windows": n_win,
            "false_placed": quiet, "false_placed_rate": 1.0,
            "phantom_boxes": absent, "phantom_rate": 1.0,
            "false_handling": 0,
            # It calls every window a placement, so it "finds" all of them --
            # which is the point of scoring recall and false rate together.
            "placement_recall": 1.0,
        }),
        Case("events/box-less claim", "events", "silent", {
            "windows": n_win, "no_box_windows": n_win,
            "placement_recall": 0.0, "false_placed": 0, "phantom_boxes": 0,
            "window_accuracy": silent_accuracy,
        }),
        Case("events/always-handling", "events", "handling", {
            "windows": n_win,
            "false_handling": quiet, "false_handling_rate": 1.0,
            "false_placed": 0, "placement_recall": 0.0,
        }),
        # The guard. `--timeout` is per call, so without it a server that dies
        # -- or a mistyped `--base-url` -- runs the whole axis at one timeout
        # each on a card billed by the hour. Both axes must stop at the limit,
        # and neither may write a report: a partial run is not comparable.
        Case("grounding/server-dies", "grounding", "http_error", {},
             exit_code=1, calls=S.CONSECUTIVE_ERROR_LIMIT,
             in_output="abandoned after 5 consecutive transport failures"),
        Case("events/server-dies", "events", "http_error", {},
             exit_code=1, calls=S.CONSECUTIVE_ERROR_LIMIT,
             in_output="abandoned after 5 consecutive transport failures"),
        # ... and the other half of the guard: the counter resets on success,
        # so an arm that fails every other call still completes. Half the
        # windows are lost, which is a result rather than a crash.
        Case("events/flaky", "events", "flaky", {
            "windows": n_win // 2, "errors": n_win - n_win // 2,
        }, calls=n_win),
    ]
    # fmt: on

    script = Script()
    Handler.script = script
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"scripted arm listening on 127.0.0.1:{port}\n")

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        for case in cases:
            script.mode = case.mode
            script.answers = ground if case.task == "grounding" else events
            script.served = 0
            script.overrun = 0
            code, output = run_harness(port, case.task, out, case.extra)
            problems: list[str] = []
            if code != case.exit_code:
                problems.append(f"exit {code}, expected {case.exit_code}")
            elif case.exit_code:
                if out.exists():
                    problems.append("wrote a report for a run it abandoned")
            elif not out.exists():
                problems.append("no report written")
            else:
                report = json.loads(out.read_text())
                problems += check(report, case.expect)
                if report.get("prompt_drift", "").startswith("DRIFTED"):
                    problems.append("prompt drift reported")
            if case.in_output and case.in_output not in output:
                problems.append(f"output does not say {case.in_output!r}")
            if case.calls is not None and script.served != case.calls:
                problems.append(f"{script.served} calls, expected {case.calls}")
            if script.overrun:
                problems.append(f"{script.overrun} calls past the end of the script")
            status = "PASS" if not problems else "FAIL"
            print(f"{status}  {case.name}  ({script.served} calls)")
            for problem in problems:
                print(f"        {problem}")
            if problems:
                failures.append(case.name)
                print("        --- harness output ---")
                for line in output.strip().splitlines()[-15:]:
                    print(f"        {line}")
            out.unlink(missing_ok=True)

        # The message a human meets first, on a box where the server died.
        #
        # A *closed* port, not an unused well-known one: 127.0.0.1 refuses a
        # connection to a closed ephemeral port in ~25 ms, whereas a filtered
        # port (9, discard) is silently dropped and every call waits out
        # `--timeout`. Which is itself worth knowing -- see the note in the
        # spike README about pointing the harness at the wrong URL. The guard
        # bounds even the filtered case at five timeouts rather than a whole axis.
        script.answers = []
        code, output = run_harness(closed_port(), "events", out, [])
        if code == 1 and "is the server up" in output:
            print("PASS  events/server-down")
        else:
            failures.append("events/server-down")
            print(f"FAIL  events/server-down  (exit {code}, no 'is the server up' message)")

    server.shutdown()
    print()
    if failures:
        print(f"{len(failures)} scenario(s) failed: {', '.join(failures)}")
        return 1
    print(f"all {len(cases) + 1} scenarios pass -- the harness is safe to point at a real arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
