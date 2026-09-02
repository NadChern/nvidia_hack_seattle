#!/usr/bin/env python3
"""Measure the whole stack's co-resident VRAM on one cloud GPU.

Every VRAM number in this project was taken on the 8 GB laptop, one model at a
time, and the speech models were never in the budget at all. This answers the
real deployment question: **do C-RADIOv4-H, the grounder, SAM2.1-tiny, Parakeet
and Kokoro fit on one card together, and how big must that card be?**

Two facts force the design:

  * **Weights are not the binding cost for Parakeet.** The speech config records
    it trying to allocate *10.82 GiB* to transcribe one long utterance against
    8 GiB total -- the cost scales with utterance length, capped in production at
    8 s. So each model is measured at its *workload peak*, not just its resident
    weights, and Parakeet's is swept over `--utterance-seconds`.

  * **They cannot share one Python process.** LFM2.5-VL needs transformers >=5.0,
    SAM2 video ships in 4.57.6, Parakeet is NeMo -- the same conflict that made
    every grounding spike run in an isolated venv. In production they are already
    separate services, so the honest measurement is one **process per model** on
    one GPU. That also captures the ~1-2 GiB CUDA context each process pays,
    which a single-process test would hide and the real deployment cannot avoid.

So this runs in two modes:

  worker  -- load ONE model, run its workload, print its torch peak, then hold
             the allocation until released. One `--model`.
  driver  -- spawn a worker per `--models`, each optionally under its own
             `--python` interpreter, wait until all are holding, then read the
             GPU's *total* used memory (the number that decides the card) and
             attribute it. This is the mode you run.

Each loader is wrapped so a missing dependency reports `skipped: <reason>` and
the run continues -- on a fresh Brev box you measure whatever is installed and
the report says what was not, rather than the whole probe dying on one import.

Brev / any CUDA box:
    # one venv with everything is ideal but the transformers split usually
    # needs two; see README.md. Then, from the box:
    python vram_probe.py driver --models radio,sam2,lfm,parakeet,kokoro \\
        --utterance-seconds 8 --out runs/coresident.json

    # or measure one model in isolation:
    python vram_probe.py worker --model parakeet --utterance-seconds 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# GPU accounting -- torch for per-process, nvidia-smi for the true card total
# --------------------------------------------------------------------------- #

def smi_total_used_mib(index: int = 0) -> float:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out.splitlines()[0])


def smi_process_used_mib(pid: int, index: int = 0) -> float:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}",
         "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return float(parts[1])
    return 0.0


# --------------------------------------------------------------------------- #
# Model loaders + workloads. Each returns None (with a printed reason) when its
# dependency stack is absent, so the probe degrades instead of crashing.
# --------------------------------------------------------------------------- #

def _synthetic_rgb(width: int, height: int):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype="uint8"))


def load_radio(_: argparse.Namespace):
    """nvidia/C-RADIOv4-H -- the identity embedder, workload = embed a crop batch."""
    try:
        import torch
        from transformers import AutoModel
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: {exc}"
    model_id = "nvidia/C-RADIOv4-H"
    revision = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    ).to("cuda").eval()

    def workload() -> None:
        # A batch of query crops, the identity path's real shape.
        import numpy as np
        batch = torch.from_numpy(
            np.random.default_rng(1).random((8, 3, 512, 512), dtype="float32")
        ).to("cuda")
        with torch.no_grad():
            model(batch)

    return workload, f"{model_id}@{revision[:8]}"


def load_lfm(_: argparse.Namespace):
    """LiquidAI/LFM2.5-VL-3B -- grounder, workload = one 768px grounding call."""
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: {exc}"
    model_id = "LiquidAI/LFM2.5-VL-3B"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    def workload() -> None:
        image = _synthetic_rgb(768, 768)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Return the bounding box of the keychain."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=64)

    return workload, model_id


def load_sam2(args: argparse.Namespace):
    """facebook/sam2.1-hiera-tiny -- tracker; workload propagates N frames so the
    VOS memory bank (31.6 MiB/frame, spike 2b) shows up, not just the weights."""
    try:
        import numpy as np
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: {exc}"
    model_id = "facebook/sam2.1-hiera-tiny"
    processor = Sam2VideoProcessor.from_pretrained(model_id)
    model = Sam2VideoModel.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

    def workload() -> None:
        frames = [_synthetic_rgb(1024, 768) for _ in range(args.frames)]
        state = processor.init_video_session(video=frames, inference_device="cuda")
        processor.add_inputs_to_inference_session(
            inference_session=state, frame_idx=0, obj_ids=1,
            input_boxes=[[[300, 200, 500, 400]]],
        )
        with torch.no_grad():
            for _ in model.propagate_in_video_iterator(state):
                pass

    _ = np  # keep the import meaningful if the body is edited
    return workload, f"{model_id} (frames={args.frames})"


def load_parakeet(args: argparse.Namespace):
    """nvidia/parakeet-tdt-0.6b-v3 -- ASR. Workload transcribes `--utterance-seconds`
    of audio: the activation, not the weights, is what nearly OOM'd the 8 GB card."""
    try:
        import nemo.collections.asr as nemo_asr
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: {exc}"
    model_id = "nvidia/parakeet-tdt-0.6b-v3"
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id).cuda().eval()

    def workload() -> None:
        seconds = args.utterance_seconds
        tone = 0.1 * np.sin(
            2 * np.pi * 220 * np.arange(int(16_000 * seconds)) / 16_000
        ).astype("float32")
        path = os.path.join(args.scratch, "parakeet_probe.wav")
        sf.write(path, tone, 16_000)
        model.transcribe([path], batch_size=1)

    return workload, f"{model_id} ({args.utterance_seconds}s utterance)"


def load_kokoro(_: argparse.Namespace):
    """hexgrad/Kokoro-82M -- TTS, workload = synthesize one sentence."""
    try:
        from kokoro import KPipeline
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: {exc}"
    pipeline = KPipeline(lang_code="a", device="cuda")

    def workload() -> None:
        for _ in pipeline("Your keys are on the kitchen table by the window.", voice="af_heart"):
            pass

    return workload, "hexgrad/Kokoro-82M"


LOADERS = {
    "radio": load_radio,
    "lfm": load_lfm,
    "sam2": load_sam2,
    "parakeet": load_parakeet,
    "kokoro": load_kokoro,
}


# --------------------------------------------------------------------------- #
# Served models: the two that dominate the budget, and the bake-off contenders.
#
# These are NOT torch workers. A vLLM or SGLang server is its own process with
# its own CUDA context, and -- the part that matters -- its footprint is set by
# `--gpu-memory-utilization`, which is a *policy*, not a need. Point it at a
# 141 GiB card and it will happily reserve 120 GiB of KV cache for a model whose
# weights are 32 GiB.
#
# So "how much does Cosmos use" is not a well-formed question. The two numbers
# that are well-formed, and that this probe extracts from the server's own
# startup log, are:
#
#   weights_mib   -- the floor. Cannot be tuned away.
#   kv_cache_mib  -- what the fraction bought, at *this* utilization setting.
#
# The deployable figure is weights + enough KV for the real concurrency, which
# for this stack is one in-flight grounding call. Reading the co-resident total
# without separating these would size the card against vLLM's appetite rather
# than the workload's requirement, and over-buy by tens of gigabytes.
# --------------------------------------------------------------------------- #

#: `{model}`, `{port}` and `{util}` are filled per run. Kept as templates rather
#: than argv lists so the README's copy-pasteable commands and the probe's own
#: invocation cannot drift apart.
SERVERS = {
    "nemotron": {
        "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "runtime": "vllm",
        "command": (
            "vllm serve {model} --port {port} --gpu-memory-utilization {util} "
            "--kv-cache-dtype fp8 --enable-prefix-caching"
        ),
        "role": "agent reasoning + tool routing",
        "modality": "text",
    },
    "cosmos": {
        "model": "nvidia/Cosmos3-Nano",
        "runtime": "vllm",
        "command": (
            "vllm serve {model} --port {port} --gpu-memory-utilization {util} "
            "--max-model-len 8192"
        ),
        "role": "grounder + event reasoner (incumbent)",
        "modality": "image",
    },
    "qwen3vl4b": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "runtime": "vllm",
        "command": (
            "vllm serve {model} --port {port} --gpu-memory-utilization {util} "
            "--limit-mm-per-prompt '{{\"image\":1}}'"
        ),
        "role": "grounder candidate (Apache-2.0)",
        "modality": "image",
    },
    "qwen3vl8b": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "runtime": "vllm",
        "command": (
            "vllm serve {model} --port {port} --gpu-memory-utilization {util} "
            "--limit-mm-per-prompt '{{\"image\":1}}'"
        ),
        "role": "grounder candidate (Apache-2.0)",
        "modality": "image",
    },
    "mossvl": {
        "model": "OpenMOSS-Team/MOSS-VL-Realtime",
        "runtime": "sglang",
        "command": (
            "python -m sglang.launch_server --model-path {model} --port {port} "
            "--mem-fraction-static {util}"
        ),
        "role": "grounder candidate (Apache-2.0, streaming-native)",
        "modality": "image",
    },
}

#: Ports are assigned per served model so several can co-reside in one run --
#: which is the entire point of the co-resident measurement.
SERVER_PORTS = {name: 8100 + i for i, name in enumerate(sorted(SERVERS))}

#: vLLM and SGLang both announce their two allocations on startup. Parsed rather
#: than inferred, because the difference between them is the whole argument for
#: not sizing the card off the total.
_WEIGHTS_LOG = re.compile(
    r"(?:model weights took|Loading model weights took|weight memory[:=]?)"
    r"\s*([\d.]+)\s*(GiB|GB|MiB)",
    re.IGNORECASE,
)
_KV_LOG = re.compile(
    r"(?:GPU KV cache size|KV cache size|kv cache memory[:=]?)"
    r"\s*[:=]?\s*([\d.,]+)\s*(GiB|GB|MiB|tokens)",
    re.IGNORECASE,
)


def _to_mib(value: str, unit: str) -> float | None:
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return None
    unit = unit.lower()
    if unit in ("gib", "gb"):
        return number * 1024.0
    if unit == "mib":
        return number
    return None  # "tokens" is not a memory figure


def _wait_for_health(port: int, deadline: float, proc) -> bool:
    """Poll until the server answers or dies. 50 GiB of weights load slowly."""
    import urllib.error
    import urllib.request

    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(3.0)
    return False


def _server_workload(port: int, model: str, modality: str, scratch: str) -> float | None:
    """One representative call, so the reported peak includes activations.

    Weights alone under-report: a 768px image is hundreds of vision tokens whose
    activations never appear in a load-time measurement, and the deployment pays
    them on every window.
    """
    import base64
    import urllib.request

    content: list[dict] = []
    if modality == "image":
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (768, 768), (90, 90, 96)).save(buffer, format="JPEG", quality=90)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": "Where are the keys? Return a bounding box."})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer local"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response.read()
    except Exception:  # noqa: BLE001 -- a failed call is reported, not fatal
        return None
    return time.perf_counter() - started


def launch_server(name: str, args: argparse.Namespace):
    """Start one served model, wait for health, exercise it, and leave it up.

    Returns `(proc, record)`. The process is deliberately left running: the
    driver's whole method is to read the card total while everything holds.
    """
    spec = SERVERS[name]
    port = SERVER_PORTS[name]
    command = spec["command"].format(model=spec["model"], port=port, util=args.vllm_util)
    log_path = os.path.join(args.scratch, f"vram_probe_{name}.log")
    print(f"  {name}: launching ({spec['runtime']}), log -> {log_path}")

    handle = open(log_path, "w")
    proc = subprocess.Popen(  # noqa: S602 -- template is ours, not user input
        command, shell=True, stdout=handle, stderr=subprocess.STDOUT, text=True
    )
    ready = _wait_for_health(port, time.time() + args.server_timeout, proc)
    if not ready:
        proc.kill()
        handle.close()
        tail = ""
        try:
            with open(log_path) as fh:
                tail = "".join(fh.readlines()[-6:]).strip()
        except OSError:
            pass
        return None, {
            "model": name,
            "status": f"server never became healthy in {args.server_timeout:.0f}s",
            "log": log_path,
            "tail": tail,
        }

    latency = _server_workload(port, spec["model"], spec["modality"], args.scratch)
    time.sleep(1.0)
    smi_self = smi_process_used_mib(proc.pid, args.gpu)

    weights_mib = kv_mib = None
    try:
        with open(log_path) as fh:
            log = fh.read()
        if match := _WEIGHTS_LOG.search(log):
            weights_mib = _to_mib(*match.groups())
        if match := _KV_LOG.search(log):
            kv_mib = _to_mib(*match.groups())
    except OSError:
        pass

    return proc, {
        "model": name,
        "status": "loaded",
        "tag": spec["model"],
        "runtime": spec["runtime"],
        "role": spec["role"],
        "kind": "server",
        "port": port,
        "gpu_memory_utilization": args.vllm_util,
        # The floor, and what the utilization fraction bought on top of it. Only
        # the first is a property of the model.
        "weights_mib": round(weights_mib, 1) if weights_mib else None,
        "kv_cache_mib": round(kv_mib, 1) if kv_mib else None,
        "smi_process_mib": round(smi_self, 1),
        "workload_latency_s": round(latency, 3) if latency else None,
        "log": log_path,
        "pid": proc.pid,
    }


# --------------------------------------------------------------------------- #
# Worker: load one model, run its workload, report, then hold until released.
# --------------------------------------------------------------------------- #

def run_worker(args: argparse.Namespace) -> int:
    import torch

    loader = LOADERS[args.model]
    torch.cuda.reset_peak_memory_stats()
    workload, tag = loader(args)
    if workload is None:
        print(f"RESULT {json.dumps({'model': args.model, 'status': tag})}", flush=True)
        # Nothing to hold; tell the driver and exit cleanly.
        return 0

    resident = torch.cuda.memory_reserved() / 2**20
    workload()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved() / 2**20
    smi_self = smi_process_used_mib(os.getpid(), args.gpu)

    print("RESULT " + json.dumps({
        "model": args.model, "status": "loaded", "tag": tag,
        "resident_mib": round(resident, 1),
        "workload_peak_mib": round(peak, 1),
        "smi_process_mib": round(smi_self, 1),
        "pid": os.getpid(),
    }), flush=True)

    # Hold the allocation so the driver can read the co-resident total, then
    # release on a line from stdin. max_memory stays reserved by the allocator.
    print("HOLDING", flush=True)
    sys.stdin.readline()
    return 0


# --------------------------------------------------------------------------- #
# Driver: spawn a worker per model, wait until all hold, read the card total.
# --------------------------------------------------------------------------- #

def run_driver(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in LOADERS and m not in SERVERS]
    if unknown:
        print(f"unknown models: {unknown}; known: "
                  f"{sorted(LOADERS)} + {sorted(SERVERS)}", file=sys.stderr)
        return 2

    python_map: dict[str, str] = {}
    for pair in args.python or []:
        name, _, interp = pair.partition("=")
        python_map[name] = interp

    baseline = smi_total_used_mib(args.gpu)
    workers: list[dict] = []
    procs: list[subprocess.Popen] = []

    try:
        for name in models:
            if name in SERVERS:
                # A server holds its own allocation by simply staying up; it
                # does not speak the worker protocol.
                proc, record = launch_server(name, args)
                if proc is not None:
                    procs.append(proc)
                workers.append(record)
                print(f"  {name}: {record.get('status')}")
                continue
            interp = python_map.get(name, sys.executable)
            cmd = [
                interp, os.path.abspath(__file__), "worker", "--model", name,
                "--gpu", str(args.gpu), "--frames", str(args.frames),
                "--utterance-seconds", str(args.utterance_seconds),
                "--scratch", args.scratch,
            ]
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                bufsize=1,
            )
            procs.append(proc)

            # Read until this worker either holds, or reports it was skipped.
            record: dict | None = None
            while True:
                line = proc.stdout.readline()
                if not line:
                    record = {"model": name, "status": "worker exited early"}
                    break
                line = line.strip()
                if line.startswith("RESULT "):
                    record = json.loads(line[len("RESULT "):])
                elif line == "HOLDING":
                    break
                else:
                    print(f"[{name}] {line}")
            workers.append(record or {"model": name, "status": "no result"})
            loaded = record and record.get("status") == "loaded"
            print(f"  {name}: {'holding' if loaded else record.get('status')}")

        # Every live worker is now holding its allocation. Snapshot the card.
        time.sleep(1.0)
        co_resident = smi_total_used_mib(args.gpu)

        loaded = [w for w in workers if w.get("status") == "loaded"]
        sum_smi = sum(w.get("smi_process_mib", 0.0) for w in loaded)
        report = {
            "gpu": args.gpu,
            "models_requested": models,
            "utterance_seconds": args.utterance_seconds,
            "sam_frames": args.frames,
            "baseline_used_mib": round(baseline, 1),
            "co_resident_used_mib": round(co_resident, 1),
            "co_resident_over_baseline_mib": round(co_resident - baseline, 1),
            "sum_of_per_process_mib": round(sum_smi, 1),
            "implied_shared_or_unattributed_mib": round(co_resident - baseline - sum_smi, 1),
            # The 16/24 GiB pair answered the RTX 5080 question, which the
            # all-local decision retired. These are the Nebius cards.
            "fits_48gib_l40s": co_resident <= 48 * 1024,
            "fits_80gib_h100": co_resident <= 80 * 1024,
            "fits_96gib_rtx_pro_6000": co_resident <= 96 * 1024,
            "fits_141gib_h200": co_resident <= 141 * 1024,
            # Weights are the floor no utilization setting can tune away; the
            # total above includes KV cache vLLM took because it was offered.
            "sum_of_server_weights_mib": round(
                sum(w.get("weights_mib") or 0.0 for w in loaded), 1
            ),
            "per_model": loaded,
            "skipped": [w for w in workers if w.get("status") != "loaded"],
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text)
        return 0
    finally:
        for proc in procs:
            try:
                if proc.stdin:
                    proc.stdin.write("\n")
                    proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass
        for proc in procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpu", type=int, default=0)
    common.add_argument("--frames", type=int, default=20,
                        help="SAM2 frames to propagate; 20 ~= a 10 s sighting at 2 fps")
    common.add_argument("--utterance-seconds", type=float, default=8.0,
                        help="Parakeet workload length; production caps this at 8 s")
    common.add_argument("--scratch", default="/tmp")
    common.add_argument("--vllm-util", type=float, default=0.25,
                        help="gpu-memory-utilization per served model. Deliberately "
                             "NOT vLLM's 0.9 default: several servers share this card, "
                             "and 0.9 means the first one takes it all")
    common.add_argument("--server-timeout", type=float, default=1800.0,
                        help="seconds to wait for a server to become healthy; "
                             "50 GiB of weights load slowly on first run")

    w = sub.add_parser("worker", parents=[common])
    w.add_argument("--model", required=True, choices=sorted(LOADERS))

    d = sub.add_parser("driver", parents=[common])
    d.add_argument("--models", required=True,
                   help="comma-separated. In-process: radio,sam2,lfm,parakeet,kokoro. "
                        "Served: nemotron,cosmos,qwen3vl4b,qwen3vl8b,mossvl")
    d.add_argument("--python", action="append",
                   help="per-model interpreter, e.g. lfm=/path/tf5/bin/python (repeatable)")
    d.add_argument("--out")

    args = ap.parse_args()
    if args.mode == "worker":
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
