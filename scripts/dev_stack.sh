#!/usr/bin/env bash
#
# The whole assistant, on one command, on a Mac or on Linux.
#
# Seven processes have to agree about ports, credentials, frame rates and which
# model runtime this machine can actually run. Started by hand that is seven
# terminals and a handful of environment variables that fail quietly when they
# disagree -- a rate mismatch does not error, it silently rescales every
# threshold in the state machine. This starts them in dependency order, waits
# for each to be genuinely ready, and stops all of them together.
#
#   ./scripts/dev_stack.sh                    everything this machine can run
#   ./scripts/dev_stack.sh --skip speech      leave one out (repeatable, or comma-separated)
#   ./scripts/dev_stack.sh --fixture          no real models, even where they exist
#   ./scripts/dev_stack.sh --no-sync          skip dependency install (faster restarts)
#   ./scripts/dev_stack.sh --keep-livekit     leave the LiveKit container running on exit
#   ./scripts/dev_stack.sh --allow-lan        permit listeners beyond loopback (implied
#                                             by VMA_BIND_ADDR, which real glasses on
#                                             Wi-Fi need; see docs/07)
#
# Every VMA_* value below is a default, not an override: anything already set
# in your environment wins, so
#
#   VMA_DETECTION_LABELS="a red stapler" ./scripts/dev_stack.sh
#
# does what it looks like.
#
# Ctrl-C stops everything this script started. Logs are written to logs/.

set -euo pipefail

# --- What this machine can run ---------------------------------------------
#
# Two decisions, both made from the hardware rather than from a flag, because
# getting them wrong is slow rather than loud:
#
#   speech  -- which dependency group carries a real Parakeet and Kokoro.
#              `mlx` on Apple Silicon, `cuda` on an NVIDIA Linux box, neither
#              elsewhere. The service itself then probes what imported and
#              picks its backend; this only decides what gets installed.
#
#   vision  -- whether the real detector is worth installing. Yes wherever
#              there is an accelerator, which now includes Apple Silicon;
#              elsewhere the fixture detector, because YOLOE on CPU cores is
#              seconds per frame and the CUDA wheels are gigabytes. The
#              detector then picks its own device -- CUDA, Metal, or CPU.

SKIP=""
FORCE_FIXTURE=0
DO_SYNC=1
KEEP_LIVEKIT=0
ALLOW_LAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip) SKIP="$SKIP,${2:?--skip needs a service name}"; shift ;;
    --fixture) FORCE_FIXTURE=1 ;;
    --no-sync) DO_SYNC=0 ;;
    --keep-livekit) KEEP_LIVEKIT=1 ;;
    --allow-lan) ALLOW_LAN=1 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"

# What the services listen on. Loopback by default, which is the posture
# docs/07-Privacy-and-Security.md asks for: nothing is published to a network
# unless somebody says so.
#
# Set VMA_BIND_ADDR=0.0.0.0 when real glasses connect over Wi-Fi rather than
# over `adb reverse`. Under WSL2 the services sit behind the VM's own NAT, so
# reaching them from the wireless network needs all three of:
#   - this bind address, so something is listening off-loopback at all;
#   - a Windows portproxy from the laptop's Wi-Fi address into the WSL IP;
#   - a Windows firewall rule for those ports.
# Binding alone changes nothing, and neither does forwarding alone.
BIND_ADDR="${VMA_BIND_ADDR:-127.0.0.1}"
# Health checks and service-to-service URLs must use an address the listener
# actually owns. Loopback works for wildcard binds; an exact trusted-LAN bind
# (the GN100 posture) must be probed through that exact address instead.
PROBE_HOST="$BIND_ADDR"
case "$PROBE_HOST" in 0.0.0.0|::|"[::]") PROBE_HOST="127.0.0.1" ;; esac
ENV_FILE="$REPO_ROOT/.env"

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
info() { printf '   %s\n' "$1"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '\n\033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

skipped() { case ",$SKIP," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Parallel arrays rather than an associative one: macOS still ships bash 3.2,
# where `declare -A` does not exist. Everything here stays inside that dialect
# so the script behaves the same on a stock Mac as on Linux.
PIDS=()
NAMES=()
STARTED_LIVEKIT=0
SHUTTING_DOWN=0

cleanup() {
  trap - EXIT INT TERM
  local i pid name
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    printf '\n'
    bold "Stopping"
    # Reverse order: consumers before the things they consume, so the last
    # lines in each log are a clean shutdown rather than a relay error.
    for (( i=${#PIDS[@]}-1; i>=0; i-- )); do
      pid="${PIDS[$i]}"
      name="${NAMES[$i]}"
      if kill -0 "$pid" 2>/dev/null; then
        info "$name"
        kill -TERM "$pid" 2>/dev/null || true
        # And the server underneath it. Each service runs as `uv run <server>`,
        # so the tracked pid is uv and the server is its child -- and uv did
        # not pass the signal down, measured as a vision worker still
        # reconnecting to a gateway that had already exited. Signalling the
        # child directly does not depend on uv's behaviour either way; the
        # server receiving TERM twice is harmless.
        pkill -TERM -P "$pid" 2>/dev/null || true
      fi
    done
    # One grace period for all of them together rather than one each, and long
    # enough for the slowest: the vision worker drains its relay for up to
    # fifteen seconds before exiting, and cutting that short is what turned a
    # clean stop into an orphan.
    for _ in $(seq 1 50); do
      local alive=0
      for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=1
      done
      [[ "$alive" == "0" ]] && break
      sleep 0.5
    done
    # Children before parents. Killing a parent first orphans whatever it
    # spawned, and an orphaned server keeps its port -- which the next run
    # then reports as "port 8082 is already in use" with nothing holding it
    # that the user can see.
    for pid in "${PIDS[@]}"; do
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    done
    # Nothing should still hold a port by now. Reported rather than fixed
    # silently: the next run's port check would name the port but not what is
    # holding it, and by then this script is gone.
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        warn "pid $pid survived SIGKILL"
      fi
    done
  fi

  if [[ "$STARTED_LIVEKIT" == "1" && "$KEEP_LIVEKIT" == "0" ]]; then
    info "livekit"
    docker compose -f "$REPO_ROOT/compose.dev.yaml" down >/dev/null 2>&1 || true
  fi
}

# Ctrl-C at a terminal signals the whole process group, so the children are
# usually already stopping by the time this runs -- `cleanup` is what covers
# the other case, a plain `kill` of this script alone. Either way the explicit
# `exit 0` matters: without it a signal handler *returns* to the supervision
# loop below, which then finds the processes it just stopped missing and
# reports a clean Ctrl-C as a crash.
on_signal() {
  SHUTTING_DOWN=1
  cleanup
  exit 0
}
trap cleanup EXIT
trap on_signal INT TERM

# --- Preflight --------------------------------------------------------------

command -v uv >/dev/null 2>&1 || fail "uv is not installed -- see https://docs.astral.sh/uv/"
# uv can supply its own interpreter, so a machine can have uv and no system
# python3. This script needs one for the port probes and the listener check.
command -v python3 >/dev/null 2>&1 || fail "python3 is not on PATH"
command -v curl >/dev/null 2>&1 || fail "curl is not on PATH"

OS="$(uname -s)"
ARCH="$(uname -m)"

has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

gpu_vram_mib() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' ' | sed -n '/^[0-9][0-9]*$/p'
}

SPEECH_GROUP=""
SPEECH_TTS_BACKEND="${VMA_TTS_BACKEND:-auto}"
VISION_MODELS=0
DEPTH_KIND="none"
PLATFORM="$OS/$ARCH"

case "$OS" in
  Darwin)
    if [[ "$ARCH" == "arm64" ]]; then
      SPEECH_GROUP="mlx"
      # The real detector installs here. PyPI publishes macOS arm64 wheels
      # for the exact torch and torchvision versions this service pins, and
      # only the cu126 index -- Linux and Windows wheels only -- ever stood
      # in the way; `pyproject.toml` now takes them from PyPI on darwin.
      #
      # Depth stays off, unlike the CUDA path. MoGe is a second model on a
      # device that has to run the detector first, and unlike the detector it
      # would be competing for the same Metal queue. `VMA_DEPTH_KIND=moge`
      # turns it on for anyone who wants to measure that rather than assume.
      VISION_MODELS=1
      PLATFORM="macOS (Apple Silicon)"
    else
      # The mlx group's entries are marked `sys_platform == 'darwin'`, but mlx
      # itself publishes no Intel wheels, so requesting it here would fail the
      # sync rather than degrade.
      PLATFORM="macOS (Intel)"
    fi
    ;;
  Linux)
    if has_nvidia_gpu; then
      SPEECH_GROUP="cuda"
      VISION_MODELS=1
      DEPTH_KIND="moge"
      PLATFORM="Linux + NVIDIA ($ARCH)"
    else
      PLATFORM="Linux, no GPU ($ARCH)"
    fi
    ;;
esac

# Asking for the real detector explicitly overrides the hardware guess -- and
# has to reach the install step, not just the environment, or the service would
# start with VMA_DETECTOR_KIND=yoloe against a venv that has no ultralytics in
# it. Somebody willing to wait seconds per frame on a CPU can have that; on a
# Mac it cannot be installed at all, and `sync_optional` degrades and says so.
[[ "${VMA_DETECTOR_KIND:-}" == "yoloe" ]] && VISION_MODELS=1
[[ "${VMA_DETECTOR_KIND:-}" == "fixture" ]] && { VISION_MODELS=0; DEPTH_KIND="none"; }

[[ "$FORCE_FIXTURE" == "1" ]] && { VISION_MODELS=0; SPEECH_GROUP=""; }

# MoGe CUDA warmup beside Speech has exhausted this 8 GB GPU. On NVIDIA GPUs
# below 12 GB, the default remains fixture Vision; an explicit
# VMA_ENABLE_CONSTRAINED_VISION=true selects the smaller YOLOE-11s detector
# and YOLO26 metric-depth model so the glasses Console can show real boxes and
# ranges without loading MoGe. Otherwise force the deterministic Agent
# when its configured LLM is local. Parakeet STT and Kokoro TTS have been
# validated together on this laptop and remain enabled. An explicitly opted-in
# external endpoint remains a real Agent without local model pressure. The
# full local-model benchmark is available through the oversubscription override.
RESOURCE_SAFE_MODE=0
VRAM_MIB="$(gpu_vram_mib || true)"
AGENT_BACKEND="${VMA_AGENT_BACKEND:-llm}"
LLM_BASE_URL="${VMA_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
LOCAL_LLM=0
case "$LLM_BASE_URL" in
  http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*|https://localhost:*|\
  http://\[::1\]:*|https://\[::1\]:*|http://0.0.0.0:*|https://0.0.0.0:*) LOCAL_LLM=1 ;;
esac
if [[ "$AGENT_BACKEND" != "stub" && "$LOCAL_LLM" == "0" ]] && ! skipped agent; then
  [[ "${VMA_ALLOW_EXTERNAL_LLM:-false}" == "true" ]] \
    || fail "external Agent endpoint requires VMA_ALLOW_EXTERNAL_LLM=true"
  [[ -n "${VMA_LLM_API_KEY:-}" ]] \
    || fail "external Agent endpoint requires a non-empty VMA_LLM_API_KEY"
fi
if [[ -n "$VRAM_MIB" && "$VRAM_MIB" -lt 12288 && -n "$SPEECH_GROUP" \
      && "${VMA_ALLOW_RESOURCE_OVERSUBSCRIPTION:-false}" != "true" ]] \
      && ! skipped speech; then
  RESOURCE_SAFE_MODE=1
  if ! skipped vision; then
    if [[ "${VMA_ENABLE_CONSTRAINED_VISION:-false}" == "true" ]]; then
      VISION_MODELS=1
      DEPTH_KIND="yolo"
      warn "${VRAM_MIB} MiB GPU: enabling constrained YOLOE + YOLO depth beside Speech"
    else
      VISION_MODELS=0
      DEPTH_KIND="none"
      warn "${VRAM_MIB} MiB GPU: using fixture Vision to reserve CUDA for Speech STT"
      warn "VMA_ENABLE_CONSTRAINED_VISION=true enables live boxes and depth with smaller models"
    fi
  fi
  if [[ "$AGENT_BACKEND" != "stub" && "$LOCAL_LLM" == "1" ]] && ! skipped agent; then
    AGENT_BACKEND="stub"
    warn "using the stub Agent because the configured LLM is local"
    warn "use the external MiniCPM profile for a real LLM without local GPU pressure"
  fi
  warn "VMA_ALLOW_RESOURCE_OVERSUBSCRIPTION=true enables the full local-model profile"
fi

if [[ "$VISION_MODELS" == "1" ]]; then
  DETECTOR_KIND="yoloe"
else
  DETECTOR_KIND="fixture"
  DEPTH_KIND="none"
fi

bold "This machine"
info "platform:  $PLATFORM"
info "detector:  $DETECTOR_KIND        depth: $DEPTH_KIND"
if [[ "$SPEECH_TTS_BACKEND" == "stub" && -n "$SPEECH_GROUP" ]]; then
  info "speech:    $SPEECH_GROUP STT, stub TTS (explicitly configured)"
elif [[ -n "$SPEECH_GROUP" ]]; then
  info "speech:    $SPEECH_GROUP STT + TTS"
else
  info "speech:    stub backends (silence in, silence out)"
fi

if [[ "$OS" == "Darwin" && "$DETECTOR_KIND" == "yoloe" ]] && ! skipped vision; then
  warn "YOLOE on Apple Silicon is new and nobody has measured it. It runs on"
  warn "Metal where torch offers it and CPU otherwise, and either is slower than"
  warn "the CUDA path -- expect boxes to lag the video. The pipeline drops stale"
  warn "frames rather than queueing them, so it stays live and shows fewer boxes."
  warn "If Metal misbehaves, VMA_YOLOE_DEVICE=cpu is the way back."
fi

mkdir -p "$LOG_DIR"

# A real listener check, not a health probe: anything at all on the port makes
# uvicorn fail to bind, and probing /health would only notice a second copy of
# the same service and let every other collision through to a traceback.
# Python rather than `ss` or `lsof`, which differ between Linux and macOS.
port_taken() {
  python3 - "$1" "$PROBE_HOST" <<'EOF'
import socket, sys
with socket.socket() as probe:
    probe.settimeout(0.4)
    sys.exit(0 if probe.connect_ex((sys.argv[2], int(sys.argv[1]))) == 0 else 1)
EOF
}

require_port() { # port, service, hint
  if port_taken "$1"; then
    fail "port $1 is already in use, and $2 needs it -- $3"
  fi
}

bold "Checking ports"
skipped gateway || require_port 8080 "the media gateway" "stop whatever holds it"
skipped memory  || require_port 8081 "memory" "stop whatever holds it"
skipped vision  || require_port 8082 "vision" "stop whatever holds it"
skipped speech  || require_port 8085 "speech" "stop whatever holds it"
skipped agent   || require_port 8086 "the agent" "stop whatever holds it"
skipped console || require_port 5173 "the console" "stop whatever holds it"
info "clear"

# --- Credentials ------------------------------------------------------------

bold "LiveKit credentials"
# shellcheck source=lib/livekit_env.sh
source "$REPO_ROOT/scripts/lib/livekit_env.sh"
ensure_livekit_credentials "$ENV_FILE"

# --- Dependencies -----------------------------------------------------------

uv_sync() { # dir, extra uv args...
  local dir="$1"
  shift
  ( cd "$REPO_ROOT/$dir" && uv sync --frozen "$@" ) >>"$LOG_DIR/sync.log" 2>&1
}

importable() { # dir, module
  ( cd "$REPO_ROOT/$1" && uv run --no-sync python -c "import $2" ) >/dev/null 2>&1
}

cuda_operational() { # dir
  # Importing Torch and even cuda.is_available() are not sufficient. The
  # stable cu126 ARM64 wheel returns true on a GB10, then fails its first
  # kernel because it contains no compute-capability-12.1 code. Exercise one
  # real operation so the launcher never advertises a model profile that
  # cannot infer.
  ( cd "$REPO_ROOT/$1" && uv run --no-sync python -c \
      'import torch; assert torch.cuda.is_available(); print(torch.ones(1, device="cuda").sum().item())' \
  ) >/dev/null 2>&1
}

# Verify rather than trust. A .venv left from an earlier checkout can be
# missing the project's editable link while `uv sync --frozen` still reports
# "Audited N packages" -- it compares against the lock, not against what is
# importable. uvicorn then dies with a bare ModuleNotFoundError that reads like
# a broken repository rather than a stale environment.
sync_service() { # dir, import_name, extra uv args...
  local dir="$1" module="$2"
  shift 2
  info "$(basename "$dir")"
  uv_sync "$dir" "$@" || fail "uv sync failed in $dir -- see logs/sync.log"
  if ! importable "$dir" "$module"; then
    warn "$module is not importable; reinstalling"
    uv_sync "$dir" "$@" --reinstall \
      || fail "uv sync --reinstall failed in $dir -- see logs/sync.log"
    importable "$dir" "$module" \
      || fail "$module is still not importable -- remove $dir/.venv and re-run"
  fi
}

# The real model runtimes are optional by construction: each service probes
# what imported and falls back to a fixture or a stub, reporting which it chose
# at /v1/status. So a failure to install one degrades the stack instead of
# stopping it -- an mlx wheel that needs a newer macOS than the machine runs,
# or a CUDA download that fails, should not cost a teammate the other five
# services. It is said loudly here, and again by the service itself, precisely
# because a silent stub is indistinguishable from a broken model.
#
# Returns non-zero when it degraded, so the caller can adjust what it starts.
#
# `probe` is the model runtime, not the service: a successful install is not
# the same as a working one. The CUDA wheels resolved and installed cleanly
# and then `import torch` failed on a missing libcudnn -- see the note in
# services/vision-worker/pyproject.toml. Importing the runtime is the check
# that would have caught it.
sync_optional() { # dir, service_module, probe, what, extra uv args...
  local dir="$1" module="$2" probe="$3" what="$4" before line
  shift 4
  info "$(basename "$dir")"
  # Where this attempt's output starts, so the reason can be shown rather than
  # referred to. "see logs/sync.log" costs a round trip when the person who
  # hit it is on another machine, and this is the one failure most likely to
  # happen somewhere nobody can look.
  before="$(wc -l < "$LOG_DIR/sync.log" 2>/dev/null || echo 0)"
  if uv_sync "$dir" --inexact "$@" && importable "$dir" "$probe"; then
    return 0
  fi
  warn "could not install $what:"
  while IFS= read -r line; do
    [[ -n "$line" ]] && warn "    $line"
  done < <(sed -n "$((before + 1)),\$p" "$LOG_DIR/sync.log" 2>/dev/null | grep -v '^ *$' | tail -4)
  warn "carrying on without it; $(basename "$dir") will say so at /v1/status"
  warn "the full output is in logs/sync.log"
  uv_sync "$dir" --inexact || fail "uv sync failed in $dir -- see logs/sync.log"
  importable "$dir" "$module" \
    || fail "$module is not importable -- remove $dir/.venv and re-run"
  return 1
}

if [[ "$DO_SYNC" == "1" ]]; then
  bold "Syncing dependencies"
  info "(first run pulls a few GB of model runtime; later runs are seconds)"
  : > "$LOG_DIR/sync.log"

  # --all-groups for the gateway, not a bare sync: without it uv prunes the
  # optional groups and silently uninstalls `av`, breaking `virtual-glasses
  # --file` for anyone who had set it up. A dev script must not take tools away.
  skipped gateway || sync_service services/media-gateway media_gateway --all-groups

  skipped memory || sync_service services/application-memory application_memory
  skipped agent || sync_service services/agent agent --all-groups

  # --inexact throughout the heavy two: a plain `uv sync` prunes anything not
  # requested, so running this on a machine without a GPU would *uninstall* a
  # multi-gigabyte torch that a previous run had installed, and the next run
  # would download it again.
  if ! skipped vision; then
    if [[ "$VISION_MODELS" == "1" ]]; then
      if ! sync_optional services/vision-worker vision_worker ultralytics \
          "the models extra (YOLOE + MoGe)" --extra models \
          || { has_nvidia_gpu && ! cuda_operational services/vision-worker; }; then
        warn "Vision CUDA failed a real tensor operation; using the fixture detector"
        VISION_MODELS=0
        DETECTOR_KIND="fixture"
        DEPTH_KIND="none"
      fi
    else
      sync_service services/vision-worker vision_worker --inexact
    fi
  fi

  if ! skipped speech; then
    if [[ -n "$SPEECH_GROUP" ]]; then
      # The probe differs by runtime: torch is what the CUDA path fails to
      # import when its libcudnn is absent, mlx_audio what the Mac path fails
      # to install on a macOS older than mlx ships wheels for.
      if [[ "$SPEECH_GROUP" == "cuda" ]]; then SPEECH_PROBE=torch; else SPEECH_PROBE=mlx_audio; fi
      if ! sync_optional services/speech speech "$SPEECH_PROBE" \
          "the $SPEECH_GROUP group (Parakeet + Kokoro)" --group "$SPEECH_GROUP" \
          || { [[ "$SPEECH_GROUP" == "cuda" ]] && ! cuda_operational services/speech; }; then
        [[ "$SPEECH_GROUP" == "cuda" ]] \
          && warn "Speech CUDA failed a real tensor operation; using stub backends"
        SPEECH_GROUP=""
      fi
    else
      sync_service services/speech speech --inexact
    fi
  fi
else
  bold "Dependencies"
  info "skipped (--no-sync)"
fi

# --- LiveKit ----------------------------------------------------------------

livekit_up() {
  local url="${VMA_LIVEKIT_URL:-ws://127.0.0.1:7880}"
  url="${url/ws:\/\//http://}"
  url="${url/wss:\/\//https://}"
  curl -fs --max-time 2 "$url" >/dev/null 2>&1
}

livekit_credentials_ok() {
  python3 "$REPO_ROOT/tools/dev-livekit/check_credentials.py" >/dev/null 2>&1
}

compose_livekit() { # extra docker-compose args
  command -v docker >/dev/null 2>&1 \
    || fail "docker is not installed -- see tools/dev-livekit/ for a no-Docker path"
  # Order matters. `docker compose version` also fails when the daemon is
  # unreachable, so checking the plugin first would report a missing plugin
  # to someone who has simply not started Docker Desktop.
  docker info >/dev/null 2>&1 \
    || fail "the docker daemon is not running -- start Docker Desktop and re-run"
  docker compose version >/dev/null 2>&1 \
    || fail "the docker compose plugin is missing (v1 'docker-compose' will not do) -- see tools/dev-livekit/"
  docker compose -f "$REPO_ROOT/compose.dev.yaml" up -d "$@" livekit \
    >>"$LOG_DIR/livekit.log" 2>&1 \
    || fail "could not start LiveKit -- see logs/livekit.log"
  STARTED_LIVEKIT=1
  for _ in $(seq 1 40); do
    livekit_up && break
    sleep 0.5
  done
  livekit_up || fail "LiveKit did not become reachable on 127.0.0.1:7880 -- see logs/livekit.log"
}

if ! skipped livekit && ! skipped gateway; then
  if livekit_up; then
    bold "LiveKit is already running"
  else
    bold "Starting LiveKit"
    compose_livekit
  fi

  # Reachable is not the same as usable. A LiveKit container outlives the
  # shell that started it, so the pair it holds can differ from the pair in
  # .env that the gateway will sign join tokens with -- and that failure
  # surfaces nowhere near its cause: the token is well-formed, the server
  # returns 401, and the console says "could not join the livekit room".
  # Found exactly this way, against a container 27 hours old.
  if ! livekit_credentials_ok; then
    if [[ "$STARTED_LIVEKIT" == "1" ]]; then
      python3 "$REPO_ROOT/tools/dev-livekit/check_credentials.py" || true
      fail "LiveKit rejected the credentials it was just started with"
    fi
    warn "the running LiveKit holds different credentials than .env; recreating it"
    compose_livekit --force-recreate
    livekit_credentials_ok \
      || fail "LiveKit still rejects .env's credentials -- if it is running outside Docker, stop it and re-run"
    info "credentials now match"
  fi

  # Exposure is a privacy requirement, not a nicety: docs/07 asks for listeners
  # to be verified rather than inferred from the WebSocket URL.
  #
  # The check's own escape hatch is honoured here, because binding off loopback
  # is already a deliberate act -- the guard exists to catch *accidental*
  # exposure, and refusing to start after someone explicitly asked for a LAN
  # bind would make the documented Wi-Fi path impossible rather than safe.
  LISTENER_ARGS=()
  if [[ "$ALLOW_LAN" == "1" || "$BIND_ADDR" != "127.0.0.1" ]]; then
    LISTENER_ARGS+=(--allow-lan --expected-host "$BIND_ADDR")
    warn "LiveKit is published on $BIND_ADDR -- reachable by anything on this network"
    warn "the internal API token is the only thing protecting it; see docs/07"
  fi
  # `${a[@]+"${a[@]}"}` rather than `"${a[@]}"`: under `set -u`, bash 3.2 --
  # which macOS still ships, and which this script stays compatible with --
  # treats expanding an empty array as an unbound variable and exits.
  python3 "$REPO_ROOT/tools/dev-livekit/check_listeners.py" \
    ${LISTENER_ARGS[@]+"${LISTENER_ARGS[@]}"} \
    || fail "LiveKit is reachable beyond loopback -- see docs/07-Privacy-and-Security.md"
fi

# --- Starting things --------------------------------------------------------

track() { NAMES+=("$1"); PIDS+=("$2"); }

log_tail() { # name
  printf '\n\033[1m--- last 20 lines of logs/%s.log ---\033[0m\n' "$1" >&2
  tail -n 20 "$LOG_DIR/$1.log" >&2 || true
  printf '\n'
}

wait_ready() { # name, url, attempts, pid
  local name="$1" url="$2" attempts="$3" pid="$4" i=0
  while [[ $i -lt $attempts ]]; do
    # `-fs`, not `-fsS`: -S re-enables error output that -s suppressed, and a
    # readiness poll is *expected* to fail repeatedly. With it, a normal
    # startup buries the progress it is meant to show under a screenful of
    # "Failed to connect".
    if curl -fs --max-time 2 -o /dev/null "$url"; then
      [[ $i -gt 8 ]] && printf '\n'
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log_tail "$name"
      fail "$name exited during startup"
    fi
    # Silent for the first few seconds, then a progress dot, so a long model
    # load looks like work rather than a hang.
    if [[ $i -gt 8 && $((i % 10)) -eq 0 ]]; then printf '.'; fi
    sleep 0.5
    i=$((i + 1))
  done
  printf '\n'
  log_tail "$name"
  fail "$name did not become ready at $url"
}

# start <name> <dir> <command...> -- runs the command in that directory.
#
# Every service is launched through `uv run`, never by locating its
# virtualenv and executing the binary inside it. Resolving the environment
# by hand looks equivalent and is not: an activated or exported virtualenv
# makes `uv run` and a hand-built path disagree about which one is current,
# and the service then starts against an environment its own package was
# never installed into. That failed on a Mac with `ModuleNotFoundError: No
# module named 'media_gateway'` from an unrelated venv in /tmp -- while the
# importability check two steps earlier had passed, because that check went
# through `uv run` and therefore looked somewhere else.
#
# `uv run` is a parent process, so the server is this script's grandchild.
# `cleanup` signals descendants as well as the pids it tracks, which is what
# makes that safe -- see the note there.
start() {
  local name="$1" dir="$2"
  shift 2
  info "$name"
  ( cd "$REPO_ROOT/$dir" && exec "$@" ) >>"$LOG_DIR/$name.log" 2>&1 &
  track "$name" $!
}

bold "Starting services"

if ! skipped gateway; then
  : > "$LOG_DIR/gateway.log"
  # VMA_DIMENSION_GUARD_MODE is the one that matters. The Settings default is
  # `strict` at 320x180, which rejects every frame a webcam sends;
  # `first_frame_wins` is worse here, because LiveKit's encoder ramps from
  # 320x180 to 720p over ~25s and it latches the first rung. `sustained`
  # latches a size that has actually held, and re-latches when a new one does.
  VMA_ENVIRONMENT="${VMA_ENVIRONMENT:-dev}" \
  VMA_MEDIA_SOURCE="${VMA_MEDIA_SOURCE:-livekit}" \
  VMA_DIMENSION_GUARD_MODE="${VMA_DIMENSION_GUARD_MODE:-sustained}" \
  VMA_SAMPLE_FPS="${VMA_SAMPLE_FPS:-8}" \
  VMA_LIVEKIT_URL="${VMA_LIVEKIT_URL:-ws://127.0.0.1:7880}" \
    start gateway services/media-gateway \
      uv run --no-sync uvicorn media_gateway.main:app \
        --host "$BIND_ADDR" --port 8080 --log-level warning
  wait_ready gateway "http://$PROBE_HOST:8080/health/ready" 120 "${PIDS[${#PIDS[@]}-1]}"
fi

if ! skipped memory; then
  : > "$LOG_DIR/memory.log"
  # Run from the service directory: `database_url` and `evidence_dir` are
  # relative paths by default, so the working directory decides where the
  # database lands.
  VMA_ENVIRONMENT="${VMA_ENVIRONMENT:-dev}" \
    start memory services/application-memory \
      uv run --no-sync uvicorn application_memory.main:app \
        --host "$BIND_ADDR" --port 8081 --log-level warning
  wait_ready memory "http://$PROBE_HOST:8081/health/ready" 120 "${PIDS[${#PIDS[@]}-1]}"
fi

if ! skipped vision; then
  : > "$LOG_DIR/vision.log"
  EFFECTIVE_DETECTOR_KIND="${VMA_DETECTOR_KIND:-$DETECTOR_KIND}"
  EFFECTIVE_DEPTH_KIND="${VMA_DEPTH_KIND:-$DEPTH_KIND}"
  if [[ "$RESOURCE_SAFE_MODE" == "1" && "${VMA_ENABLE_CONSTRAINED_VISION:-false}" != "true" ]]; then
    EFFECTIVE_DETECTOR_KIND="fixture"
    EFFECTIVE_DEPTH_KIND="none"
  fi
  # VMA_SOURCE_FPS must match the gateway's VMA_SAMPLE_FPS above. Every
  # stability threshold is a duration converted to a frame count at this rate,
  # so a mismatch does not error -- it silently rescales all of them.
  #
  # The labels are open-vocabulary prompts, so their wording moves the numbers
  # more than the checkpoint size does: "a keychain with a colorful fob" finds
  # what "keys" misses.
  VMA_SOURCE_FPS="${VMA_SOURCE_FPS:-8}" \
  VMA_DETECTOR_KIND="$EFFECTIVE_DETECTOR_KIND" \
  VMA_DEPTH_KIND="$EFFECTIVE_DEPTH_KIND" \
  VMA_IMAGE_RESIDUAL_THRESHOLD="${VMA_IMAGE_RESIDUAL_THRESHOLD:-0.05}" \
  VMA_MAX_DETECTIONS_PER_FRAME="${VMA_MAX_DETECTIONS_PER_FRAME:-12}" \
  VMA_DETECTION_LABELS="${VMA_DETECTION_LABELS:-a set of keys,a keychain with a colorful fob,a coffee mug,a mobile phone,a laptop,a computer monitor,a pair of glasses}" \
  VMA_YOLOE_TEXT_MODEL="${VMA_YOLOE_TEXT_MODEL:-$([[ "${VMA_ENABLE_CONSTRAINED_VISION:-false}" == "true" ]] && echo yoloe-11s-seg.pt || echo yoloe-26l-seg.pt)}" \
  VMA_YOLOE_PROMPT_FREE_MODEL="${VMA_YOLOE_PROMPT_FREE_MODEL:-$([[ "${VMA_ENABLE_CONSTRAINED_VISION:-false}" == "true" ]] && echo yoloe-11s-seg-pf.pt || echo yoloe-26l-seg-pf.pt)}" \
  VMA_GATEWAY_VIDEO_URL="${VMA_GATEWAY_VIDEO_URL:-ws://$PROBE_HOST:8080/v1/stream/video}" \
  VMA_MEMORY_BASE_URL="${VMA_MEMORY_BASE_URL:-http://$PROBE_HOST:8081}" \
    start vision services/vision-worker \
      uv run --no-sync uvicorn vision_worker.main:app \
        --host "$BIND_ADDR" --port 8082 --log-level warning
  # Generous: the detector loads before the app reports ready, and on a first
  # run that includes downloading checkpoints.
  wait_ready vision "http://$PROBE_HOST:8082/health/ready" 600 "${PIDS[${#PIDS[@]}-1]}"
fi

if ! skipped speech; then
  : > "$LOG_DIR/speech.log"
  VMA_INTERNAL_API_TOKEN="${VMA_INTERNAL_API_TOKEN:-}" \
  VMA_GATEWAY_AUDIO_URL="${VMA_GATEWAY_AUDIO_URL:-ws://$PROBE_HOST:8080/v1/stream/audio}" \
  VMA_TTS_BACKEND="$SPEECH_TTS_BACKEND" \
    start speech services/speech \
      uv run --no-sync uvicorn speech.main:app \
        --host "$BIND_ADDR" --port 8085 --log-level warning
  wait_ready speech "http://$PROBE_HOST:8085/health/ready" 240 "${PIDS[${#PIDS[@]}-1]}"
fi

if ! skipped agent; then
  : > "$LOG_DIR/agent.log"
  # Hands-free discovery watches the gateway's active sessions, owns the one
  # Speech STT socket for each publisher, pushes transcript/reply HUD events to
  # the Gateway, and sends guarded TTS PCM back through return audio. The
  # console consumes those events instead of starting another Parakeet stream.
  VMA_AGENT_BACKEND="$AGENT_BACKEND" \
  VMA_MEMORY_BASE_URL="${VMA_MEMORY_BASE_URL:-http://$PROBE_HOST:8081}" \
  VMA_MEMORY_API_TOKEN="${VMA_MEMORY_API_TOKEN:-${VMA_INTERNAL_API_TOKEN:-}}" \
  VMA_GATEWAY_BASE_URL="${VMA_GATEWAY_BASE_URL:-http://$PROBE_HOST:8080}" \
  VMA_SPEECH_BASE_URL="${VMA_SPEECH_BASE_URL:-http://$PROBE_HOST:8085}" \
  VMA_HANDS_FREE_ENABLED="${VMA_HANDS_FREE_ENABLED:-true}" \
    start agent services/agent \
      uv run --no-sync uvicorn agent.main:app \
        --host "$BIND_ADDR" --port 8086 --log-level warning
  wait_ready agent "http://$PROBE_HOST:8086/health/ready" 120 "${PIDS[${#PIDS[@]}-1]}"
fi

if ! skipped console; then
  command -v npm >/dev/null 2>&1 || fail "npm is not on PATH -- install Node 20.19+ or pass --skip console"
  : > "$LOG_DIR/console.log"
  if [[ "$DO_SYNC" == "1" || ! -d "$REPO_ROOT/apps/console/node_modules" ]]; then
    info "npm install"
    ( cd "$REPO_ROOT/apps/console" && npm install --no-fund --no-audit ) >>"$LOG_DIR/console.log" 2>&1 \
      || fail "npm install failed -- see logs/console.log"
  fi
  # Vite directly rather than `npm run dev`: npm would be the tracked child and
  # vite its grandchild, so stopping npm can leave vite holding port 5173.
  # Its same-origin proxies must use the same exact trusted-LAN bind as the
  # native services; loopback cannot reach a process bound only to that address.
  VMA_GATEWAY_URL="${VMA_GATEWAY_URL:-http://$PROBE_HOST:8080}" \
  VMA_VISION_URL="${VMA_VISION_URL:-http://$PROBE_HOST:8082}" \
  VMA_MEMORY_URL="${VMA_MEMORY_URL:-http://$PROBE_HOST:8081}" \
  VMA_SPEECH_URL="${VMA_SPEECH_URL:-http://$PROBE_HOST:8085}" \
  VMA_AGENT_URL="${VMA_AGENT_URL:-http://$PROBE_HOST:8086}" \
  VITE_VMA_INTERNAL_API_TOKEN="${VITE_VMA_INTERNAL_API_TOKEN:-${VMA_INTERNAL_API_TOKEN:-}}" \
    start console apps/console ./node_modules/.bin/vite --host "$BIND_ADDR" --port 5173 --strictPort
  wait_ready console "http://$PROBE_HOST:5173/" 120 "${PIDS[${#PIDS[@]}-1]}"
fi

# --- Ready ------------------------------------------------------------------

printf '\n\033[32m READY \033[0m  %s\n\n' "$PLATFORM"
if ! skipped console; then
  printf '  Open              \033[1mhttp://%s:5173\033[0m  then Glasses -> Publish\n' "$PROBE_HOST"
  printf '                    (on WSL2, open this in your Windows browser)\n\n'
fi

if ! skipped vision; then
  if [[ "$DETECTOR_KIND" == "yoloe" ]]; then
    printf '  Boxes appear ~25s after Publish -- that is LiveKit'"'"'s encoder climbing to\n'
    printf '  720p, and the dimension guard waiting for a size that holds.\n\n'
  else
    printf '  The fixture detector finds nothing, so no boxes will appear. Everything\n'
    printf '  else -- publishing, the overlay stream, latency, memory -- is real.\n\n'
  fi
fi

# Asked rather than assumed. Installing the runtime and *using* it are two
# different things -- the service makes its own choice at startup and reports
# it, so this reads that answer instead of restating the intent from 200 lines
# up. A stub that says nothing is the failure mode the whole service guards
# against; a launcher that quietly implies otherwise would reintroduce it.
if ! skipped agent; then
  agent_backend="$(curl -fs --max-time 3 "http://$PROBE_HOST:8086/v1/status" \
    | python3 -c "
import json, sys
try:
    status = json.load(sys.stdin)
    print(status['backend'] + ' ' + status['model'] + ' @ ' + status['endpoint_host'])
except Exception:
    sys.exit(1)
" 2>/dev/null)" || agent_backend=""
  if [[ -n "$agent_backend" ]]; then
    printf '  Agent             %s\n\n' "$agent_backend"
  fi
fi

if ! skipped speech; then
  speech_backends="$(curl -fs --max-time 3 "http://$PROBE_HOST:8085/v1/status" \
    | python3 -c "
import json, sys
try:
    b = json.load(sys.stdin)['backends']
except Exception:
    sys.exit(1)
real = [k for k, v in b.items() if v.get('real')]
stub = [k for k, v in b.items() if not v.get('real')]
bits = []
if real: bits.append('real ' + '+'.join(sorted(real)))
if stub: bits.append('stub ' + '+'.join(sorted(stub)) + ' (silence)')
print(', '.join(bits))
" 2>/dev/null)" || speech_backends=""
  if [[ -n "$speech_backends" ]]; then
    printf '  Speech            %s\n\n' "$speech_backends"
  fi
fi

printf '  Logs              tail -f logs/*.log\n'
if ! skipped vision; then
  printf '  What is happening curl -s %s:8082/v1/status | python3 -m json.tool\n' "$PROBE_HOST"
fi
printf '\n  Ctrl-C stops everything.\n\n'

if [[ ${#PIDS[@]} -eq 0 ]]; then
  info "nothing left to wait for -- everything was skipped"
  exit 0
fi

# Polled rather than `wait`: any one of them exiting should bring the rest
# down, and `wait -n` -- which would do this without a loop -- needs bash 4.3,
# newer than the bash macOS ships.
while true; do
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      [[ "$SHUTTING_DOWN" == "1" ]] && exit 0
      log_tail "${NAMES[$i]}"
      fail "${NAMES[$i]} exited"
    fi
  done
  sleep 1
done
