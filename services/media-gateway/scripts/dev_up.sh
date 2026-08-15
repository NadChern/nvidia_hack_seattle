#!/usr/bin/env bash
#
# Bring up everything needed to publish a real camera into the pipeline.
#
# Replaces the three-step sequence -- generate credentials, start LiveKit,
# start the gateway -- with one command. The three steps are not hard, but the
# credentials have to match between the server and the gateway, and a mismatch
# fails in a way that looks like a network problem rather than a config one.
# This generates them once, persists them, and reuses them forever after.
#
#   ./scripts/dev_up.sh                 LiveKit + gateway, ready for a camera
#   ./scripts/dev_up.sh --scripted      no LiveKit at all; synthetic video only
#   ./scripts/dev_up.sh --strict-guard  reject anything that is not 320x180
#   ./scripts/dev_up.sh --port 9000     serve the gateway somewhere else
#   ./scripts/dev_up.sh --keep-livekit  leave the server running on exit
#
# Ctrl-C stops whatever this script started. Run it from anywhere.

set -euo pipefail

SCRIPTED=0
KEEP_LIVEKIT=0
GUARD="first_frame_wins"
PORT=8080

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scripted) SCRIPTED=1 ;;
    --keep-livekit) KEEP_LIVEKIT=1 ;;
    --strict-guard) GUARD="strict" ;;
    --port) PORT="${2:?--port needs a value}"; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SERVICE_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
STARTED_LIVEKIT=0

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
info() { printf '   %s\n' "$1"; }
fail() { printf '\033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

cleanup() {
  if [[ "$STARTED_LIVEKIT" == "1" && "$KEEP_LIVEKIT" == "0" ]]; then
    printf '\n'
    bold "Stopping the LiveKit server this script started"
    docker compose -f "$REPO_ROOT/compose.dev.yaml" down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

command -v uv >/dev/null || fail "uv is not installed -- see https://docs.astral.sh/uv/"
# uv can supply its own interpreter, so a machine can have uv and no system
# python3. This script needs one for the port probe and the listener check.
command -v python3 >/dev/null || fail "python3 is not on PATH"

# --- Credentials -----------------------------------------------------------
#
# Shared with scripts/dev_stack.sh rather than duplicated: two copies of the
# generate-and-persist logic could drift, and a key/secret pair that disagrees
# between the server and the gateway is exactly the failure this exists to
# prevent.

bold "LiveKit credentials"
# shellcheck source=scripts/lib/livekit_env.sh
source "$REPO_ROOT/scripts/lib/livekit_env.sh"
ensure_livekit_credentials "$ENV_FILE"

# --- Dependencies ----------------------------------------------------------

bold "Syncing dependencies"
cd "$SERVICE_DIR"
# --all-groups, not a bare sync: without it uv prunes the optional groups and
# silently uninstalls `av`, breaking `virtual-glasses --file` for anyone who
# had set it up. A dev script must not take tools away.
uv sync --frozen --all-groups >/dev/null || fail "uv sync"

# Verify rather than trust. A .venv left over from an earlier checkout can be
# missing the project's editable link while `uv sync --frozen` still reports
# "Audited N packages" -- it compares against the lock, not against what is
# actually importable. uvicorn then dies with a bare ModuleNotFoundError that
# reads like a broken repository rather than a stale environment.
importable() { uv run --no-sync python -c "import media_gateway" >/dev/null 2>&1; }

if ! importable; then
  info "environment is stale; reinstalling"
  uv sync --frozen --all-groups --reinstall >/dev/null || fail "uv sync --reinstall"
  importable || fail "media_gateway is still not importable -- remove .venv and re-run"
fi
info "ready"

# --- LiveKit ---------------------------------------------------------------

livekit_up() { curl -fsS http://127.0.0.1:7880 >/dev/null 2>&1; }

if [[ "$SCRIPTED" == "1" ]]; then
  bold "Scripted mode -- no LiveKit, synthetic video only"
  MEDIA_SOURCE="scripted"
else
  MEDIA_SOURCE="livekit"
  if livekit_up; then
    bold "LiveKit is already running"
  else
    command -v docker >/dev/null \
      || fail "docker is not installed -- use --scripted, or see tools/dev-livekit/"
    # Order matters. `docker compose version` also fails when the daemon is
    # unreachable, so checking the plugin first would report a missing plugin
    # to someone who simply has not started Docker Desktop.
    #
    # `docker` on PATH does not mean the daemon is up, and on macOS it usually
    # is not until Docker Desktop has been launched.
    docker info >/dev/null 2>&1 \
      || fail "the docker daemon is not running -- start Docker Desktop, or use --scripted"
    docker compose version >/dev/null 2>&1 \
      || fail "the docker compose plugin is missing (v1 'docker-compose' will not do) -- see tools/dev-livekit/ for a no-Docker path"
    bold "Starting LiveKit"
    docker compose -f "$REPO_ROOT/compose.dev.yaml" up -d livekit >/dev/null \
      || fail "could not start LiveKit -- check 'docker compose -f compose.dev.yaml logs livekit'"
    STARTED_LIVEKIT=1
    for _ in $(seq 1 40); do
      livekit_up && break
      sleep 0.5
    done
    livekit_up || fail "LiveKit did not become reachable on 127.0.0.1:7880"
  fi

  # Exposure is a privacy requirement, not a nicety: docs/07 asks for listeners
  # to be verified rather than inferred from the WebSocket URL.
  bold "Checking what LiveKit is listening on"
  python3 "$REPO_ROOT/tools/dev-livekit/check_listeners.py" \
    || fail "LiveKit is reachable beyond loopback -- see docs/07-Privacy-and-Security.md"
fi

# --- Gateway ---------------------------------------------------------------

# A real listener check, not a health probe: anything at all on this port makes
# uvicorn fail to bind, and port 8080 is popular. Probing /health/live would
# only notice another gateway and let every other collision through to a
# confusing traceback. Python rather than `ss` or `lsof`, which differ across
# Linux and macOS.
port_taken() {
  python3 - "$PORT" <<'EOF'
import socket, sys
with socket.socket() as probe:
    probe.settimeout(0.4)
    sys.exit(0 if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
EOF
}

if port_taken; then
  fail "something is already listening on port $PORT -- stop it, or pass --port"
fi

bold "Starting the gateway"
info "source: $MEDIA_SOURCE   guard: $GUARD   port: $PORT"

VMA_ENVIRONMENT=dev \
VMA_MEDIA_SOURCE="$MEDIA_SOURCE" \
VMA_DIMENSION_GUARD_MODE="$GUARD" \
  uv run uvicorn media_gateway.main:app --port "$PORT" --log-level warning &
GATEWAY_PID=$!

for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$PORT/health/ready" >/dev/null 2>&1 && break
  kill -0 "$GATEWAY_PID" 2>/dev/null || fail "the gateway exited during startup"
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/health/ready" >/dev/null 2>&1 \
  || fail "the gateway did not become ready"

printf '\n\033[32m READY \033[0m\n\n'
if [[ "$SCRIPTED" == "0" ]]; then
  # This script starts the gateway alone, and publishing a camera needs a
  # browser page that no longer ships with it.
  printf '  Publish a camera   cd %s/apps/console && npm run dev\n' "$REPO_ROOT"
  printf '                     then \033[1mhttp://localhost:5173\033[0m\n'
  printf '                     (on WSL2, open this in your Windows browser)\n'
  printf '                     Or ./scripts/dev_stack.sh for the whole stack.\n\n'
fi
printf '  Watch the frames   cd %s\n' "$SERVICE_DIR"
printf '                     uv run python -m visual_memory_media_contract.tap \\\n'
printf '                       ws://127.0.0.1:%s/v1/stream/video\n\n' "$PORT"
printf '  Gateway state      curl -fsS localhost:%s/v1/status | python3 -m json.tool\n\n' "$PORT"
printf '  Ctrl-C to stop.\n\n'

wait "$GATEWAY_PID"
