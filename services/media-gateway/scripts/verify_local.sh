#!/usr/bin/env bash
#
# Verify the media gateway on a dev machine, with no glasses and no GN100.
#
# The assertions live in pytest, not here: tests/integration/ already ports the
# S01 spike's ten checks and runs them against the real service. This script is
# the operator-facing sequence around them -- bring up a server, confirm what it
# is listening on, run every check in order, and exit non-zero on the first
# failure -- so that "does my machine work?" is one command rather than a page
# of README. Duplicating the assertions in bash would only let the two drift.
#
#   ./scripts/verify_local.sh              structural + unit checks, then live
#   ./scripts/verify_local.sh --quick      skip the live LiveKit round trip
#   ./scripts/verify_local.sh --docker     also build the image and the arm64 gate
#
# Run it from services/media-gateway.

set -euo pipefail

QUICK=0
DOCKER=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --docker) DOCKER=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SERVICE_DIR/../.." && pwd)"
LIVEKIT_URL="${VMA_TEST_LIVEKIT_URL:-ws://127.0.0.1:7880}"
STARTED_LIVEKIT=0

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

cleanup() {
  if [[ "$STARTED_LIVEKIT" == "1" ]]; then
    bold "Stopping the LiveKit server this script started"
    docker compose -f "$REPO_ROOT/compose.dev.yaml" down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# --- Structural ------------------------------------------------------------

bold "Repository standards"
python3 "$REPO_ROOT/.agents/skills/visual-memory-repo-standards/scripts/validate_repo.py" \
  || fail "repository structure"

bold "Shared contract package"
cd "$REPO_ROOT/packages/media-contract"
uv sync --frozen --all-groups >/dev/null
uv run ruff format --check . || fail "media-contract formatting"
uv run ruff check .          || fail "media-contract lint"
uv run pyright               || fail "media-contract types"
uv run pytest                || fail "media-contract tests"

bold "Media gateway"
cd "$SERVICE_DIR"
uv sync --frozen --all-groups >/dev/null
uv run ruff format --check . || fail "gateway formatting"
uv run ruff check .          || fail "gateway lint"
uv run pyright               || fail "gateway types"
uv run pytest                || fail "gateway tests"

# --- Container -------------------------------------------------------------

if [[ "$DOCKER" == "1" ]]; then
  bold "Image build (context is the repository root)"
  docker build -f "$SERVICE_DIR/Dockerfile" -t vma/media-gateway:dev "$REPO_ROOT" \
    || fail "docker build"

  bold "ARM64 packaging gate (emulated; layering only, never CUDA or GN100)"
  docker buildx build --platform linux/arm64 -f "$SERVICE_DIR/Dockerfile" "$REPO_ROOT" \
    || fail "arm64 build"
fi

# --- Live round trip -------------------------------------------------------

if [[ "$QUICK" == "1" ]]; then
  bold "Skipping the live round trip (--quick)"
  printf '\n\033[32mOK\033[0m  structural and unit checks passed.\n'
  exit 0
fi

livekit_up() { curl -fsS http://127.0.0.1:7880 >/dev/null 2>&1; }

bold "LiveKit server"
if livekit_up; then
  echo "already running at $LIVEKIT_URL"
  if [[ -z "${VMA_LIVEKIT_API_KEY:-}" || -z "${VMA_LIVEKIT_API_SECRET:-}" ]]; then
    fail "a LiveKit server is already running, but VMA_LIVEKIT_API_KEY and
      VMA_LIVEKIT_API_SECRET are not set. They must match the pair that server
      was started with, or every token it issues will be rejected."
  fi
else
  if [[ -z "${VMA_LIVEKIT_API_KEY:-}" || -z "${VMA_LIVEKIT_API_SECRET:-}" ]]; then
    # Ephemeral and never written to disk. The gateway's validator refuses the
    # well-known dev values, so a generated pair is the only thing that works.
    export VMA_LIVEKIT_API_KEY="vma-verify-$(openssl rand -hex 3)"
    export VMA_LIVEKIT_API_SECRET="$(openssl rand -hex 24)"
    echo "generated an ephemeral credential pair for this run"
  fi
  echo "starting one from compose.dev.yaml"
  docker compose -f "$REPO_ROOT/compose.dev.yaml" up -d livekit >/dev/null \
    || fail "could not start LiveKit; see tools/dev-livekit/README.md for the
      no-Docker path"
  STARTED_LIVEKIT=1
  for _ in $(seq 1 30); do
    livekit_up && break
    sleep 1
  done
  livekit_up || fail "LiveKit did not become reachable"
fi

bold "Listener exposure"
# docs/07: verify listeners, never infer exposure from the WebSocket URL.
python3 "$REPO_ROOT/tools/dev-livekit/check_listeners.py" \
  || fail "LiveKit is listening beyond loopback; restrict it to the trusted LAN"

bold "Round trip through a real server (the spike's ten assertions)"
cd "$SERVICE_DIR"
VMA_TEST_LIVEKIT_URL="$LIVEKIT_URL" uv run pytest tests/integration -m livekit \
  || fail "integration round trip"

printf '\n\033[32mOK\033[0m  everything passed, including the privacy sweep for\n'
printf '    off-machine connections.\n\n'
printf 'To drive it by hand with your own camera and microphone, run\n'
printf './scripts/dev_stack.sh from the repository root and open\n'
printf 'http://localhost:5173 -- see services/media-gateway/README.md.\n'
