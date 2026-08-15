#!/usr/bin/env bash
# Run a second laptop-hosted operator console against GN100 without rewriting
# the laptop development profile in apps/console/.env.local.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  VMA_TARGET_INTERNAL_API_TOKEN=... start_target_console.sh <gn100-ip> [port]

Starts a Vite console on http://127.0.0.1:<port> (default 5174), proxies every
API/WebSocket to the GN100, and tells the browser to use GN100 LiveKit directly.
The normal laptop console remains on 5173. The target must publish ports
8080/8081/8082/8085/8086 and LiveKit 7880/7881/7882 UDP to the trusted LAN.
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac
TARGET_IP=$1
PORT=${2:-5174}
[[ "$TARGET_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo "invalid GN100 IPv4" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "invalid console port" >&2; exit 2; }
TOKEN="${VMA_TARGET_INTERNAL_API_TOKEN:-}"
if [[ -n "${VMA_TARGET_INTERNAL_API_TOKEN_FILE:-}" ]]; then
    [[ -f "$VMA_TARGET_INTERNAL_API_TOKEN_FILE" ]] || { echo "token file missing" >&2; exit 1; }
    TOKEN_MODE="$(stat -c '%a' "$VMA_TARGET_INTERNAL_API_TOKEN_FILE" 2>/dev/null || true)"
    [[ "$TOKEN_MODE" == "600" || "$TOKEN_MODE" == "400" ]] \
        || { echo "token file must be mode 600 or 400 (found ${TOKEN_MODE:-unknown})" >&2; exit 1; }
    TOKEN="$(<"$VMA_TARGET_INTERNAL_API_TOKEN_FILE")"
fi
[[ -n "$TOKEN" ]] || { echo "set VMA_TARGET_INTERNAL_API_TOKEN or _FILE" >&2; exit 1; }

for endpoint in 8080/health/live 8081/health/live 8082/health/live 8085/health/live 8086/health/live; do
    port=${endpoint%%/*}; path=${endpoint#*/}
    curl -fsS --max-time 4 "http://$TARGET_IP:$port/$path" >/dev/null \
        || { echo "target service is unavailable at $TARGET_IP:$port/$path" >&2; exit 1; }
done
curl -fsS --max-time 4 "http://$TARGET_IP:7880" >/dev/null \
    || { echo "target LiveKit signaling is unavailable" >&2; exit 1; }

export VITE_VMA_INTERNAL_API_TOKEN="$TOKEN"
export VITE_VMA_GATEWAY_PUBLIC_URL="http://$TARGET_IP:8080"
export VITE_VMA_LIVEKIT_URL="ws://$TARGET_IP:7880"
export VMA_GATEWAY_URL="http://$TARGET_IP:8080"
export VMA_MEMORY_URL="http://$TARGET_IP:8081"
export VMA_VISION_URL="http://$TARGET_IP:8082"
export VMA_SPEECH_URL="http://$TARGET_IP:8085"
export VMA_AGENT_URL="http://$TARGET_IP:8086"

printf 'GN100 operator console: http://127.0.0.1:%s\n' "$PORT"
printf 'Gateway/LiveKit target: %s\n' "$TARGET_IP"
printf 'Keep this terminal open; Ctrl-C stops only this console.\n\n'
cd apps/console
npm run dev -- --host 127.0.0.1 --port "$PORT" --strictPort
