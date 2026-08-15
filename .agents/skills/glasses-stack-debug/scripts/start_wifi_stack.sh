#!/usr/bin/env bash
# Start the proven WSL2 mirrored-networking stack in one foreground terminal.
# Ctrl-C stops dev_stack and any native LiveKit process started by this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

CHECK_ONLY=0
DEV_STACK_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        -h|--help)
            cat <<'EOF'
Usage: start_wifi_stack.sh [--check] [dev_stack arguments]

Starts native LiveKit first, then dev_stack with LAN binding and constrained
YOLOE + metric-depth Vision. Keep it in a foreground terminal; Ctrl-C performs
a supervised shutdown.

  --check    verify persistent Wi-Fi/WSL/LiveKit/firewall configuration only
  --no-sync  may be passed through for a faster restart after dependencies sync
EOF
            exit 0
            ;;
        *) DEV_STACK_ARGS+=("$1") ;;
    esac
    shift
done

fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }
pass() { printf '\033[32mPASS\033[0m %s\n' "$*"; }

[[ -f .env ]] || fail ".env is missing"
[[ -x .tools/livekit-1.13.4/livekit-server ]] || fail "native LiveKit binary is missing"

WIFI_IP="$(grep -oE 'ws://[0-9.]+:7880' .env | grep -oE '[0-9]+(\.[0-9]+){3}' | head -1 || true)"
[[ -n "$WIFI_IP" && "$WIFI_IP" != "127.0.0.1" ]] \
    || fail "VMA_LIVEKIT_PUBLIC_URL in .env must use the laptop Wi-Fi address"
ACTUAL_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
if ! ip -4 addr | grep -q "inet $WIFI_IP/"; then
    if [[ -n "$ACTUAL_IP" && "$ACTUAL_IP" != "$WIFI_IP" ]]; then
        fail "Wi-Fi IP changed from $WIFI_IP to $ACTUAL_IP after reboot; update .env public LiveKit URLs and apps/console/.env.local gateway URL, then re-pair the glasses"
    fi
    fail "WSL does not own $WIFI_IP; enable networkingMode=mirrored in %USERPROFILE%\\.wslconfig, run wsl --shutdown, and reopen WSL"
fi
pass "WSL mirrored networking owns $WIFI_IP"

LIVEKIT_CONFIG=tools/dev-livekit/livekit.dev.yaml
! grep -Eq '^[[:space:]]*node_ip:' "$LIVEKIT_CONFIG" \
    || fail "leave LiveKit node_ip unset: glasses need Wi-Fi while Windows Chrome needs loopback"
grep -Eq '^[[:space:]]*enable_loopback_candidate:[[:space:]]*true' "$LIVEKIT_CONFIG" \
    || fail "set enable_loopback_candidate: true in $LIVEKIT_CONFIG"
grep -Eq '^[[:space:]]*force_tcp:[[:space:]]*false' "$LIVEKIT_CONFIG" \
    || fail "set force_tcp: false in $LIVEKIT_CONFIG; Chrome needs UDP"
pass "LiveKit advertises both Wi-Fi and loopback UDP candidates"

CONSOLE_ENV=apps/console/.env.local
[[ -f "$CONSOLE_ENV" ]] || fail "$CONSOLE_ENV is missing"
grep -q "^VITE_VMA_GATEWAY_PUBLIC_URL=http://$WIFI_IP:8080" "$CONSOLE_ENV" \
    || fail "set VITE_VMA_GATEWAY_PUBLIC_URL=http://$WIFI_IP:8080 in $CONSOLE_ENV"
grep -q '^VITE_VMA_LIVEKIT_URL=ws://127\.0\.0\.1:7880' "$CONSOLE_ENV" \
    || fail "set VITE_VMA_LIVEKIT_URL=ws://127.0.0.1:7880 in $CONSOLE_ENV"
pass "console uses Wi-Fi for gateway control and loopback for LiveKit media"

if command -v netsh.exe >/dev/null 2>&1; then
    stale="$(netsh.exe interface portproxy show all 2>/dev/null | tr -d '\r' \
        | grep -cE '^(10|127|192)\.[0-9.]+ +(8080|8081|8082|8085|8086|5173|7880|7881) ' || true)"
    [[ "${stale:-0}" -eq 0 ]] \
        || fail "remove stale Windows portproxy rules; mirrored networking needs no forwarding"
    pass "no harmful Windows portproxy rules"
fi

if command -v powershell.exe >/dev/null 2>&1; then
    FIREWALL="$(powershell.exe -NoProfile -Command '
        $name="VMA glasses (WSL2 mirrored)"
        Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue |
          Where-Object {$_.Enabled -eq "True" -and $_.Direction -eq "Inbound" -and $_.Action -eq "Allow"} |
          ForEach-Object { $_ | Get-NetFirewallPortFilter | ForEach-Object {
            "{0}:{1}" -f $_.Protocol, (($_.LocalPort) -join ",")
          }}' 2>/dev/null | tr -d '\r')"
    printf '%s\n' "$FIREWALL" | grep -q 'TCP:.*8080.*7880.*7881' \
        || fail "Windows TCP firewall ports are missing; run scripts\\wsl_lan_expose.ps1 in Administrator PowerShell"
    printf '%s\n' "$FIREWALL" | grep -q 'UDP:.*7882' \
        || fail "Windows UDP 7882 is missing; run scripts\\wsl_lan_expose.ps1 in Administrator PowerShell"
    pass "Windows Firewall allows TCP 8080/7880/7881 and UDP 7882"
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
    printf '\nKnown-good Wi-Fi configuration is present.\n'
    exit 0
fi

for port in 8080 8081 8082 8085 8086 5173; do
    if ss -ltn 2>/dev/null | grep -qE "[:.]$port "; then
        fail "port $port is already in use; run stack_doctor.sh and stop the existing dev_stack supervisor cleanly"
    fi
done

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q livekit; then
    fail "LiveKit is running in Docker; stop Compose LiveKit before continuing"
fi

LIVEKIT_STARTED=0
LIVEKIT_PID=""
cleanup() {
    if [[ "$LIVEKIT_STARTED" == "1" && -n "$LIVEKIT_PID" ]] && kill -0 "$LIVEKIT_PID" 2>/dev/null; then
        printf '\nStopping native LiveKit (pid %s)\n' "$LIVEKIT_PID"
        kill -TERM "$LIVEKIT_PID" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$LIVEKIT_PID" 2>/dev/null || break
            sleep 1
        done
        # A second TERM tells LiveKit to stop waiting for stale participants.
        kill -0 "$LIVEKIT_PID" 2>/dev/null && kill -TERM "$LIVEKIT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if ss -ltn 2>/dev/null | grep -qE '[:.]7880 '; then
    pgrep -f 'livekit-server --config' >/dev/null \
        || fail "7880 is occupied by something other than native LiveKit"
    pass "reusing existing native LiveKit"
else
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
    export LIVEKIT_KEYS="$VMA_LIVEKIT_API_KEY: $VMA_LIVEKIT_API_SECRET"
    mkdir -p logs
    .tools/livekit-1.13.4/livekit-server --config "$LIVEKIT_CONFIG" >>logs/livekit.log 2>&1 &
    LIVEKIT_PID=$!
    LIVEKIT_STARTED=1
    for _ in $(seq 1 30); do
        curl -fsS --max-time 1 http://127.0.0.1:7880 >/dev/null 2>&1 && break
        sleep 1
    done
    curl -fsS --max-time 1 http://127.0.0.1:7880 >/dev/null \
        || fail "native LiveKit did not become ready; inspect logs/livekit.log"
    pass "native LiveKit started (pid $LIVEKIT_PID)"
fi

export VMA_BIND_ADDR=0.0.0.0
export VMA_ENABLE_CONSTRAINED_VISION="${VMA_ENABLE_CONSTRAINED_VISION:-true}"

printf '\nStarting supervised stack with constrained detection + metric depth.\n'
printf 'Keep this terminal open. Use Ctrl-C to stop everything cleanly.\n\n'
./scripts/dev_stack.sh "${DEV_STACK_ARGS[@]}"
