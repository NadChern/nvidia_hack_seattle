#!/usr/bin/env bash
#
# One command that says what is actually wrong with the glasses <-> backend
# path, instead of a person guessing which of nine things broke this time.
#
#   .agents/skills/glasses-stack-debug/scripts/stack_doctor.sh
#
# Read-only. It never starts, stops, or kills anything -- diagnosis and repair
# are deliberately separate, because half the outages in this stack were caused
# by someone (including an agent) killing processes to "clean up" and taking
# the healthy siblings with them.
#
# Exit code is the number of failed checks, so it composes:
#   stack_doctor.sh && echo "ready to pair"

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 99

PASS=0
FAIL=0
NOTE=()

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); NOTE+=("$2"); }
skip() { printf '  \033[2m--  \033[0m  %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

listening() { ss -ltn 2>/dev/null | grep -qE "[:.]$1 "; }
bind_of()   { ss -ltn 2>/dev/null | grep -E "[:.]$1 " | awk '{print $4}' | head -1; }

http() { # url -> prints code; 000 means no HTTP response
    # No `|| echo 000` fallback: curl already prints 000 on a failed connection
    # *and* exits non-zero, so the fallback concatenates and yields "000000".
    curl -s -m "${2:-4}" -o /dev/null -w "%{http_code}" "$1" 2>/dev/null
}

# --- Which topology is this supposed to be? ---------------------------------
#
# Two supported modes, and almost every confusing failure is a half-configured
# mix of them. Detected rather than asked, so the report matches reality.

WIFI_IP="$(grep -oE 'ws://[0-9.]+:7880' .env 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
NODE_IP="$(grep -E '^\s*node_ip:' tools/dev-livekit/livekit.dev.yaml 2>/dev/null | awk '{print $2}')"
FORCE_TCP="$(grep -E '^\s*force_tcp:' tools/dev-livekit/livekit.dev.yaml 2>/dev/null | awk '{print $2}')"
LOOPBACK_CANDIDATE="$(grep -E '^\s*enable_loopback_candidate:' tools/dev-livekit/livekit.dev.yaml 2>/dev/null | awk '{print $2}')"
HAVE_ADB=0
command -v adb >/dev/null 2>&1 && adb devices 2>/dev/null | grep -q "device$" && HAVE_ADB=1

if [[ -n "$WIFI_IP" && "$WIFI_IP" != "127.0.0.1" ]]; then
    MODE="wifi"; DEVICE_HOST="$WIFI_IP"
else
    MODE="usb"; DEVICE_HOST="127.0.0.1"
fi

head_ "Topology"
printf '  mode: %s   device reaches host at: %s   livekit node_ip: %s\n' \
    "$MODE" "$DEVICE_HOST" "${NODE_IP:-<unset>}"

# --- Services ---------------------------------------------------------------

head_ "Services"
for entry in "8080:gateway" "8081:memory" "8082:vision" "8085:speech" "8086:agent" "5173:console"; do
    port="${entry%%:*}"; name="${entry##*:}"
    if listening "$port"; then
        ok "$name listening on $(bind_of "$port")"
    else
        bad "$name is not listening on $port" \
            "start the stack: VMA_BIND_ADDR=${MODE:+0.0.0.0} ./scripts/dev_stack.sh"
    fi
done

if listening 7880; then
    ok "livekit signalling on $(bind_of 7880)"
else
    bad "livekit is not listening on 7880" \
        "start native LiveKit before dev_stack: set -a && . ./.env && set +a; export LIVEKIT_KEYS=\"\$VMA_LIVEKIT_API_KEY: \$VMA_LIVEKIT_API_SECRET\"; .tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml"
fi
# Native, not Docker. Measured: with LiveKit in a container, ICE never
# completes even for a client on the same host -- the gateway's own join fails
# with `wait_pc_connection timed out`. Docker's bridge NAT is the last address
# translation in the path and WebRTC does not survive it. Running the pinned
# binary directly fixed it on the first attempt.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q livekit; then
    bad "livekit is running in Docker; ICE will not complete" \
        "stop it and run the binary instead: docker compose -f compose.dev.yaml down; set -a && . ./.env && set +a; export LIVEKIT_KEYS=\"\$VMA_LIVEKIT_API_KEY: \$VMA_LIVEKIT_API_SECRET\"; .tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml &"
elif pgrep -f "livekit-server --config" >/dev/null 2>&1; then
    ok "livekit running natively (no Docker NAT in the media path)"
else
    bad "livekit is not running" \
        "set -a && . ./.env && set +a; export LIVEKIT_KEYS=\"\$VMA_LIVEKIT_API_KEY: \$VMA_LIVEKIT_API_SECRET\"; .tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml &"
fi

# --- The path the glasses actually take -------------------------------------

head_ "Path from the glasses"
if [[ "$MODE" == "wifi" ]]; then
    code="$(http "http://$DEVICE_HOST:8080/health/live")"
    if [[ "$code" == "200" ]]; then
        ok "gateway reachable at $DEVICE_HOST:8080"
    else
        bad "gateway NOT reachable at $DEVICE_HOST:8080 (got $code)" \
            "needs VMA_BIND_ADDR=0.0.0.0, WSL mirrored networking, and the firewall rule. Re-run: powershell -File scripts/wsl_lan_expose.ps1 (as Administrator); do not add a portproxy"
    fi
    code="$(http "http://$DEVICE_HOST:7880")"
    if [[ "$code" == "200" ]]; then
        ok "livekit reachable at $DEVICE_HOST:7880"
    else
        bad "livekit NOT reachable at $DEVICE_HOST:7880 (got $code)" \
            "start native LiveKit, keep WSL mirrored, bind to the LAN, and apply scripts/wsl_lan_expose.ps1 as Administrator"
    fi
    if [[ "$NODE_IP" == "$DEVICE_HOST" ]]; then
        ok "livekit advertises the address the glasses can reach (device-only pin)"
    elif [[ -z "$NODE_IP" && "$FORCE_TCP" == "false" && "$LOOPBACK_CANDIDATE" == "true" ]]; then
        ok "livekit auto-advertises Wi-Fi for glasses and loopback for the Windows viewer"
    else
        bad "livekit candidate settings cannot serve both glasses and viewer (node_ip=${NODE_IP:-<unset>}, force_tcp=${FORCE_TCP:-<unset>})" \
            "for mirrored Wi-Fi mode leave node_ip unset, set enable_loopback_candidate: true and force_tcp: false, then restart native LiveKit"
    fi
else
    if [[ "$HAVE_ADB" == "1" ]]; then
        tunnels="$(adb reverse --list 2>/dev/null | wc -l)"
        if [[ "$tunnels" -ge 3 ]]; then
            ok "adb reverse tunnels present ($tunnels)"
        else
            bad "adb reverse tunnels missing (found $tunnels, need 8080/7880/7881)" \
                "adb reverse tcp:8080 tcp:8080 && adb reverse tcp:7880 tcp:7880 && adb reverse tcp:7881 tcp:7881"
        fi
    else
        skip "no adb device; cannot check tunnels"
    fi
    if [[ "$NODE_IP" == "127.0.0.1" ]]; then
        ok "livekit advertises loopback, correct for the USB tunnel"
    else
        bad "livekit advertises '${NODE_IP:-<unset>}', unreachable through adb reverse" \
            "set node_ip: 127.0.0.1 for USB mode"
    fi
fi

# --- Leftover Windows forwards ----------------------------------------------
#
# With mirrored networking WSL and Windows share one port space, so a stale
# `netsh portproxy` rule squats on the port its Linux service wants. It shows
# up as "port NNNN is already in use" with nothing visible in `ss`, because the
# listener belongs to Windows.

head_ "Windows port forwards"
if command -v netsh.exe >/dev/null 2>&1; then
    stale=$(netsh.exe interface portproxy show all 2>/dev/null | tr -d '\r' \
        | grep -cE '^(10|127|192)\.[0-9.]+ +(8080|8081|8082|8085|8086|5173|7880|7881) ')
    stale="${stale:-0}"
    if [[ "$stale" -eq 0 ]]; then
        ok "no portproxy rules on the stack's ports"
    else
        bad "$stale portproxy rule(s) squat on ports this stack needs" \
            "with mirrored networking nothing needs forwarding; in an Administrator PowerShell: netsh interface portproxy show all, then delete each with: netsh interface portproxy delete v4tov4 listenaddress=<addr> listenport=<port>"
    fi
else
    skip "not on WSL; no Windows forwards to check"
fi

# Mirrored WSL traffic still crosses Windows Firewall. Signalling can work
# while Chrome media fails if UDP 7882 is missing, so test protocols and ports
# rather than merely checking that a similarly named rule exists.
head_ "Windows firewall"
if command -v powershell.exe >/dev/null 2>&1; then
    firewall=$(powershell.exe -NoProfile -Command '
        $name="VMA glasses (WSL2 mirrored)"
        Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue |
          Where-Object {$_.Enabled -eq "True" -and $_.Direction -eq "Inbound" -and $_.Action -eq "Allow"} |
          ForEach-Object { $_ | Get-NetFirewallPortFilter | ForEach-Object {
            "{0}:{1}" -f $_.Protocol, (($_.LocalPort) -join ",")
          }}' 2>/dev/null | tr -d '\r')
    if printf '%s\n' "$firewall" | grep -q 'TCP:.*8080.*7880.*7881' \
        && printf '%s\n' "$firewall" | grep -q 'UDP:.*7882'; then
        ok "inbound TCP 8080/7880/7881 and UDP 7882 are allowed"
    else
        bad "required mirrored-WSL firewall ports are missing" \
            "in Administrator PowerShell: powershell -ExecutionPolicy Bypass -File scripts\\wsl_lan_expose.ps1"
    fi
else
    skip "not on WSL; cannot inspect Windows Firewall"
fi

# --- Capacity ---------------------------------------------------------------

head_ "Session slots"
TOKEN="$(grep -oE '^VMA_INTERNAL_API_TOKEN=.*' .env 2>/dev/null | cut -d= -f2-)"
if listening 8080 && [[ -n "$TOKEN" ]]; then
    sessions="$(curl -s -m4 -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/v1/sessions 2>/dev/null)"
    total=$(printf '%s' "$sessions" | grep -o '"session_id"' | wc -l)
    live=$(printf '%s' "$sessions" | grep -o '"publisher_present":true' | wc -l)
    if [[ "$total" -lt 2 ]]; then
        ok "$total/2 slots used ($live with a publisher)"
    else
        bad "$total/2 slots used, $live with a publisher -- new joins get 429" \
            "console -> Glasses -> Clear stale, or wait unclaimed_session_ttl_s (90s)"
    fi
else
    skip "gateway down or no token; cannot read sessions"
fi

# --- The device -------------------------------------------------------------

head_ "Glasses"
if [[ "$HAVE_ADB" == "1" ]]; then
    ok "adb sees $(adb devices 2>/dev/null | awk 'NR==2{print $1}')"
    if adb shell pm list packages 2>/dev/null | grep -q com.visualmemory.glasses; then
        ok "app installed"
    else
        bad "app not installed" "build and install: see apps/glasses-x3/README.md"
    fi
    for perm in CAMERA RECORD_AUDIO; do
        if adb shell dumpsys package com.visualmemory.glasses 2>/dev/null \
            | grep -q "$perm: granted=true"; then
            ok "$perm granted"
        else
            bad "$perm not granted" \
                "adb shell pm grant com.visualmemory.glasses android.permission.$perm"
        fi
    done
else
    skip "no adb device attached"
    NOTE+=("on WSL the device needs usbipd: usbipd bind --busid <id>; usbipd attach --wsl --busid <id>")
fi

# --- Known failure signatures in the logs -----------------------------------

head_ "Recent log signatures"
sig() { # file pattern human-meaning
    local n
    # `grep -c` already prints 0 when it matches nothing, so a `|| echo 0`
    # fallback emits *two* values and the arithmetic below then explodes.
    [[ -f "logs/$1" ]] || return 0
    # Only the tail. An all-time count reports failures that were fixed hours
    # ago as if they were happening now, which is worse than not looking: it
    # sends you chasing a symptom that has already gone away.
    n=$(tail -n "${SIG_WINDOW:-300}" "logs/$1" 2>/dev/null | grep -c "$2" | head -1)
    n="${n:-0}"
    [[ "$n" -gt 0 ]] && printf '  \033[33m!\033[0m    %s (x%s in last %s lines of logs/%s)\n' \
        "$3" "$n" "${SIG_WINDOW:-300}" "$1"
}
sig gateway.log "signal_stream - connecting" "gateway retrying LiveKit -- ICE may not survive the topology"
sig livekit.log "JOIN_TIMEOUT" "a participant never completed ICE"
sig livekit.log "removing participant without connection" "signalling succeeded, media never did"
sig speech.log "length limit before any silence" "VAD never found silence; utterances cut at max_seconds"
sig speech.log "CUDA out of memory" "Parakeet OOM -- lower stt_utterance_max_seconds or free the GPU"
sig agent.log "hands-free transcript processing failed" "agent could not reach Memory/Speech; check tokens match"

# --- Verdict ----------------------------------------------------------------

head_ "Verdict"
printf '  %d passed, %d failed\n' "$PASS" "$FAIL"
if [[ "$FAIL" -gt 0 ]]; then
    printf '\n\033[1mDo this next\033[0m\n'
    for n in "${NOTE[@]+"${NOTE[@]}"}"; do printf '  - %s\n' "$n"; done
fi
exit "$FAIL"
