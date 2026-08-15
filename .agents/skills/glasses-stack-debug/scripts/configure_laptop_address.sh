#!/usr/bin/env bash
# Rewrite only the public laptop addresses after moving to another Wi-Fi LAN.
# Secrets and the browser's loopback LiveKit route are left untouched.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: configure_laptop_address.sh [IPv4] [--check]

Without an address, detects the source IPv4 of the current default route.
Updates:
  .env: VMA_LIVEKIT_URL and VMA_LIVEKIT_PUBLIC_URL
  apps/console/.env.local: VITE_VMA_GATEWAY_PUBLIC_URL

It deliberately keeps VITE_VMA_LIVEKIT_URL=ws://127.0.0.1:7880 because the
Windows browser reaches LiveKit over mirrored loopback while glasses use Wi-Fi.
Run before starting the stack. A changed address requires re-pairing/switching
the glasses because their persisted Gateway URL cannot redirect itself.
EOF
}

CHECK_ONLY=0
REQUESTED=""
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            [[ -z "$REQUESTED" ]] || { usage >&2; exit 2; }
            REQUESTED="$arg"
            ;;
    esac
done

fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }
pass() { printf '\033[32mPASS\033[0m %s\n' "$*"; }

DETECTED="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
IP="${REQUESTED:-$DETECTED}"
[[ "$IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || fail "could not determine a valid IPv4 address"
IFS=. read -r a b c d <<<"$IP"
for octet in "$a" "$b" "$c" "$d"; do
    (( 10#$octet >= 0 && 10#$octet <= 255 )) || fail "invalid IPv4 address: $IP"
done
ip -4 addr | grep -q "inet $IP/" || fail "$IP is not assigned to this WSL instance"

[[ -f .env ]] || fail ".env is missing"
[[ -f apps/console/.env.local ]] || fail "apps/console/.env.local is missing"

CURRENT="$(grep -E '^VMA_LIVEKIT_PUBLIC_URL=ws://[0-9.]+:7880$' .env | sed -E 's#.*ws://([^:]+):.*#\1#' || true)"
if [[ "$CHECK_ONLY" == "1" ]]; then
    [[ "$CURRENT" == "$IP" ]] || fail "configured laptop address is ${CURRENT:-unset}, current address is $IP"
    pass "laptop public URLs already use $IP"
    exit 0
fi

for port in 8080 5173 7880; do
    if ss -ltn 2>/dev/null | grep -qE "[:.]$port "; then
        fail "port $port is active; stop the existing stack cleanly before changing its public address"
    fi
done

replace_key() {
    local file=$1 key=$2 value=$3
    if grep -q "^${key}=" "$file"; then
        sed -i -E "s#^${key}=.*#${key}=${value}#" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >>"$file"
    fi
}

replace_key .env VMA_LIVEKIT_URL "ws://$IP:7880"
replace_key .env VMA_LIVEKIT_PUBLIC_URL "ws://$IP:7880"
replace_key apps/console/.env.local VITE_VMA_GATEWAY_PUBLIC_URL "http://$IP:8080"
replace_key apps/console/.env.local VITE_VMA_LIVEKIT_URL "ws://127.0.0.1:7880"

pass "configured laptop Gateway/LiveKit public address $IP"
if [[ -n "$CURRENT" && "$CURRENT" != "$IP" ]]; then
    printf '\nAddress changed from %s to %s. Existing glasses pairing points at the old host.\n' "$CURRENT" "$IP"
    printf 'After startup, switch without QR using:\n'
    printf '  %s laptop\n' ".agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh"
fi
