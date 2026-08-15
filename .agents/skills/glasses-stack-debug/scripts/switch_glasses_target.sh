#!/usr/bin/env bash
# Re-pair the attached glasses to laptop or GN100 without clearing app data.
# The app's existing pairing_payload intent extra claims a fresh single-use
# credential while PairingStore preserves the stable device ID.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  switch_glasses_target.sh laptop [--check]
  VMA_TARGET_INTERNAL_API_TOKEN=... switch_glasses_target.sh http://<gn100-ip>:8080 [--check]

The target Gateway must already be healthy. For GN100, export its operator token
in the current shell (leading space if shell history is configured to ignore
such commands), or point VMA_TARGET_INTERNAL_API_TOKEN_FILE at a mode-600 file.
Never pass a token as a command-line argument.

Requires ADB for the switch operation only. Wi-Fi media remains untethered.
The GN100 VMA_DEVICE_ID_ALLOWLIST must contain this glasses device ID.
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
TARGET=$1
CHECK_ONLY=0
case "$TARGET" in -h|--help) usage; exit 0 ;; esac
if [[ $# -eq 2 ]]; then
    [[ "$2" == "--check" ]] || { usage >&2; exit 2; }
    CHECK_ONLY=1
fi

fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }
pass() { printf '\033[32mPASS\033[0m %s\n' "$*"; }

command -v adb >/dev/null 2>&1 || fail "adb is not installed"
DEVICE_COUNT="$(adb devices 2>/dev/null | awk 'NR>1 && $2=="device" {n++} END {print n+0}')"
[[ "$DEVICE_COUNT" -eq 1 ]] || fail "expected exactly one authorized ADB device, found $DEVICE_COUNT"

if [[ "$TARGET" == "laptop" ]]; then
    # shellcheck disable=SC1091
    set -a; . ./.env; set +a
    GATEWAY_URL="$(printf '%s' "$VMA_LIVEKIT_PUBLIC_URL" | sed -E 's#^ws://#http://#; s#:7880$#:8080#')"
    TOKEN="${VMA_INTERNAL_API_TOKEN:-}"
    LABEL=laptop
else
    GATEWAY_URL="${TARGET%/}"
    [[ "$GATEWAY_URL" =~ ^https?://([0-9]{1,3}\.){3}[0-9]{1,3}:8080$ ]] \
        || fail "target must be http://<IPv4>:8080 or https://<IPv4>:8080"
    if [[ -n "${VMA_TARGET_INTERNAL_API_TOKEN_FILE:-}" ]]; then
        [[ -f "$VMA_TARGET_INTERNAL_API_TOKEN_FILE" ]] || fail "target token file does not exist"
        TOKEN_MODE="$(stat -c '%a' "$VMA_TARGET_INTERNAL_API_TOKEN_FILE" 2>/dev/null || true)"
        [[ "$TOKEN_MODE" == "600" || "$TOKEN_MODE" == "400" ]] \
            || fail "target token file must be mode 600 or 400 (found ${TOKEN_MODE:-unknown})"
        TOKEN="$(<"$VMA_TARGET_INTERNAL_API_TOKEN_FILE")"
    else
        TOKEN="${VMA_TARGET_INTERNAL_API_TOKEN:-}"
    fi
    LABEL=GN100
fi
[[ -n "$TOKEN" ]] || fail "target internal API token is not configured"

curl -fsS --max-time 4 "$GATEWAY_URL/health/live" >/dev/null \
    || fail "$LABEL Gateway is not reachable at $GATEWAY_URL"
pass "$LABEL Gateway reachable at $GATEWAY_URL"
if [[ "$CHECK_ONLY" == "1" ]]; then
    pass "ADB, target URL, operator token, and Gateway preflight passed"
    exit 0
fi

# Keep the bearer out of argv/process listings by feeding curl a mode-600
# config file. The response is a short-lived single-use code, never the token.
CURL_CONFIG="$(mktemp)"
chmod 600 "$CURL_CONFIG"
cleanup() { rm -f "$CURL_CONFIG"; }
trap cleanup EXIT
printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" >"$CURL_CONFIG"
PAIRING_RESPONSE="$(curl -fsS --max-time 8 --config "$CURL_CONFIG" \
    -X POST "$GATEWAY_URL/v1/pairing")" \
    || fail "target refused to issue a pairing code"

PAIRING_PAYLOAD="$(PAIRING_RESPONSE="$PAIRING_RESPONSE" GATEWAY_URL="$GATEWAY_URL" python3 - <<'PY'
import json
import os

issued = json.loads(os.environ["PAIRING_RESPONSE"])
print(json.dumps({
    "gateway_url": os.environ["GATEWAY_URL"],
    "pairing_code": issued["pairing_code"],
    "expires_at": issued["expires_at"],
}, separators=(",", ":")))
PY
)" || fail "target returned an invalid pairing response"

# Force-stop first so claiming a new target cannot overlap the old room/session.
# Do not `pm clear`: PairingStore's stable device ID must survive the switch.
adb shell am force-stop com.visualmemory.glasses
adb shell pm grant com.visualmemory.glasses android.permission.CAMERA >/dev/null
adb shell pm grant com.visualmemory.glasses android.permission.RECORD_AUDIO >/dev/null
adb shell am start -n com.visualmemory.glasses/.MainActivity \
    --es pairing_payload "$PAIRING_PAYLOAD" >/dev/null

for _ in $(seq 1 15); do
    PID="$(adb shell pidof com.visualmemory.glasses 2>/dev/null | tr -d '\r')"
    if [[ -n "$PID" ]]; then
        break
    fi
    sleep 1
done
[[ -n "${PID:-}" ]] || fail "glasses app did not stay running"

pass "sent a fresh $LABEL pairing to the glasses (app pid $PID)"
printf 'Watch the %s console/session list; the device should publish within seconds.\n' "$LABEL"
