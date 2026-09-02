#!/usr/bin/env bash
# Is anything running, and is it ready? Cheap enough to run before a call.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF="${TF:-terraform}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ ! -f "$HERE/session/terraform.tfstate" ]]; then
  echo "session: not running (no meter)"
  exit 0
fi
IP="$("$TF" -chdir="$HERE/session" output -raw public_ip 2>/dev/null || true)"
PLATFORM="$("$TF" -chdir="$HERE/session" output -raw platform 2>/dev/null || echo unknown)"
PREEMPT="$("$TF" -chdir="$HERE/session" output -raw preemptible 2>/dev/null || echo unknown)"
[[ -n "$IP" ]] || { echo "session: state exists but no IP -- apply may have failed"; exit 1; }

echo "session: $PLATFORM at $IP (preemptible=$PREEMPT)"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -i "$SSH_KEY" \
    "memo@$IP" /usr/local/bin/memo-ready 2>/dev/null \
  && echo "status : READY" \
  || echo "status : NOT READY (see the per-check lines above)"
