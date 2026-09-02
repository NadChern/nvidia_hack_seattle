#!/usr/bin/env bash
# Stop the meter. Keeps the model cache and the public address.
#
# Destroys `session/` only. `persistent/` holds the checkpoints and the static
# IP and is never touched here -- that split is the whole design, because
# re-downloading 50+ GiB of weights costs more than the lead time before a call.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF="${TF:-terraform}"

for var in NEBIUS_SA_ID NEBIUS_SA_PUBLIC_KEY_ID NEBIUS_SA_PRIVATE_KEY_FILE; do
  [[ -n "${!var:-}" ]] || { echo "error: $var is unset" >&2; exit 1; }
done

if [[ ! -f "$HERE/session/terraform.tfstate" ]]; then
  echo "no session state -- nothing running."
  exit 0
fi

echo "==> destroying the GPU instance (model cache and IP survive)"
"$TF" -chdir="$HERE/session" destroy -input=false -auto-approve

echo
echo "Stopped. Standing cost is now the model filesystem and the static IP only."
echo "Next demo: deploy/nebius/up.sh"
