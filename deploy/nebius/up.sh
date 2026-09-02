#!/usr/bin/env bash
# Bring the demo box up and do not return until it can actually answer.
#
# The reason this is a script and not `terraform apply` is the last part.
# `apply` returns when the VM exists, which is several minutes before any model
# can serve a request -- 50+ GiB of weights have to reach the GPU first. Twenty
# minutes before a customer call, "Terraform said OK" is not the question. The
# question is whether the stack answers, and this polls until it does.
#
#   deploy/nebius/up.sh                    # H200, on-demand -- the demo default
#   deploy/nebius/up.sh --measure          # preemptible, for a probe run
#   deploy/nebius/up.sh --platform gpu-rtx6000 --preset 1gpu-24vcpu-218gb
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="$HERE/session"
PERSISTENT="$HERE/persistent"
TF="${TF:-terraform}"
READY_TIMEOUT="${READY_TIMEOUT:-1500}"   # 25 min: first boot pulls images
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

TF_ARGS=()
PREEMPTIBLE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --measure)   PREEMPTIBLE=true; shift ;;
    --platform)  TF_ARGS+=(-var "platform=$2"); shift 2 ;;
    --preset)    TF_ARGS+=(-var "preset=$2"); shift 2 ;;
    -h|--help)   sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *)           echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
TF_ARGS+=(-var "preemptible=$PREEMPTIBLE")

die() { echo "error: $*" >&2; exit 1; }

for var in NEBIUS_SA_ID NEBIUS_SA_PUBLIC_KEY_ID NEBIUS_SA_PRIVATE_KEY_FILE; do
  [[ -n "${!var:-}" ]] || die "$var is unset. See deploy/nebius/README.md."
done
[[ -f "$PERSISTENT/terraform.tfstate" ]] || die \
  "no persistent state. Run: $TF -chdir=$PERSISTENT apply   (once, ever)"

# A preemptible box can be reclaimed mid-sentence. Fine for a probe run that can
# be restarted; not fine while a customer is watching.
if [[ "$PREEMPTIBLE" == true ]]; then
  echo "!! preemptible: ~half price, reclaimable at any moment."
  echo "!! Correct for a measurement run. NEVER for a customer demo."
fi

echo "==> provisioning"
"$TF" -chdir="$SESSION" init -input=false >/dev/null
"$TF" -chdir="$SESSION" apply -input=false -auto-approve "${TF_ARGS[@]}"

IP="$("$TF" -chdir="$SESSION" output -raw public_ip)"
PLATFORM="$("$TF" -chdir="$SESSION" output -raw platform)"
[[ -n "$IP" ]] || die "no public IP in session outputs"
echo "==> $PLATFORM at $IP"

ssh_box() {
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
      -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
      -i "$SSH_KEY" "memo@$IP" "$@" 2>/dev/null
}

echo "==> waiting for cloud-init"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until ssh_box test -f /var/lib/cloud/memo-provisioned; do
  (( $(date +%s) < deadline )) || die "cloud-init did not finish in ${READY_TIMEOUT}s"
  sleep 10
done
echo "    provisioned"

# Hand back the server's WireGuard public key: the laptop needs it to complete
# its peer config, and it is generated on the box at first boot rather than
# committed anywhere.
WG_PUB="$(ssh_box cat /etc/wireguard/server.pub || true)"

echo "==> starting the stack"
ssh_box "cd /opt/memo && sudo docker compose --env-file /etc/memo/stack.env \
  -f compose.yaml -f compose.gpu.yaml -f compose.cloud.yaml up -d" \
  || die "compose failed; ssh in and check: sudo docker compose logs"

# The gate that matters. Model servers load weights long after the container
# reports started, so this polls the endpoints rather than the containers.
echo "==> waiting for models (this is the slow part -- weights to GPU)"
until ssh_box /usr/local/bin/memo-ready; do
  if (( $(date +%s) >= deadline )); then
    echo
    echo "NOT READY within ${READY_TIMEOUT}s. Last check above names what is missing."
    echo "The box is still running and still costing money -- deploy/nebius/down.sh to stop it."
    exit 1
  fi
  sleep 15
done

cat <<INFO

=========================================================================
READY   $PLATFORM   $IP

WireGuard peer config for the laptop -- /etc/wireguard/memo.conf:

  [Interface]
  Address = 10.10.0.2/24
  PrivateKey = <the laptop's private key>

  [Peer]
  PublicKey = ${WG_PUB:-<ssh in: cat /etc/wireguard/server.pub>}
  Endpoint = $IP:51820
  AllowedIPs = 10.10.0.0/24
  PersistentKeepalive = 25

  sudo wg-quick up memo

Then point the glasses at  ws://10.10.0.1:7880  -- media stays in the tunnel.

  ssh memo@$IP                     shell on the box
  deploy/nebius/down.sh            stop the meter (keeps the model cache)
=========================================================================
INFO
