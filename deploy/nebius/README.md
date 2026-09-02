# Nebius demo deployment

A GPU box that comes up ~20 minutes before a customer call and is destroyed
straight after it. That shape drives every decision here.

```bash
terraform -chdir=deploy/nebius/persistent apply   # once, ever
deploy/nebius/up.sh                               # before each demo
deploy/nebius/status.sh                           # is it ready?
deploy/nebius/down.sh                             # after each demo
```

## Why the state is split in two

| | `persistent/` | `session/` |
|---|---|---|
| Holds | model filesystem, static public IP | the GPU instance |
| Lifetime | applied once, left alone | created and destroyed per demo |
| Cost | ~$20/mo standing | $1.80–$4.50 per hour running |

The split exists for one reason: **the checkpoints must not be downloaded
again on every boot.** Cosmos alone is ~32 GiB and the full set is 50–125 GiB.
Pulling that fresh takes longer than the entire lead time before a call, which
would make the ephemeral approach unusable. Instead the weights live on a
filesystem that outlives the GPU, and a session boot mounts a warm cache.

`persistent/` also carries `forbid_deletion = true` on the filesystem, so a
`terraform destroy` run in the wrong directory cannot silently delete every
checkpoint.

## First-time setup

```bash
# 1. Service account + key pair (the provider reads these from the environment,
#    so no key material is ever written into a .tf file)
nebius iam service-account create --name memo-deploy
openssl genrsa -out ~/.nebius/memo-deploy.pem 4096
nebius iam auth-public-key create \
  --service-account-id "$SA_ID" --data "$(openssl rsa -in ~/.nebius/memo-deploy.pem -pubout)"

export NEBIUS_SA_ID=serviceaccount-...
export NEBIUS_SA_PUBLIC_KEY_ID=publickey-...
export NEBIUS_SA_PRIVATE_KEY_FILE=$HOME/.nebius/memo-deploy.pem

# 2. Project and subnet ids
nebius iam project list
nebius vpc subnet list

# 3. WireGuard key pair on the laptop
wg genkey | tee ~/.wg/memo.key | wg pubkey > ~/.wg/memo.pub

# 4. terraform.tfvars in each directory
cat > deploy/nebius/persistent/terraform.tfvars <<EOF
parent_id = "project-..."
subnet_id = "vpcsubnet-..."
EOF
cat > deploy/nebius/session/terraform.tfvars <<EOF
ssh_public_key            = "ssh-ed25519 AAAA... you@laptop"
wireguard_peer_public_key = "$(cat ~/.wg/memo.pub)"
EOF

# 5. Once, ever
terraform -chdir=deploy/nebius/persistent apply

# 6. Seed the cache -- the one slow boot. Bring a box up and pull the
#    checkpoints onto /mnt/models before the first real demo.
deploy/nebius/up.sh --measure
```

`terraform.tfvars` files are gitignored: they carry project ids and public keys
that are not secret but are not ours to publish either.

## Picking the card

`up.sh` defaults to H200 on-demand. The four single-GPU options:

| Platform | Preset | VRAM | On-demand | Note |
|---|---|---|---|---|
| `gpu-h200-sxm` | `1gpu-16vcpu-200gb` | 141 GB | $4.50/hr | default; today's torch pins work |
| `gpu-h100-sxm` | `1gpu-16vcpu-200gb` | 80 GB | $3.85/hr | fits only with KV capped — measure first |
| `gpu-rtx6000` | `1gpu-24vcpu-218gb` | 96 GB | **$1.80/hr** | Blackwell — see below |
| `gpu-l40s-d` | `1gpu-16vcpu-96gb` | 48 GB | from $1.55/hr | only if Cosmos is replaced |

B200/B300 have no single-GPU preset on Nebius and are not options here.

**The RTX PRO 6000 trap.** It is the cheapest card that fits the current stack,
but it is Blackwell (sm_120), and `services/vision-worker/pyproject.toml` pins
`torch==2.6.0` on cu126 for x86_64 — which does not emit sm_120. Select it
without bumping torch to ≥2.7/cu128 and you get a box where vLLM runs fine and
vision-worker cannot start at all. At one demo-hour at a time the saving is
about **$2.70 a demo**, so the bump usually is not worth it; the option is here
for when demo volume changes that.

**Preemptible** (`--measure`) is roughly half price and can be reclaimed at any
moment. Right for a `vram_probe` or bake-off run, which can just be restarted.
Never right for a customer demo.

## Why WireGuard and not an SSH tunnel

The glasses stream is WebRTC and wants UDP 7882. SSH forwards TCP only, so a
tunnel would push media onto LiveKit's TCP fallback (7881) and reintroduce
head-of-line blocking on a video path that took an H.265 switch and a bitrate
floor to get to 15 fps in the first place.

WireGuard carries UDP, costs one config file, and keeps
[deploy/livekit.yaml](../livekit.yaml)'s trusted-LAN assumptions
(`use_external_ip: false`, no TURN) true — the box exposes exactly one port to
the internet. `up.sh` prints the laptop's peer config, including the server
public key generated on the box at first boot.

Glasses then point at `ws://10.10.0.1:7880`; media stays inside the tunnel.

## What `up.sh` waits for, and why that is not `terraform apply`

`apply` returns when the VM exists. That is several minutes before anything can
answer a request, because 50+ GiB of weights still have to reach the GPU.
Twenty minutes before a customer call the useful question is not whether
Terraform succeeded — so `up.sh` polls `/usr/local/bin/memo-ready` on the box,
which checks the model cache mount, the tunnel, both model servers, and the
three application services, and names whichever one is not up yet.

Rough budget against the 20–30 minute lead time:

| Step | Cold | Warm cache |
|---|---|---|
| instance create + boot | ~3 min | ~3 min |
| driver + CUDA | 0 (pre-baked in the image) | 0 |
| container images | 5–10 min | ~1 min |
| **weights to GPU** | **40+ min (downloading)** | **4–8 min** |
| service warmup | ~2 min | ~2 min |

The middle row is the entire argument for `persistent/`.

## The GPU-fraction trap

`compose.cloud.yaml` sets `--gpu-memory-utilization` explicitly on both model
servers. vLLM's default is 0.9 **of the whole card** and it takes what it is
offered. Two servers both defaulting to 0.9 do not split the card — the first
one started takes ~90% and the second dies at startup with an allocation error
that reads like a model problem.

Current values (`0.20` Nemotron, `0.35` grounder) are **derived from weight
sizes, not measured.** `docs/spikes/cloud-vram/` exists to replace them with
real numbers; its `runs/` is empty until a card is rented. A stack that OOMs on
startup is a sizing result, not a bug — record it there.

## Known gaps

- **Nothing here has been applied.** The Terraform validates against the real
  `nebius/nebius` provider schema and the cloud-init renders to valid YAML, but
  no run has touched Nebius. Expect the first `apply` to surface project-specific
  ids and quota limits.
- **`compose config` has not been run** on the three-file merge — the merge is
  structurally checked (service names, `depends_on`, networks) but not by Docker.
- **`/opt/memo` is assumed to exist on the box** with the repository checked out.
  Seeding it is currently a manual step in the first `--measure` boot; folding it
  into cloud-init needs a deploy key or a public clone and is deliberately not
  guessed at here.
- **Image family `ubuntu22.04-cuda12` is assumed.** Confirm against
  `nebius compute image list` for the project before the first apply.
