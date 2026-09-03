# The GPU box itself -- created before a demo, destroyed after it.
#
# Everything expensive and everything slow lives in `persistent/`. This state
# holds only the instance, so `terraform destroy` here stops the meter without
# touching the model cache or the public address.
#
# Driven by ../up.sh and ../down.sh rather than run directly, because the part
# that matters operationally is not `apply` returning -- it is the readiness
# gate afterwards. `apply` finishes when the VM exists, which is several minutes
# before any model can answer a request.

terraform {
  required_version = ">= 1.6"
  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.8"
    }
  }
}

provider "nebius" {
  service_account = {
    account_id_env       = "NEBIUS_SA_ID"
    public_key_id_env    = "NEBIUS_SA_PUBLIC_KEY_ID"
    private_key_file_env = "NEBIUS_SA_PRIVATE_KEY_FILE"
  }
}

# Read what persistent/ built rather than duplicating ids into tfvars, so the
# two states cannot drift.
data "terraform_remote_state" "persistent" {
  backend = "local"
  config = {
    path = "${path.module}/../persistent/terraform.tfstate"
  }
}

# --- What card, and is it allowed to vanish ---------------------------------

variable "platform" {
  type        = string
  default     = "gpu-h200-sxm"
  description = <<-EOT
    Nebius GPU platform. The four in play, with their single-GPU presets:

      gpu-h200-sxm   141 GB  $4.50/hr  1gpu-16vcpu-200gb   safe; today's pins work
      gpu-h100-sxm    80 GB  $3.85/hr  1gpu-16vcpu-200gb   fits only if KV is capped
      gpu-rtx6000     96 GB  $1.80/hr  1gpu-24vcpu-218gb   Blackwell sm_120 -- see below
      gpu-l40s-d      48 GB  from $1.55/hr  1gpu-16vcpu-96gb   only if Cosmos is replaced

    RTX PRO 6000 is the cheapest card that fits the current stack, but it is
    Blackwell: services/vision-worker pins torch==2.6.0 on cu126 for x86_64,
    which does not emit sm_120. Selecting it without bumping torch to >=2.7/cu128
    gives a box where vLLM runs and vision-worker cannot start. At roughly one
    demo-hour at a time the saving is ~$2.70 a demo, so the bump is usually not
    worth it -- but the option is here once the volume justifies it.

    B200/B300 are deliberately absent: Nebius has no single-GPU preset for them.
  EOT
}

variable "preset" {
  type        = string
  default     = "1gpu-16vcpu-200gb"
  description = "Must match the platform; see the table above."
}

variable "preemptible" {
  type        = bool
  default     = false
  description = <<-EOT
    Roughly half price, and reclaimable at any moment.

    Correct for a measurement run (vram_probe / the bake-off), which can simply
    be restarted. Never correct for a customer demo: losing the box mid-call is
    a worse outcome than any saving. Defaults to false so the dangerous choice
    is the one you have to type.
  EOT
}

variable "boot_disk_gibibytes" {
  type        = number
  default     = 128
  description = "Root disk. Container images and CUDA userspace, not weights."
}

variable "image_family" {
  type        = string
  default     = "ubuntu22.04-cuda12"
  description = <<-EOT
    Driver and CUDA userspace pre-baked. This is load-bearing for the 20-minute
    lead time: installing an NVIDIA driver at boot costs 5-8 minutes and can
    fail in ways that are invisible until the first CUDA call.
    `nebius compute image list` for what the project can see.
  EOT
}

variable "ssh_public_key" {
  type        = string
  description = "Operator's SSH public key, injected via cloud-init."
}

variable "wireguard_peer_public_key" {
  type        = string
  description = <<-EOT
    The laptop's WireGuard public key.

    WireGuard rather than an SSH tunnel because the glasses stream is WebRTC and
    SSH forwards TCP only. Forcing media onto LiveKit's TCP fallback (7881)
    reintroduces head-of-line blocking on a video path that took an H.265 switch
    and a bitrate floor to get to 15 fps. WireGuard carries UDP 7882, so the
    trusted-LAN assumptions in deploy/livekit.yaml stay true.
  EOT
}

variable "wireguard_listen_port" {
  type    = number
  default = 51820
}

variable "repo_url" {
  type        = string
  default     = "https://github.com/NadChern/nvidia_hack_seattle.git"
  description = "Public, so the box needs no deploy key and holds no secret."
}

variable "repo_ref" {
  type        = string
  default     = "main"
  description = <<-EOT
    Branch or tag to check out. Pin to a tag before a customer demo: "whatever
    main was that morning" is not a thing to discover during a call.
  EOT
}

variable "hf_token" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Only needed to seed the cache for a gated checkpoint; unused once warm."
}

variable "labels" {
  type    = map(string)
  default = { project = "memo", tier = "session" }
}

locals {
  parent_id = data.terraform_remote_state.persistent.outputs.parent_id
  subnet_id = data.terraform_remote_state.persistent.outputs.subnet_id

  #: The tag the instance mounts the model filesystem under. Matched by
  #: cloud-init's mount unit; changing it here alone silently yields an
  #: instance with no cache and a 40-minute first boot.
  models_mount_tag = "models"
}

resource "nebius_compute_v1_instance" "demo" {
  parent_id = local.parent_id
  name      = "memo-demo"
  hostname  = "memo-demo"
  labels    = var.labels

  resources = {
    platform = var.platform
    preset   = var.preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    managed_disk = {
      name = "memo-demo-boot"
      spec = {
        type           = "NETWORK_SSD"
        size_gibibytes = var.boot_disk_gibibytes
        source_image_family = {
          image_family = var.image_family
        }
      }
    }
  }

  # The warm model cache from persistent/. This is why boot-to-ready is minutes
  # rather than the better part of an hour.
  filesystems = [{
    attach_mode         = "READ_WRITE"
    mount_tag           = local.models_mount_tag
    existing_filesystem = { id = data.terraform_remote_state.persistent.outputs.models_filesystem_id }
  }]

  network_interfaces = [{
    name      = "eth0"
    subnet_id = local.subnet_id
    ip_address = {}
    public_ip_address = {
      allocation_id = data.terraform_remote_state.persistent.outputs.public_allocation_id
      static        = true
    }
  }]

  cloud_init_user_data = templatefile("${path.module}/../cloud-init.yaml.tftpl", {
    ssh_public_key            = var.ssh_public_key
    wireguard_peer_public_key = var.wireguard_peer_public_key
    wireguard_listen_port     = var.wireguard_listen_port
    models_mount_tag          = local.models_mount_tag
    hf_token                  = var.hf_token
    repo_url                  = var.repo_url
    repo_ref                  = var.repo_ref
  })

  # RECOVER would silently rebuild a box mid-demo and lose every warm model.
  # A visible failure is the better outcome: the operator can decide.
  recovery_policy = "FAIL"

  # A nested attribute, not a block -- null means "not preemptible" rather than
  # requiring a separate resource for each case.
  preemptible = var.preemptible ? { on_preemption = "STOP" } : null
}

output "instance_id" { value = nebius_compute_v1_instance.demo.id }
output "public_ip" { value = data.terraform_remote_state.persistent.outputs.public_ip }
output "preemptible" { value = var.preemptible }
output "platform" { value = "${var.platform} / ${var.preset}" }
