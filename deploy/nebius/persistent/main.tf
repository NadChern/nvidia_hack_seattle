# What must outlive the GPU -- the weights, the address, the network.
#
# The demo box is ephemeral by design: it comes up ~20 minutes before a customer
# call and is destroyed straight after, because a GPU costs $1.80-$4.50/hour and
# a call lasts one. That only works if the 50-125 GiB of checkpoints do NOT come
# down again on every boot. Downloading them each time takes longer than the
# entire lead time and would make the whole approach unusable.
#
# So this state is applied once and left alone. `session/` is applied and
# destroyed around each demo. Destroying a session leaves everything here
# standing, and the next boot mounts a warm cache.
#
#   terraform -chdir=deploy/nebius/persistent apply     # once, ever
#   deploy/nebius/up.sh                                 # before each demo
#   deploy/nebius/down.sh                               # after each demo
#
# Standing cost is this file's contents only: the filesystem, plus a static
# public IP. See README.md for the arithmetic.

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
  # Service-account auth via environment, so no key material is ever written
  # into a .tf file or committed. See README.md for generating the pair.
  service_account = {
    account_id_env       = "NEBIUS_SA_ID"
    public_key_id_env    = "NEBIUS_SA_PUBLIC_KEY_ID"
    private_key_file_env = "NEBIUS_SA_PRIVATE_KEY_FILE"
  }
}

variable "parent_id" {
  type        = string
  description = "Nebius project id (project-...). `nebius iam project list`."
}

variable "subnet_id" {
  type        = string
  description = "VPC subnet id (vpcsubnet-...). `nebius vpc subnet list`."
}

variable "models_size_gibibytes" {
  type        = number
  default     = 256
  description = <<-EOT
    Sized for the bake-off, not for production. All five grounder arms plus
    Nemotron and the speech models is ~125 GiB of checkpoints; 256 GiB leaves
    room for HF's blob/ref duplication during download.

    Once a grounder is chosen, ~96 GiB holds the shipping set (winner +
    Nemotron + C-RADIO + SAM2 + Parakeet + Kokoro) and cuts this line's cost by
    more than half. Shrinking a filesystem is not an in-place operation, so
    revisit this deliberately after the bake-off rather than guessing now.
  EOT
}

variable "labels" {
  type    = map(string)
  default = { project = "memo", tier = "persistent" }
}

# The model cache. A filesystem rather than a secondary disk because it is
# mounted by tag into whatever instance currently exists, which is exactly the
# lifetime mismatch this split exists to express -- and because a filesystem
# cannot be accidentally reformatted by a boot script the way a raw block
# device can.
resource "nebius_compute_v1_filesystem" "models" {
  parent_id      = var.parent_id
  name           = "memo-models"
  type           = "NETWORK_SSD"
  size_gibibytes = var.models_size_gibibytes
  labels         = var.labels

  # The entire point of this file. Without it, a stray `terraform destroy` in
  # the wrong directory silently deletes every checkpoint and the next demo
  # spends 40 minutes downloading instead of 20 minutes warming.
  forbid_deletion = true
}

# A static address, so the WireGuard peer config on the laptop and the glasses
# does not have to be rewritten before every demo. Allocated here rather than
# in `session/` precisely because a per-session address would defeat that.
resource "nebius_vpc_v1_allocation" "public" {
  parent_id = var.parent_id
  name      = "memo-public-ip"
  labels    = var.labels

  ipv4_public = {
    subnet_id = var.subnet_id
  }
}

output "models_filesystem_id" {
  value       = nebius_compute_v1_filesystem.models.id
  description = "Mounted into each session instance by tag."
}

output "public_allocation_id" {
  value       = nebius_vpc_v1_allocation.public.id
  description = "Static public IP; the WireGuard endpoint the laptop dials."
}

output "public_ip" {
  # The provider reports the assigned address as a CIDR under `status.details`,
  # not on the `ipv4_public` request block -- that block is what was asked for,
  # this is what was granted. Stripping the mask gives the dialable address.
  value       = try(split("/", nebius_vpc_v1_allocation.public.status.details.allocated_cidr)[0], null)
  description = "Put this in the laptop's WireGuard peer Endpoint."
}

output "parent_id" { value = var.parent_id }
output "subnet_id" { value = var.subnet_id }
