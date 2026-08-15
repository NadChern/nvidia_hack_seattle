# Local LiveKit for development

Everything the Media Gateway needs on a dev machine: a LiveKit server, the
credentials to reach it, and a way to check it is not listening anywhere it
should not be.

Both paths read the same [`livekit.dev.yaml`](livekit.dev.yaml) and credentials,
but they are not network-topology interchangeable. WSL2 with real glasses must
use the native binary because Docker bridge NAT breaks ICE. Apple-silicon macOS
has no official v1.13.4 Darwin binary and therefore uses Docker or an explicitly
accepted unpinned Homebrew build. The pinned Docker path carried physical
RayNeo X3 Pro media on an Apple-silicon Mac on 2026-08-14; follow the complete
[macOS runbook](../../.agents/skills/glasses-stack-debug/MACOS.md).

## Credentials first

The config file deliberately carries **no `keys:` block**. Both the server and
the gateway read the same pair from the environment, so they cannot drift, and
nothing secret is committed:

```bash
export VMA_LIVEKIT_API_KEY=vma-dev
export VMA_LIVEKIT_API_SECRET=$(openssl rand -hex 24)
```

Generate your own. The gateway's startup validator **refuses** LiveKit's
well-known `devkey`/`secret` and the spike's published pair, and requires at
least 32 characters, so a copied credential fails loudly rather than quietly
shipping. See [Privacy and Security](../../docs/07-Privacy-and-Security.md).

## Path A — Docker Compose (Mac and loopback development)

```bash
docker compose -f compose.dev.yaml up -d livekit
```

For physical glasses on a Mac, use `scripts/dev_stack.sh --allow-lan` as shown
in the macOS runbook rather than starting this service by hand. The launcher
generates and shares credentials, publishes the required trusted-LAN ports,
checks listener exposure, and supervises shutdown.

Ports are mapped explicitly to loopback rather than using `network_mode: host`:
host networking for Linux containers is a separate opt-in on Docker Desktop and
behaves differently there than on native Linux, while explicit mappings work
identically on WSL2, macOS, and Windows.

**The `7882/udp` mapping is easy to lose in an edit, and ICE degrades silently
without it.** `check_listeners.py` cannot probe a UDP port over TCP, so if media
stalls but signaling works, check that mapping first.

## Path B — pinned binary (no Docker)

```bash
python3 tools/dev-livekit/get_livekit_server.py
```

Stdlib only, so it runs before you have a virtualenv. It downloads LiveKit
**1.13.4** — the version the S01 spike validated and the version
`compose.dev.yaml` pins — verifies the archive against a committed SHA-256, and
extracts to a gitignored `.tools/`. A mismatch deletes the archive rather than
leaving something unverified on disk to be extracted by hand.

Then start it with the command the script prints:

```bash
export LIVEKIT_KEYS="$VMA_LIVEKIT_API_KEY: $VMA_LIVEKIT_API_SECRET"
./.tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml
```

**macOS is not covered by this path.** LiveKit publishes no darwin binary for
v1.13.4 — only linux amd64/arm64/armv7 and windows amd64/arm64. On a Mac use
Path A, or `brew install livekit` and accept an unpinned version.

Do **not** use `livekit-server --dev`. It runs with the well-known
`devkey`/`secret`, which the gateway refuses.

## Check what it is actually listening on

```bash
python3 tools/dev-livekit/check_listeners.py
```

The spike found ICE/TCP bound to **every interface** on port 7881 even though
signaling was bound to loopback. docs/07 requires verifying listeners rather
than inferring exposure from the WebSocket URL, so this makes that check
executable. It exits non-zero on any non-loopback exposure; pass `--allow-lan`
if you are deliberately serving a trusted LAN and your firewall says so.

## WSL2 networking

Only relevant if the server runs in WSL2 and the browser runs on Windows —
which is the normal setup for the
[browser publisher](../../services/media-gateway/README.md), since that is
where the camera is.

Default WSL2 NAT forwards `localhost` for **TCP only** and cannot support this
dual-peer path. The proven setup is Windows 11 mirrored networking plus native
LiveKit:

```ini
[wsl2]
networkingMode=mirrored
hostAddressLoopback=true
```

In mirrored Wi-Fi mode, `livekit.dev.yaml` deliberately leaves `node_ip` unset,
enables loopback candidates, and sets `force_tcp: false`. The glasses select
the Wi-Fi UDP candidate while Windows Chrome selects loopback UDP. Pinning the
Wi-Fi address fixes the glasses but breaks Chrome; forcing TCP also breaks the
Chrome viewer. Apply `scripts/wsl_lan_expose.ps1` as Administrator and never use
a Windows portproxy.

See `.agents/skills/glasses-stack-debug/PLATFORMS.md` for the proven WSL reboot
checklist and the separate Mac/GN100 status.

## TLS

**You almost certainly do not need it.** `getUserMedia` requires a secure
context, and `http://localhost` already counts as one — so the browser
publisher works over plain HTTP on the machine running the gateway.

TLS is only needed when a **second device** (a phone, a teammate's laptop)
reaches this workstation over the LAN. In that case use
[mkcert](https://github.com/FiloSottile/mkcert):

```bash
mkcert -install
mkcert your-workstation.local 192.168.1.42
```

Terminate TLS at LiveKit's native TLS config or a small reverse proxy in front.
Zero external traffic, a real padlock, and the privacy assertion below stays
green.

### Do not use a tunnel for media or signaling

Cloudflare Tunnel, ngrok, and friends are **rejected** here, and the rejection
is enforced by a test:
`tests/integration/test_livekit_roundtrip.py::test_the_gateway_talks_to_nothing_off_this_machine`
sweeps for established sockets to non-local peers and fails if it finds any.

Three reasons, in the order that matters:

1. **"Just tunnel the page" is not achievable.** An HTTPS page cannot open a
   `ws://` WebSocket, so tunneling the console forces you to tunnel LiveKit
   signaling too. Tunnels proxy TCP only, so ICE then falls back to TCP and
   **the raw camera video itself traverses the third party** — not merely the
   HTML.
2. **docs/07 names it.** Cloud signaling and TURN relay sit outside the trust
   boundary and must be disabled by default and disclosed when enabled. Raw
   first-person video through a third-party edge is the precise failure this
   product promises not to have.
3. The privacy test goes red, correctly.

Tailscale is the reasonable choice if devs are genuinely on different networks —
a WireGuard mesh, usually peer-to-peer — but it is still an external
coordination dependency and needs the same disclosure. Prefer mkcert on one LAN.

The one legitimate tunnel use is sharing a **read-only status dashboard**: no
media, no evidence, no transcripts. Even then it must be opt-in, off by default,
disclosed, and never the same process that terminates media.

## Related

- [`compose.dev.yaml`](../../compose.dev.yaml) — Path A
- [Media Gateway README](../../services/media-gateway/README.md) — running the
  gateway against this server, and the browser publisher
- [Privacy and Security](../../docs/07-Privacy-and-Security.md) — port policy
  and the trust boundary
- [S01 spike results](../../docs/spikes/livekit-media-gateway/RESULTS.md) —
  where the listener finding and the pinned version come from
