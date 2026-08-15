# Glasses stack platform and reboot matrix

This file prevents a successful WSL session from being mistaken for proof of
Mac or GN100 compatibility. A platform is **proven** only when its own physical
acceptance checks pass.

## Current status

| Platform | Status | What is established | What is still required |
|---|---|---|---|
| WSL2 mirrored + Windows Chrome + RTX 4070 Laptop | **Proven locally, including cold reboot** (2026-08-13) | After a Windows restart, persistent preflight passed; native LiveKit started cleanly; the supervised stack reached READY; glasses publish over Wi-Fi UDP; Chrome views over loopback UDP; YOLOE + YOLO metric depth and CUDA Speech start together | Commit the skill and configuration so the recovery path is preserved |
| Apple-silicon Mac | **End-to-end physically validated with limitations** (2026-08-14) | Direct ADB build/install/launch; QR pairing; pinned Docker LiveKit carrying physical glasses video to the JS viewer; YOLOE + YOLO26 depth on MPS; MLX Parakeet/Kokoro; HUD and audible answer | Fix the hanging optional Parakeet WebSocket test; measure/tune the 2.62 observed Vision FPS; complete reconnect and long soak evidence; Mac results do not close GN100 gates |
| Acer GN100 / GB10 Linux ARM64 CUDA | **Not release-ready; physical gates pending** | Production Compose exists; LiveKit image is multi-arch pinned; services own ARM64-capable base images and locks; contracts and health surfaces exist | S01–S05 physical gates; actual glasses media and reconnect; ARM64 codec/decode; CUDA model startup; complete-workload memory/latency; real Parakeet/Kokoro adapter; rollback and soak |

Never claim GN100 readiness from an x86 build, Compose rendering, QEMU/buildx,
or a successful Mac/WSL run. `docs/08-Development-and-Deployment.md` makes the
physical GN100 the final release gate.

## USB/ADB after every Windows reboot

`usbipd bind` persists, but WSL attachment does not. Even if Administrator
PowerShell reports the RayNeo as `Shared`, reattach it:

```powershell
usbipd list
usbipd attach --wsl --busid <RayNeo BUSID>
```

Then run `adb kill-server`, `adb start-server`, and `adb devices -l` in WSL;
require state `device`. The current workstation observed BUSID `2-1`, but USB
positions can change. See [`SKILL.md`](SKILL.md) for `Device busy` recovery and
the reliable non-streaming APK install command.

## Venue network: do not trust the hackathon Wi-Fi

Different DHCP addresses are expected and supported; **client isolation is the
real demo killer**. Many event networks block peer-to-peer traffic even when
all devices show the same SSID. Raw TCP and UDP will never reach the laptop or
GN100 in that topology, and no WSL setting can fix the router.

Preferred physical topology:

```text
travel router (private trusted LAN; client isolation off)
  ├── laptop (Ethernet preferred; DHCP reservation)
  ├── GN100 (Ethernet; DHCP reservation)
  └── RayNeo glasses (Wi-Fi)
```

Give laptop and GN100 separate fixed DHCP leases. Record both addresses on a
card before the demo. The router may use the venue network only as its uplink;
all camera/media traffic remains on the private LAN. Test glasses → laptop and
glasses → GN100 separately before opening the room to viewers.

When the laptop address changes, with the stack stopped:

```bash
.agents/skills/glasses-stack-debug/scripts/configure_laptop_address.sh
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh --check
```

The first command detects the current IPv4 and updates only public URLs; it
keeps Chrome's LiveKit route on loopback. A changed Gateway address requires a
fresh device pairing.

## Switching glasses between laptop and GN100

Pairing stores one `{gateway_url, credential}` at a time, and each credential
is signed by that target's internal token. Therefore a target switch really is
a new pairing; changing only LiveKit URLs is insufficient.

Fast switch with one ADB-attached device, preserving its stable device ID:

```bash
# To laptop (uses local .env token and address)
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh laptop

# To GN100 (do not put its token in argv or source control)
export VMA_TARGET_INTERNAL_API_TOKEN_FILE=$HOME/.config/vma/gn100-token
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh http://<gn100-ip>:8080
```

The helper asks the selected Gateway for a single-use code, force-stops the old
session, and sends the app's supported `pairing_payload` intent extra. It does
**not** run `pm clear`, so the device ID survives. Ensure the GN100
`VMA_DEVICE_ID_ALLOWLIST` includes that device ID.

Normal switching no longer requires ADB: restart the app, which opens target
selection on every new process, generate a new code on the selected target
console, and scan it. The old pairing remains stored until the new claim
succeeds. Swipe the RayNeo temple touchpad to focus **Reconnect saved target**
and single-tap to activate it. Use the ADB helper if touch/QR is unavailable.

The GN100 Compose topology does not bundle the React console. Run a second
laptop-hosted operator console (default port 5174) against the target:

```bash
export VMA_TARGET_INTERNAL_API_TOKEN_FILE=$HOME/.config/vma/gn100-token
.agents/skills/glasses-stack-debug/scripts/start_target_console.sh <gn100-ip>
```

This leaves the normal laptop console/profile on port 5173 and proxies target
Gateway, Vision overlay, Memory, Speech, and Agent APIs without rewriting the
laptop `.env.local`.

## Known platform differences

### WSL2 mirrored Wi-Fi — proven path

Use:

```bash
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh
```

LiveKit must be native, `node_ip` must be unset, loopback candidates enabled,
and TCP forcing disabled. The selected ICE pairs are different by design:

- glasses → laptop Wi-Fi address on UDP 7882;
- Windows Chrome → 127.0.0.1 on UDP 7882.

### Apple-silicon Mac — separate local-development path

Do **not** run `start_wifi_stack.sh`; it deliberately checks WSL and Windows
Firewall. Start with the repository's Mac-aware `scripts/dev_stack.sh` and the
step-by-step [`MACOS.md`](MACOS.md) runbook. The pinned Docker LiveKit path was
physically validated with an ARGF20 on 2026-08-14.

Validated path and limitations:

1. LiveKit 1.13.4 publishes no official Darwin server binary. The pinned Linux
   ARM64 container carried real glasses media on Docker Desktop; Homebrew
   remains an unpinned alternative.
2. `dev_stack.sh` selects MLX speech on Apple silicon and YOLOE using macOS
   arm64 Torch wheels. Physical STT, TTS, HUD, and return audio worked.
3. Depth remains off by default because detector and depth share the Metal
   queue. The working opt-in is `VMA_DEPTH_KIND=yolo` together with
   `VMA_YOLO_DEPTH_DEVICE=mps`. MoGe selected CPU and did not become ready
   before the launcher's five-minute readiness deadline.
4. With YOLOE and YOLO depth together, Vision was configured for 8 FPS and
   observed 2.62 FPS. It dropped 131 stale frames to remain current. This can
   delay overlay and frame-count-based placement confirmation.
5. The ordinary Speech suite passed, but the optional real-MLX
   `test_api_stt_parakeet.py` WebSocket test waited indefinitely for its second
   message. That automated test remains a Mac review finding despite the
   physical speech path working.
6. The physical run established pairing, Android publishing, JS viewing,
   detection, depth, a HUD reply, and audible speech. Reconnect and long-duration
   soak evidence remain open, and none of this substitutes for GN100 validation.

### GN100 — release deployment, not a laptop startup variant

Use `compose.yaml` (and the selected, physically validated GPU overrides), not
`dev_stack.sh` or the WSL wrapper. Set the GN100 trusted-LAN address in
`VMA_BIND_ADDR` and supply all required secrets and the device allowlist.

Current blockers that must remain visible:

- `services/speech/Dockerfile` currently runs stub STT/TTS on Linux ARM64; real
  GN100 Parakeet/Kokoro is still spike S03.
- Real GPU Vision uses `compose.gpu.yaml`/`Dockerfile.cuda`; its ARM64/CUDA
  startup, models, codec path, and coexistence are physical S02/S04 gates.
- `deploy/livekit.yaml` and the Compose network must pass actual off-box
  glasses ICE. A rendered Compose file or open TCP port does not prove media.
- The real glasses must pass camera, microphone, HUD, return audio, software
  AEC, reconnect, and 30-minute stability.
- Rollback, persistence, retention, privacy exposure, and unified-memory
  headroom remain release requirements.

## Recorded WSL cold-reboot result

**PASS — 2026-08-13.** After restarting Windows, the unmodified commands

```bash
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh --check
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh
```

validated Wi-Fi address `10.0.0.4`, mirrored networking, dual Wi-Fi/loopback
UDP candidates, absence of portproxy rules, and the TCP/UDP firewall policy.
The wrapper started native LiveKit and the supervised stack reached `READY`
with constrained CUDA YOLOE, YOLO metric depth, and CUDA STT/TTS. A changed
temporary IPv6 address did not affect the selected IPv4/loopback topology.
This closes the laptop reboot-persistence check; it does not close the separate
Mac or physical GN100 gates above.

## WSL reboot acceptance checklist

After restarting Windows:

1. Open WSL and enter the repository.
2. Validate persistent configuration:

   ```bash
   .agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh --check
   ```

   If DHCP changed the Wi-Fi address, the command reports old and new values.
   Update `.env`, `apps/console/.env.local`, and re-pair the glasses.
3. Start the stack in one foreground terminal:

   ```bash
   .agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh
   ```

4. Open `http://localhost:5173` in Windows Chrome.
5. If the glasses app did not resume, wake/start it without clearing pairing:

   ```bash
   adb shell input keyevent KEYCODE_WAKEUP
   adb shell input keyevent KEYCODE_MENU
   adb shell am start -n com.visualmemory.glasses/.MainActivity
   ```

   USB/ADB attachment is only needed for this control/debug step; Wi-Fi media
   itself does not depend on ADB.
6. Require:

   ```bash
   .agents/skills/glasses-stack-debug/scripts/stack_doctor.sh
   ```

   Expected: zero failures, one publisher, native LiveKit, firewall TCP
   8080/7880/7881, and UDP 7882.
7. In the UI, verify live video and the depth badge. In LiveKit logs, verify
   current `participant active` entries for both Android and JS.
8. Verify Vision:

   ```bash
   set -a; . ./.env; set +a
   curl -fsS -H "Authorization: Bearer $VMA_INTERNAL_API_TOKEN" \
     http://127.0.0.1:8082/v1/status | python3 -m json.tool
   ```

   Expected on this constrained laptop profile: `detector_kind: yoloe`,
   `depth_kind: yolo`, both models ready, and overlay viewers at least one when
   the Boxes view is open.
9. Stop with Ctrl-C in the startup terminal. Do not kill services by port.
