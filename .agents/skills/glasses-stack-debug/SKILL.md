---
name: glasses-stack-debug
description: Bring up, verify, and debug the RayNeo X3 Pro glasses against the local backend. Use for pairing, Windows-reboot USB/usbipd attachment, missing ADB or glasses sessions, session-without-video or depth, viewer disconnected (0), could not establish pc connection, LiveKit JOIN_TIMEOUT, 429 capacity_exhausted, Failed to fetch, services that will not stay up, WSL mirrored networking, Windows Firewall, adb reverse, portproxy, or dev_stack failures.
---

# Glasses stack: bring-up and debugging

This stack has seven processes, two possible network topologies, a device that
persists what it paired with, and a launcher whose failure reports are not
always true. Debugging it by intuition costs hours and breaks working parts.
Follow the order below. Platform support and the after-reboot acceptance steps
are recorded in [`PLATFORMS.md`](PLATFORMS.md); WSL success is not Mac or GN100
release evidence. For venue operations and laptop ↔ GN100 pairing, use the
printable [`DEMO_SWITCH.md`](DEMO_SWITCH.md) rather than reconstructing commands.

## Rule 1: diagnose before touching anything

```bash
.agents/skills/glasses-stack-debug/scripts/stack_doctor.sh
```

Read-only. Prints PASS/FAIL for every link in the chain and, for each failure,
the exact command that fixes it. Exit code is the failure count.

**Run it before and after every change.** Most of the lost time in this project
came from acting on a guess and then not knowing whether the change helped.

## Rule 2: never kill processes by port to "clean up"

`dev_stack.sh` supervises children and stops all of them together when one
dies. Killing a process it tracks makes it tear down every healthy sibling —
which is what repeatedly killed the console and produced `Failed to fetch`.

To stop the stack: **Ctrl-C in its own terminal**, or `kill` the one
`bash ./scripts/dev_stack.sh` supervisor pid and let it run its own cleanup.
Only reach for per-port kills when the doctor reports an orphan, and then kill
the orphan alone.

## The two topologies

Almost every confusing failure is a half-configured mix of these. Pick one and
make every row true.

| | USB (tethered) | Wi-Fi (untethered, what a demo needs) |
|---|---|---|
| Device reaches host at | `127.0.0.1` via `adb reverse` | the laptop's Wi-Fi IP |
| `node_ip` in `livekit.dev.yaml` | `127.0.0.1` | unset, so glasses get Wi-Fi and Windows Chrome gets loopback |
| LiveKit runtime | native binary (never Docker) | native binary (never Docker) |
| `VMA_LIVEKIT_PUBLIC_URL` in `.env` | `ws://127.0.0.1:7880` | `ws://<wifi-ip>:7880` |
| `VITE_VMA_GATEWAY_PUBLIC_URL` | `http://127.0.0.1:8080` | `http://<wifi-ip>:8080` |
| Services bind | loopback (default) | `VMA_BIND_ADDR=0.0.0.0` |
| Windows side | nothing | nothing, with mirrored networking |
| Survives | nothing (adb restart wipes tunnels) | everything |

With WSL2 **mirrored networking**, WSL owns the laptop's Wi-Fi address directly.
Wi-Fi mode needs LAN binding and the Windows firewall rule, but **never a
portproxy**. A portproxy is both unnecessary and harmful to WebRTC ICE.

## Known-good Wi-Fi start: use this, not seven hand-written commands

From WSL, in a dedicated foreground terminal:

```bash
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh
```

This preflights mirrored networking, both browser/device ICE routes, console
URLs, stale portproxies, and exact firewall protocols. It starts **native**
LiveKit first, then `dev_stack` with `VMA_BIND_ADDR=0.0.0.0` and the constrained
YOLOE + metric-depth profile. Keep the terminal open; Ctrl-C shuts it down.
Use `--check` for read-only persistent-configuration validation and pass
`--no-sync` only after dependencies are already synchronized.

Manual bring-up order, if the wrapper itself is being debugged:

1. Run `stack_doctor.sh`.
2. Start **native LiveKit first** using the command in the networking section.
3. In a second foreground terminal, start the services:
   `VMA_BIND_ADDR=0.0.0.0 VMA_ENABLE_CONSTRAINED_VISION=true ./scripts/dev_stack.sh`.
   Because 7880 already answers, `dev_stack` reuses native LiveKit instead of
   starting the Docker service.
4. Run `stack_doctor.sh` again; require zero failures before touching pairing.
5. Wake and start the existing paired app. Clear app data only when its saved
   pairing address is actually wrong.

## Venue address changes and laptop ↔ GN100 switching

Read [`PLATFORMS.md`](PLATFORMS.md) before the event. The short version:

```bash
# Stack stopped: update laptop public URLs after joining a different LAN
.agents/skills/glasses-stack-debug/scripts/configure_laptop_address.sh

# Re-pair attached glasses to laptop without clearing their stable device ID
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh laptop

# Re-pair to target; token comes from an environment variable or mode-600 file
VMA_TARGET_INTERNAL_API_TOKEN_FILE=~/.config/vma/gn100-token \
  .agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh \
  http://<gn100-ip>:8080
```

Use a private travel router with client isolation disabled and DHCP reservations
for laptop and GN100. Venue Wi-Fi commonly blocks peers on the same SSID; no
firewall, WSL, LiveKit, or portproxy change can overcome AP client isolation.

## Restore USB/ADB after a Windows reboot

`usbipd bind` persists, but attachment to WSL does not. A device shown as
`Shared` is unavailable to Linux until attached again. In **Administrator
PowerShell**:

```powershell
usbipd list
usbipd attach --wsl --busid <RayNeo BUSID>
```

If Windows ADB owns it or `usbipd` reports `Device busy`:

```powershell
Stop-Process -Name adb -Force
usbipd unbind --busid <RayNeo BUSID>
usbipd bind --force --busid <RayNeo BUSID>
usbipd attach --wsl --busid <RayNeo BUSID>
```

Then in WSL:

```bash
lsusb
adb kill-server
adb start-server
adb devices -l
```

Accept **Allow USB debugging** on the glasses if prompted. On this workstation
the RayNeo was BUSID `2-1` on 2026-08-13, but always read the current BUSID from
`usbipd list`. USB topology can change.

For APK installation, prefer push install. A streamed 33 MB install caused one
observed USB/IP detach; this completed reliably:

```bash
adb install -r --no-streaming apps/glasses-x3/app/build/outputs/apk/debug/app-debug.apk
```

## Pairing and target selection

The current app stores one gateway URL and credential, but every new app process
opens the scanner and asks the wearer to scan the laptop or GN100 QR. RayNeo's
temple touch API drives the chooser: swipe changes focus and single-tap activates
**Reconnect saved target**; QR scanning is the default. The old pairing remains
until a new claim succeeds, making invalid scans non-destructive. During a live
session, single-tap arms the consume-once manual voice trigger for 15 seconds;
wait for **Ask now**, then ask a `where…` question. `pm clear` is no longer part
of normal switching.

Before scanning, confirm the console dialog's footer shows the address for the
topology you chose. Pairing to the wrong host is the single most repeated
mistake here.

## Networking: the only configuration that carries media

Two rules, both learned the hard way. **WebRTC does not survive address
translation**, and this stack had four translating layers stacked on each
other. Remove them all:

1. **WSL2 mirrored networking.** In `%USERPROFILE%\.wslconfig`:

   ```ini
   [wsl2]
   networkingMode=mirrored
   hostAddressLoopback=true
   ```

   Apply with `wsl --shutdown` from PowerShell. WSL then owns the laptop's LAN
   address itself, so the glasses reach the stack over Wi-Fi with no forwarding
   at all. Verify with `ip -4 addr` — WSL should hold the Wi-Fi address.

2. **Run LiveKit as the native binary, never in Docker.**

   ```bash
   set -a && . ./.env && set +a
   export LIVEKIT_KEYS="$VMA_LIVEKIT_API_KEY: $VMA_LIVEKIT_API_SECRET"
   .tools/livekit-1.13.4/livekit-server --config tools/dev-livekit/livekit.dev.yaml &
   ```

   In a container, ICE never completes — even for the gateway, which runs on
   the same host. Its own error is `wait_pc_connection timed out`, while raw TCP
   to 7881 connects fine. Docker's bridge NAT is the last thing rewriting
   packets. Switching to the binary fixed it on the first attempt.

   The proven dual-peer candidate split is:

   - glasses: local `10.0.0.4:7882` ↔ device Wi-Fi, `connectionType: udp`;
   - Windows Chrome: local `127.0.0.1:7882` ↔ browser loopback,
     `connectionType: udp`.

   Seeing a session ID proves only HTTP/signalling. The video is not working
   until LiveKit logs `participant active` for both the Android publisher and
   the JS viewer.

Then open the firewall (Wi-Fi is usually a Public network, where Windows blocks
inbound by default):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\wsl_lan_expose.ps1
```

Both public URLs in `.env` and `VITE_VMA_GATEWAY_PUBLIC_URL` must use the
laptop's Wi-Fi address. Leave LiveKit `node_ip` unset in mirrored Wi-Fi mode and
set `enable_loopback_candidate: true` plus `force_tcp: false`: LiveKit then
offers Wi-Fi UDP to the glasses and loopback UDP to the Windows Chrome viewer.
Pinning `node_ip` to Wi-Fi restores the glasses but causes the console viewer to
fail with `could not establish pc connection`.

**What does not work, measured, so nobody repeats it:**

- `netsh portproxy` is a userland TCP proxy that re-originates connections, so
  LiveKit sees every peer arriving from the host address and ICE candidate
  pairs never validate. Raw TCP connects; the join fails as `JOIN_TIMEOUT`.
- Without mirrored networking, WSL2 cannot send UDP to the Windows host at all
  (five probes, zero received), so moving LiveKit to Windows does not help
  either — the glasses work and the gateway loses its media path.
- **Leftover portproxy rules are actively harmful once mirrored**, because WSL
  and Windows share one port space: an old rule squats on the port its Linux
  service wants, and the launcher reports "port 8081 is already in use" with
  nothing visible in `ss`. Check `netsh interface portproxy show all`.

USB (`adb reverse`) still works and needs `node_ip: 127.0.0.1`, but tethers the
glasses and dies on every adb or usbipd restart.

## Reading the device

The X3 Pro shows app errors on the HUD, and **screenshots are the fastest
client-side debugging tool** — the HUD renders the exact exception:

```bash
adb exec-out screencap -p > /tmp/hud.png
```

Logcat works too; filter by the app's pid, since the vendor launcher is noisy:

```bash
adb logcat -d | awk -v p="$(adb shell pidof com.visualmemory.glasses | tr -d '\r')" '$3==p'
```

The app's connect order matters when reading a hang: session create → device
events socket → foreground service → **then** LiveKit. A hang at "Connecting
camera and microphone…" with no LiveKit participant means it failed before
`room.connect`, not in it.

## Failure catalogue

Each of these was observed and diagnosed on real hardware.

| Symptom | Cause | Fix |
|---|---|---|
| HUD: `Failed to connect to /127.0.0.1:8080` | Device paired to loopback, or `adb reverse` tunnels wiped by an adb/usbipd restart | Re-create tunnels, or re-pair to the Wi-Fi address |
| Console: `Failed to fetch` on Pair | Vite is down, or started without `VITE_VMA_INTERNAL_API_TOKEN` | Restart the console; the token lives in gitignored `apps/console/.env.local` |
| Console: `gateway 401 missing bearer token` | Same as above — console has no operator token | As above |
| HUD: `429 capacity_exhausted` | Failed joins mint a new session each retry and fill both slots | Console → Glasses → **Clear stale**, or wait 90 s (`unclaimed_session_ttl_s`) |
| LiveKit `JOIN_TIMEOUT` / `removing participant without connection` | ICE candidates do not include an address that peer can reach | In mirrored Wi-Fi mode leave `node_ip` unset, enable loopback candidates, and do not force TCP |
| Console repeats `viewer disconnected (0)` / `could not establish pc connection` while glasses are live | `node_ip` is pinned to Wi-Fi or `force_tcp: true`; signalling works but Chrome has no usable media pair | Leave `node_ip` unset, use `enable_loopback_candidate: true` and `force_tcp: false`, open UDP 7882, restart native LiveKit |
| Session appears but there is no depth badge/boxes | 8 GB resource-safe mode selected fixture Vision (`depth_kind: none`) | Start with `VMA_ENABLE_CONSTRAINED_VISION=true`; verify `/v1/status` says `detector_kind: yoloe`, `depth_kind: yolo` |
| Gateway: `wait_pc_connection timed out`, `could not join the livekit room` | **LiveKit is running in Docker** — its bridge NAT breaks ICE even for same-host clients | Run the native binary instead; see the networking section |
| HUD hangs on "Connecting camera and microphone…" with no LiveKit participant | Failed *before* `room.connect` — session create, device-events socket, or foreground service | Screenshot the HUD for the exception; check `device event subscriber connected` in the gateway log |
| Glasses hear you, never answer | Rising Agent `hands_free_ignored` means STT did not preserve a supported wake + `where…` question; zero replies means no return audio was generated | Single-tap during LIVE, wait for **Ask now**, then ask a `where…` question; use the return-audio tone endpoint to isolate speaker playback |
| Wake word never fires | STT misheard the prefix; check the HUD transcript | Add the observed spelling to `wake_prefix_variants` |
| Transcript cut mid-sentence | VAD window too short, or onset trimmed | `VMA_STT_UTTERANCE_SILENCE_MS`, `VMA_STT_PREROLL_MS` |
| `CUDA out of memory` in speech | 20 s utterances on an 8 GB GPU shared with Vision | Lower `stt_utterance_max_seconds`; do not oversubscribe |
| `FAIL <service> exited` but the port still answers | `dev_stack` tracks the `uv run` parent; children orphan and it declares a crash | Trust the doctor over the launcher; stop the orphan, restart cleanly |

## Things that are true and easy to forget

- **A `dev_stack` FAIL is not proof a service died.** Check the port first.
- Old `removing participant without connection` lines remain in appended logs.
  Treat the doctor's signature counts as warnings; current success is a fresh
  `participant active` for both Android and JS plus `1 with a publisher`.
- **`grep -c` prints `0` on no match**, so `|| echo 0` yields two values.
- **`pkill -f <pattern>` matches the agent's own shell** and kills the session.
  Look the pid up from `ss -ltnp` and kill that.
- **Restarting the gateway clears its in-memory session registry**, so the
  device's stored session vanishes and it must create a new one.
- **`netsh portproxy` is TCP-only** and cannot forward LiveKit's UDP mux on
  7882. Moving LiveKit to Windows does *not* recover UDP either, because WSL2
  cannot send UDP to the Windows host -- see the Wi-Fi section above.
