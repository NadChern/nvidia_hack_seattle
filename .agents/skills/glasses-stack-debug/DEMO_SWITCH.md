# Hackathon operator card: laptop ↔ GN100

Print or keep this page open. Do not improvise network settings during the demo.

## Before leaving for the venue

- Bring a private travel router and Ethernet cables.
- Disable AP/client isolation.
- Reserve separate DHCP IPv4 addresses for laptop and GN100.
- Keep the glasses USB cable available; target switching uses one authorized ADB device.
- Put the GN100 operator token in a protected laptop file:

  ```bash
  install -d -m 700 ~/.config/vma
  install -m 600 /dev/null ~/.config/vma/gn100-token
  # Edit interactively; never commit or paste it into a command argument.
  ```

- Put the stable glasses device ID in GN100 `VMA_DEVICE_ID_ALLOWLIST`. While
  currently connected to the laptop, read it from Console → Glasses → device,
  or from authenticated `GET /v1/sessions`.

## Once connected to the private venue LAN

### Laptop

With its stack stopped:

```bash
.agents/skills/glasses-stack-debug/scripts/configure_laptop_address.sh
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh --check
.agents/skills/glasses-stack-debug/scripts/start_wifi_stack.sh
```

Normal laptop console: `http://localhost:5173`.

### GN100

Set `VMA_BIND_ADDR` to the GN100's reserved trusted-LAN IPv4—not `0.0.0.0`,
`127.0.0.1`, a Docker `172.x` address, or the laptop address. Production
Compose passes it to LiveKit as the advertised ICE IPv4 and publishes:

- TCP: 7880, 7881, 8080, 8081, 8082, 8085, 8086
- UDP: 7882

Restrict those ports to the private router subnet in the GN100 firewall. Start
the physically validated release Compose/profile and require all readiness
checks. Remember: current Linux ARM64 Speech is still stub-only until S03 is
completed on the physical target.

On the laptop, start the target operator console:

```bash
export VMA_TARGET_INTERNAL_API_TOKEN_FILE=$HOME/.config/vma/gn100-token
.agents/skills/glasses-stack-debug/scripts/start_target_console.sh <gn100-ip>
```

GN100 console: `http://localhost:5174`.

## Switch targets

Normal workflow: restart the glasses app, which always opens target selection,
and scan **New code** from either the laptop console on port 5173 or GN100
console on port 5174. The previous pairing remains stored until a new QR claim
succeeds, so a failed scan is non-destructive. To reuse it, swipe the RayNeo
temple touchpad to focus **Reconnect saved target**, then single-tap.

ADB is only the operator fallback. Attach/authorize exactly one glasses device
in WSL first (`adb devices`). The switch uses ADB only to deliver a fresh
single-use pairing; camera media remains on Wi-Fi.

### To laptop development

```bash
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh laptop --check
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh laptop
```

Watch laptop console on port 5173.

### To GN100 demo

```bash
export VMA_TARGET_INTERNAL_API_TOKEN_FILE=$HOME/.config/vma/gn100-token
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh \
  http://<gn100-ip>:8080 --check
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh \
  http://<gn100-ip>:8080
```

Watch GN100 console on port 5174.

## Acceptance after every switch

Do not accept only a session ID. Require all of:

1. target session list shows the stable device ID with publisher present;
2. LiveKit logs current `participant active` for Android publisher;
3. selected operator console adds current `participant active` for JS viewer;
4. live 720p video appears;
5. Vision status reports the intended detector/depth backends and frames rise;
6. Boxes view receives overlays; and
7. microphone, return audio, HUD, and reconnect are checked on GN100.

## If switching fails

- **No ADB device:** reconnect USB/usbipd; do not run `pm clear`.
- **Target pairing works but session is rejected:** add the unchanged device ID
  to target `VMA_DEVICE_ID_ALLOWLIST`.
- **Session exists, no video:** inspect current ICE candidates. GN100 must
  advertise its trusted-LAN IPv4; laptop dev must offer Wi-Fi plus loopback.
- **Both hosts unreachable from glasses:** suspect travel-router/client
  isolation before touching WSL or LiveKit.
- **Venue assigned new laptop IP:** stop laptop stack and rerun
  `configure_laptop_address.sh`, then switch/re-pair to laptop.
- **No ADB is possible:** clear/forget current pairing, open the desired target
  console's Pair QR, and scan. A paired app does not run its scanner.
