# RayNeo X3 Pro glasses client

Pure-Kotlin Android client for the Visual Memory Assistant. It scans the console's
single-use QR, persists only a device-scoped Gateway credential, publishes one 720p/15 FPS
camera layer plus microphone audio to LiveKit, plays remote `assistant-tts` audio, and
renders transcript/reply events in a flat Compose HUD. On every new app process it opens
the QR scanner so the wearer can select the laptop or GN100. Questions use the server-side
wake phrase. RayNeo temple touch events provide focus controls: swipe changes the focused
action, single-tap activates it, and QR scanning remains active while **Scan QR** is focused.
During a live session, single-tap arms a 15-second manual voice trigger; wait for **Ask now**
and ask a `where…` question without relying on wake-phrase transcription.

## Pinned toolchain

- JDK 17 and Android SDK platform 35. They are local prerequisites, not tracked
  repository contents. The gitignored `.tools/jdk17` and `.tools/android-sdk`
  paths are supported locations if a developer installs them there.
- Gradle 8.9 (wrapper, distribution checksum pinned)
- Android Gradle Plugin 8.7.2
- Kotlin 1.9.25 / Compose compiler 1.5.15
- compile/target SDK 35; min SDK 26 (device measured at API 32; 26 is the floor for
  notification channels and `startForegroundService`, which the session service needs)
- LiveKit Android SDK 2.28.0
- RayNeo Mercury Android SDK 0.2.6 (vendored AAR, SHA-256
  `5d408e2c5d80e8ae746c42abbda50012b50617005adfeb397661bec9c9be2676`)
- CameraX 1.4.2 and ML Kit barcode scanning 17.3.0

SG-D measured Android API 32 and `arm64-v8a` as the only supported ABI, so the app now
filters LiveKit native libraries to `arm64-v8a`. It also measured a 1280×480 side-by-side
display; Compose renders the same HUD into both 640×480 eye buffers.

## Build

Set `JAVA_HOME` and `ANDROID_HOME` to installations on the current machine. For
example, when the JDK and Android SDK are installed in the supported local
locations:

```bash
export JAVA_HOME=$(git rev-parse --show-toplevel)/.tools/jdk17
export ANDROID_HOME=$(git rev-parse --show-toplevel)/.tools/android-sdk
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The APK is `app/build/outputs/apk/debug/app-debug.apk`.

Apple-silicon developers should follow the complete prerequisite, build,
installation, LAN, model, and physical-acceptance instructions in the
[macOS end-to-end guide](../../.agents/skills/glasses-stack-debug/MACOS.md).

## Pair and run

1. Serve the Gateway and LiveKit on a trusted-LAN address.
2. Start the console with `VITE_VMA_GATEWAY_PUBLIC_URL=http://<lan-ip>:8080`.
3. In the console Glasses panel, choose **Pair**.
4. Launch the app and scan the QR.

Every new app process returns to target selection, even while its saved credential is
valid. Scan the laptop or GN100 QR to choose deliberately. The saved pairing is retained
until a new claim succeeds so a bad/expired QR is non-destructive. To use it, swipe the
temple touchpad to **Reconnect saved target**, then single-tap. Swipe back to **Scan QR** to
select a fresh laptop or GN100 code. Expired or malformed saved credentials are cleared and
require a QR.

The QR contains `{gateway_url, pairing_code, expires_at}`. It never contains the internal
bearer token. The claimed credential can create/refresh sessions, read only the owning
session's HUD events, and arm only that session's consume-once manual trigger.

### Switching laptop and GN100 targets

The app stores one Gateway URL and one target-signed credential at a time, so a
backend switch requires a new single-use pairing claim. The normal demo workflow is to
restart the app and scan the selected console's QR. With one ADB-attached device, the
repository helper remains available as an operator fallback:

```bash
# local development target
.agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh laptop

# GN100 target
VMA_TARGET_INTERNAL_API_TOKEN_FILE=~/.config/vma/gn100-token \
  .agents/skills/glasses-stack-debug/scripts/switch_glasses_target.sh \
  http://<gn100-ip>:8080
```

The helper force-stops the old session and sends `EXTRA_PAIRING_PAYLOAD`, which
this Activity already supports. It deliberately does not call `pm clear`, so
the stable device ID survives and can remain in both targets' allowlists.

## Media constraints

`LiveKitController` explicitly configures:

- back camera, 1280×720, 15 FPS;
- one 1.5 Mbps VP8 layer with `simulcast=false`;
- microphone AEC, noise suppression, and automatic gain control;
- automatic playback of subscribed remote audio;
- bounded capture and screen-bright wake locks for the active session, because RayNeo
  system overlays can make `FLAG_KEEP_SCREEN_ON` ineffective while the app remains live.

Do not enable simulcast. SG-C measured gateway-high/viewer-low requests failing under the
demo topology, while a single publisher layer admitted 120/120 gateway frames.

## Hardware acceptance

SG-D passed on an ARGF20. Install the latest build with:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

If this reports `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, another developer's debug
key signed the installed build. `adb uninstall com.visualmemory.glasses`
followed by a fresh install resolves the signature conflict, but uninstalling
also clears the saved pairing credential and local device identity. Do not
uninstall as part of an ordinary update.

The 2026-08-14 Apple-silicon Mac run established camera and microphone media,
HUD replies, and return-audio playback on the actual X3 Pro. The remaining
hardware gates are:

- a comfortable display safe area;
- accidental activation and ergonomics of the live-session single-tap manual trigger;
- software AEC in a real room (SG-F), beyond basic microphone/speaker routing;
- LiveKit codec, lifecycle, reconnect, and 30-minute stability (SG-E);
