#!/usr/bin/env bash
# SG-D: RayNeo X3 Pro preflight.
#
# Ten minutes with the glasses and a USB cable, before anyone writes Kotlin that
# assumes an answer. Needs adb only -- no Android Studio, no Gradle, no JDK.
#
#   adb devices          # confirm the glasses are listed and authorized
#   ./sg_d_device_preflight.sh > sg_d_$(date +%Y%m%d).txt
#
# Every question below can move a milestone in docs/15, and none of them can be
# answered from a datasheet.

set -uo pipefail

say() { printf '\n=== %s ===\n' "$1"; }
probe() { printf '%-34s %s\n' "$1" "$(adb shell "$2" 2>&1 | tr -d '\r' | head -3)"; }

if ! adb get-state >/dev/null 2>&1; then
  echo "no device: run 'adb devices' and authorize the prompt on the glasses" >&2
  exit 1
fi

say "identity"
probe "manufacturer"    "getprop ro.product.manufacturer"
probe "model"           "getprop ro.product.model"
probe "device"          "getprop ro.product.device"
probe "build fingerprint" "getprop ro.build.fingerprint"

say "toolchain targets (minSdk, ABI splits)"
probe "android release"  "getprop ro.build.version.release"
probe "sdk level"        "getprop ro.build.version.sdk"
probe "primary abi"      "getprop ro.product.cpu.abi"
probe "supported abis"   "getprop ro.product.cpu.abilist"

say "display (HUD safe area, monocular vs binocular)"
adb shell wm size 2>&1 | tr -d '\r'
adb shell wm density 2>&1 | tr -d '\r'
probe "displays" "dumpsys display | grep -c 'Display Id'"

say "cameras (does CameraX see a world-facing camera normally?)"
adb shell "dumpsys media.camera 2>/dev/null | grep -iE 'Camera [0-9]+ information|Facing|Available functions' | head -20" 2>&1 | tr -d '\r'
probe "camera ids" "cmd media.camera get-camera-ids 2>/dev/null || echo unsupported"

say "audio routing (speaker for TTS, mic for wake word)"
probe "input devices"  "dumpsys audio | grep -iA2 'input devices' | head -6"
probe "aec available"  "dumpsys audio | grep -ci 'acoustic_echo_canceler'"

say "packages that suggest a vendor SDK or launcher we must coexist with"
adb shell "pm list packages | grep -iE 'rayneo|tcl|ar|glass' | head -20" 2>&1 | tr -d '\r'

say "permissions model"
probe "camera perm enforced" "pm list permissions -g -d 2>/dev/null | grep -c android.permission.CAMERA"

say "touchpad and buttons -- INTERACTIVE"
cat <<'EOF'
Leaving getevent running for 15 seconds.

  Tap the temple touchpad, then swipe forward, then swipe back.

What matters: whether events arrive as an ordinary touchscreen device (so plain
Compose gesture handling works) or as a vendor HID device with custom codes (so
the trigger needs a vendor path).
EOF
timeout 15 adb shell getevent -lp 2>&1 | tr -d '\r' | head -40
echo "--- live events (15s) ---"
timeout 15 adb shell getevent -lt 2>&1 | tr -d '\r' | head -60

say "thermals baseline (compare against a 20-minute publish at G7)"
adb shell "dumpsys thermalservice | grep -iE 'Temperature|status' | head -10" 2>&1 | tr -d '\r'
probe "battery level" "dumpsys battery | grep level"

say "done"
echo "Record answers in docs/spikes/glasses-client/RESULTS.md under SG-D."
