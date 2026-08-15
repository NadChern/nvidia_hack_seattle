# RayNeo X3 Pro end-to-end testing on macOS

This is the repeatable local-development path for an Apple-silicon Mac and a
physical RayNeo X3 Pro. It covers repository checks, Android build and install,
the LAN-visible stack, QR pairing, real Apple-silicon speech, YOLOE detection,
and YOLO26 metric depth.

The path was exercised on 2026-08-14 with an ARGF20 / `MercuryLiteXR`. It is
Mac development evidence only. It does not replace the physical GN100 release
gate in [`docs/08-Development-and-Deployment.md`](../../../docs/08-Development-and-Deployment.md).

## What runs where

| Component | Mac path |
|---|---|
| Console and ordinary services | Native processes started by `scripts/dev_stack.sh` |
| LiveKit | Pinned v1.13.4 Linux ARM64 container in Docker Desktop |
| Speech | Parakeet and Kokoro through the MLX dependency group |
| Detection | YOLOE on Metal (`mps`) |
| Depth | YOLO26 metric depth on Metal (`mps`), explicitly enabled |
| Android client | Debug APK installed directly over USB with ADB |

`start_wifi_stack.sh`, `configure_laptop_address.sh`, and the current
`stack_doctor.sh` are WSL/Linux operator tools. Do not use them as the Mac
startup path.

## 1. Prerequisites

Use an Apple-silicon Mac and install:

- Node.js 22 (the version physically tested; Node 20.19 or newer is required by
  the current Vite toolchain);
- JDK 17;
- Android SDK platform 35 and platform tools;
- Docker Desktop with Docker Compose v2;
- `uv` and Python 3.

If Node is managed by `nvm`:

```bash
nvm install 22
nvm use 22
nvm alias default 22
node --version
npm --version
```

The `nvm install` command is harmless when Node 22 is already present: it
selects the installed version instead of downloading it again.

One Homebrew-based JDK and Android command-line-tools setup is:

```bash
brew install openjdk@17 android-commandlinetools

cd "$(git rev-parse --show-toplevel)"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export ANDROID_HOME="$PWD/.tools/android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$ANDROID_HOME/platform-tools:$PATH"

mkdir -p "$ANDROID_HOME"
yes | sdkmanager --sdk_root="$ANDROID_HOME" --licenses
sdkmanager --sdk_root="$ANDROID_HOME" \
  "platform-tools" \
  "platforms;android-35" \
  "build-tools;35.0.0"
```

The repository ignores `.tools/`; a fresh clone does not contain the JDK or
Android SDK. If Android Studio already manages a compatible SDK, point
`ANDROID_HOME` at that SDK instead.

Verify the tools before building:

```bash
java -version
docker info
uv --version
adb version
```

## 2. Run the automated review checks

Stop a running stack with Ctrl-C and use a fresh terminal. `dev_stack.sh`
honours exported `VMA_*` values, and those live-test overrides can change test
expectations. In particular, do not run Agent tests with
`VMA_AGENT_BACKEND=stub` still exported.

From the repository root:

```bash
python3 .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
bash -n scripts/dev_stack.sh .agents/skills/glasses-stack-debug/scripts/*.sh
git diff --check origin/main...HEAD
```

For each Python service, run the repository-standard checks from that service's
directory:

```bash
set -euo pipefail
for service in services/*; do
  [[ -f "$service/pyproject.toml" ]] || continue
  (
    cd "$service"
    uv sync --frozen --all-groups
    uv run ruff format --check .
    uv run ruff check .
    uv run pyright
    uv run pytest
  )
done
```

Two Mac review findings affect that literal loop on the current branch:

1. If Vision reports that `vision_worker`, `visual_memory_vision_contract`, or
   `visual_memory_media_contract` cannot be imported after an apparently
   successful sync, repair the stale editable environment and repeat its
   checks:

   ```bash
   cd services/vision-worker
   uv sync --frozen --all-groups --reinstall
   uv run python -c "import vision_worker, visual_memory_vision_contract, visual_memory_media_contract; print('Vision imports OK')"
   uv run pyright
   uv run pytest
   ```

2. Installing all Speech groups enables optional real-MLX tests on a Mac. On
   the 2026-08-14 review run,
   `test_api_stt_parakeet.py::test_stt_streams_real_transcripts_via_parakeet`
   waited indefinitely for a second WebSocket message and required Ctrl-C. Run
   the bounded base suite separately while that test remains unresolved:

   ```bash
   cd services/speech
   uv run pytest \
     --ignore=tests/test_api_stt_parakeet.py \
     --ignore=tests/test_api_synthesize_kokoro.py \
     --ignore=tests/test_kokoro_backend.py \
     --ignore=tests/test_parakeet_backend.py
   ```

Record the optional-test hang as a review finding; the base-suite workaround is
not evidence that the skipped real-model test passed.

Run the Console checks:

```bash
cd apps/console
npm ci --no-fund --no-audit
npm run test
npm run build
npm run lint
```

Run Android checks with Java 17 and SDK 35:

```bash
cd apps/glasses-x3
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The APK is
`apps/glasses-x3/app/build/outputs/apk/debug/app-debug.apk`.

## 3. Connect and install the glasses client

Connect the glasses by USB, wake them, and confirm that ADB reports state
`device` rather than `unauthorized` or `offline`:

```bash
cd "$(git rev-parse --show-toplevel)"
export ANDROID_HOME="$PWD/.tools/android-sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"

adb kill-server
adb start-server
adb devices -l
```

Accept **Allow USB debugging** on the glasses if prompted. A healthy entry looks
like:

```text
<serial>  device  product:RayNeoX3Pro model:ARGF20 device:MercuryLiteXR
```

Install the APK:

```bash
adb install -r apps/glasses-x3/app/build/outputs/apk/debug/app-debug.apk
```

If installation fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, the installed
copy was signed with another developer's debug key. Uninstall it, then install
this build:

```bash
adb uninstall com.visualmemory.glasses
adb install apps/glasses-x3/app/build/outputs/apk/debug/app-debug.apk
```

Uninstalling clears the app's saved pairing credential and stable local device
identity. Use this recovery only for a signature mismatch; it is not part of a
normal update.

Launch the client:

```bash
adb shell am start -n com.visualmemory.glasses/.MainActivity
```

USB is needed for installation and debugging. The live camera, microphone, and
pairing traffic use Wi-Fi, so the Mac and glasses must also be on the same
trusted LAN with peer-to-peer traffic allowed.

## 4. Configure the Mac's LAN addresses

Find the active interface and address. On a Wi-Fi Mac the interface is usually
`en0`:

```bash
MAC_INTERFACE="$(route get default | awk '/interface:/{print $2}')"
MAC_LAN_IP="$(ipconfig getifaddr "$MAC_INTERFACE")"
echo "Interface: $MAC_INTERFACE"
echo "Mac LAN IP: $MAC_LAN_IP"
```

Run the stack once if `.env` does not exist; the launcher creates a private,
gitignored LiveKit key and secret. Stop it with Ctrl-C after the credentials are
generated:

```bash
./scripts/dev_stack.sh --fixture
```

Keep the generated key and secret in `.env` and add these addresses:

```dotenv
VMA_LIVEKIT_URL=ws://127.0.0.1:7880
VMA_LIVEKIT_PUBLIC_URL=ws://<mac-lan-ip>:7880
```

Create or update the gitignored `apps/console/.env.local`:

```dotenv
VITE_VMA_GATEWAY_PUBLIC_URL=http://<mac-lan-ip>:8080
VITE_VMA_LIVEKIT_URL=ws://127.0.0.1:7880
```

Replace `<mac-lan-ip>` with the address printed above. The public Gateway and
LiveKit URLs go into the QR and must be reachable from the glasses. The Console
browser is on the Mac itself, so its LiveKit override remains loopback.

Recheck these values whenever the Mac joins a different network or receives a
new DHCP address. A changed Gateway address requires a new pairing QR.

## 5. Start the real Mac stack

Do not pass `--fixture`: that option intentionally replaces model inference
with deterministic test adapters and disables real depth.

From the repository root, in a dedicated foreground terminal:

```bash
export VMA_BIND_ADDR=0.0.0.0
export VMA_AGENT_BACKEND=stub
export VMA_DEPTH_KIND=yolo
export VMA_YOLO_DEPTH_DEVICE=mps
./scripts/dev_stack.sh --allow-lan
```

Why each override is present:

- `VMA_BIND_ADDR=0.0.0.0` lets the glasses reach the services over the LAN.
- `--allow-lan` acknowledges and permits that deliberate trusted-LAN exposure.
- `VMA_AGENT_BACKEND=stub` keeps the language/reasoning layer deterministic and
  local while exercising the complete media, memory, STT, and TTS path.
- `VMA_DEPTH_KIND=yolo` selects the smaller YOLO26 metric-depth adapter. MoGe-2
  selected CPU on the tested Mac and did not become ready within the launcher's
  five-minute model-start window.
- `VMA_YOLO_DEPTH_DEVICE=mps` puts YOLO depth on Apple Metal rather than CPU.

The first real-model run downloads and warms several model artifacts and can
take minutes. Later starts reuse the local caches. Keep this terminal open;
Ctrl-C stops the supervised services and the LiveKit container it started.

## 6. Pair through the Console

1. Open <http://localhost:5173>.
2. Open **Glasses**.
3. Click **Pair**.
4. Before scanning, confirm the dialog shows
   `http://<mac-lan-ip>:8080`, not `127.0.0.1`.
5. Put on or point the glasses at the displayed QR.
6. Swipe the temple touchpad to focus **Scan QR** and single-tap if needed.

Every new app process opens target selection. To reuse a valid saved pairing,
swipe to **Reconnect saved target** and single-tap. A failed or expired new QR
does not erase the previously saved target.

## 7. Verify the physical pipeline

In the Console, require all of the following:

- the glasses session is connected and publishing;
- live glasses video is visible;
- the Boxes view shows detected-object rectangles and labels;
- depth values appear on detected objects;
- the Speech status reports real Parakeet STT and real Kokoro TTS, not stubs;
- the Vision and Speech services remain ready after model warm-up.

Inspect Vision directly:

```bash
curl -fsS http://127.0.0.1:8082/v1/status | python3 -m json.tool
```

Expected model fields include:

```text
detector_kind: yoloe
depth_kind: yolo
detector.ready: true
detector.device: mps
depth.ready: true
depth.device: mps
```

After the video has run, require increasing detector and depth
`request_count`, `frames_processed` greater than zero, and at least one overlay
viewer while the Boxes view is open.

Test a complete spoken request:

1. During the live session, single-tap the glasses temple.
2. Wait for the green **Ask now** HUD state.
3. Say: “Where did I leave my keys?”
4. Confirm that the HUD shows a truthful reply and that the glasses play the
   synthesized response.

An object without a confirmed placement may correctly return `unknown · passed`.
That is safer than inventing a location.

Also exercise reconnect and leave the session active beyond the former
30-second display-sleep boundary. A complete release acceptance still requires
the longer soak and GN100 gates described in the platform matrix.

## 8. Interpret Mac performance

The 2026-08-14 physical run configured Vision for 8 FPS but observed about
2.62 processed FPS with YOLOE and YOLO depth sharing Apple Metal. The pipeline
dropped 131 stale frames instead of building an increasingly delayed queue.

This means the video can remain current while boxes, labels, depth, and
placement state update only two or three times per second. Brief object motion
can be missed, overlays can look jumpy, and frame-count-based confirmation can
take longer. For example, a 30-frame confirmation target is about 3.75 seconds
at 8 FPS but about 11.5 seconds at 2.62 FPS.

Treat this as degraded Mac development throughput, not a pipeline failure and
not a GN100 performance prediction. Record `observed_fps`, model latency,
stale-frame drops, and verification failures in the PR review.

## 9. Troubleshooting

| Symptom | Meaning and next action |
|---|---|
| `adb devices -l` is empty | Reconnect USB, wake the glasses, restart ADB, and accept USB debugging on the HUD. |
| APK signatures do not match | Use the explicit uninstall/reinstall recovery above; saved app data will be lost. |
| QR contains `127.0.0.1` | Fix `VITE_VMA_GATEWAY_PUBLIC_URL`, restart the Console, and generate a new code. |
| Pairing works but no media appears | Confirm Docker maps `7882/udp`, both devices share a peer-capable trusted LAN, and current LiveKit activity includes an Android publisher and JS viewer. |
| Console reports `Failed to fetch` | Confirm the stack is still running and ports 5173/8080 are reachable. Check `logs/console.log` and `logs/gateway.log`. |
| Vision reports `depth_kind: none` | Start without `--fixture` and export both YOLO depth variables before launching. |
| Depth loads on CPU | Export `VMA_YOLO_DEPTH_DEVICE=mps` and restart the complete stack. |
| Vision misses its readiness deadline with MoGe | Stop with Ctrl-C and use the measured YOLO/MPS depth profile. |
| Speech reports stubs or silence | Stop, allow the launcher to sync the MLX group, and inspect `logs/sync.log` and `logs/speech.log`. |
| Boxes lag behind video | Check `observed_fps` and stale drops; this was measured with both models sharing Metal. |

Do not stop individual service processes by port. Stop the foreground
`dev_stack.sh` supervisor with Ctrl-C so it can shut down the complete stack in
dependency order.
