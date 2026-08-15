# LiveKit Media Gateway Spike

This spike exercises a real self-hosted LiveKit server as the Visual Memory Assistant's media gateway.

The glasses client is now pinned to [LiveKit Unity SDK 2.0.0](https://livekit.com/blog/unity-sdk-production-launch), which is a production release. This harness still uses simulated Python publishers and does not replace testing SDK 2.0.0 on the actual glasses.

## What it runs

```text
Simulated glasses publisher
  320×180 RGBA video at 10 FPS
  48 kHz mono microphone tone
            ↓
LiveKit Server 1.13.4
            ↓
Gateway worker
  raw video and audio subscriptions
  size-one latest-frame queue sampled at 2 FPS
  synthetic TTS tone published back
```

The simulated glasses intentionally disconnect and rejoin three times with the same participant identity. Each join publishes new camera and microphone tracks.

## Assertions

- invalid JWT signatures are rejected;
- three publish/rejoin cycles complete;
- every cycle creates new camera and microphone track SIDs;
- the worker receives raw video and audio from every cycle;
- the size-one video queue drops stale frames before inference;
- only the configured video dimensions reach the sampler;
- every simulated-glasses cycle receives return audio;
- the LiveKit server makes no established connection to a non-local peer.

## Run on Windows

From PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

`setup.ps1` creates a Python 3.11 virtual environment, installs pinned SDK versions, downloads the official LiveKit 1.13.4 Windows server, and verifies its SHA-256 checksum.

The configuration credentials are intentionally local spike credentials. Never reuse them in a deployed system.

## Run against another server binary

The Python runner accepts explicit paths:

```powershell
.\.venv\Scripts\python.exe .\run_spike.py `
  --server-bin C:\path\to\livekit-server.exe `
  --config .\livekit.yaml
```

For the GN100, create a Linux environment from `requirements.txt` and pass the Linux ARM64 LiveKit server binary. The `livekit` 1.1.13 package publishes a `manylinux_2_28_aarch64` wheel, but it still needs execution testing on the GN100.

## Interpreting video frames

The Python RTC SDK delivered normal 320×180 frames plus transient 8×8 frames during adaptation or track teardown. The worker records unexpected dimensions and rejects them before placing a frame in the inference queue. Production code should retain this dimension guard.

## Scope limits

This spike does not validate:

- LiveKit Unity SDK 2.0.0 on the actual glasses;
- a real camera, microphone, speaker, or hardware codec;
- Linux ARM64 execution on the GN100;
- transparent recovery from packet loss or a server restart;
- TLS/WSS, TURN, or firewall policy;
- sustained load or model-inference latency.

See [Results](RESULTS.md).
