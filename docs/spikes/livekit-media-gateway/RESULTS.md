# LiveKit Media Gateway Spike Results

Test date: 2026-07-29 PDT

## Decision

**Adopt the LiveKit boundary and pin LiveKit Unity SDK 2.0.0.** The live-glasses release path remains conditional on the actual-device and GN100 ARM64 gates below.

## Unity SDK follow-up

[LiveKit Unity SDK 2.0.0](https://livekit.com/blog/unity-sdk-production-launch) was released out of Developer Preview on 2026-07-20. Its documented Unity 2022.3+ and multi-platform support removes the SDK-maturity concern. It does not change what this spike measured: the publisher was a Python simulator, not the Unity SDK or the actual glasses.

## Tested environment

| Component | Version |
|---|---|
| Host | Windows x64 |
| Python | 3.11.15 |
| LiveKit Server | 1.13.4 |
| LiveKit Python RTC | 1.1.13 |
| LiveKit API | 1.2.0 |
| Media | 320×180 RGBA at 10 FPS; 48 kHz mono PCM |
| Inference sampling | Latest-frame queue of one, sampled at 2 FPS |

The official server was run as a local process with signaling on `127.0.0.1:7880`, ICE/TCP on `7881`, and ICE/UDP mux on `7882`.

## Final repeated runs

Three consecutive complete runs passed every assertion:

| Run | Worker join | Slowest glasses join | Full-size video frames | Filtered transition frames | Server RSS | Non-local peers |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84.7 ms | 42.0 ms | 101 | 34 | 64.5 MB | 0 |
| 2 | 111.4 ms | 78.7 ms | 81 | 36 | 65.0 MB | 0 |
| 3 | 74.9 ms | 46.1 ms | 89 | 34 | 66.5 MB | 0 |

Each complete run contained three deliberate disconnect/rejoin cycles. Across the nine resulting publisher sessions:

- every camera and microphone track reached the worker;
- every session received synthetic return audio;
- every rejoin produced new camera and microphone track SIDs;
- invalid JWT signatures were rejected;
- the bounded sampler processed 320×180 frames and discarded stale frames;
- no established connection to a non-local peer was observed.

## Findings

### LiveKit covers the intended gateway boundary

The server and RTC SDK handled signaling, room membership, track routing, authentication, decoded media delivery, participant rejoin, and audio publication back to the client. The Vision Service can consume `VideoStream` directly without Egress.

### Track SID is a reliable media-epoch boundary

Each deliberate rejoin generated new track SIDs even with the same participant identity. The vision pipeline should reset tracker state when the camera track SID changes.

### The inference subscriber needs a dimension guard

The RTC stream contained valid 320×180 frames and transient 8×8 frames. The spike now filters unexpected sizes before they enter the latest-frame queue. Without that guard, a sampler may send a transition frame to detection.

### Local-only does not mean loopback-only

No non-local established connections were observed. However, LiveKit's ICE/TCP listener bound to all interfaces on port `7881` even though signaling was bound to loopback. A real deployment must restrict RTC ports to the trusted LAN with host/network firewall rules.

### ARM64 packaging exists but execution remains untested

LiveKit publishes Linux ARM64 server images and the Python RTC package provides a `manylinux_2_28_aarch64` wheel. That reduces packaging risk but does not replace a real GN100 codec and media test.

## Remaining adoption gates

1. Publish real camera and microphone tracks from pinned LiveKit Unity SDK 2.0.0 on the actual glasses.
2. Receive and play the worker's audio track on the glasses.
3. Run this spike on the GN100's Linux ARM64 environment.
4. Freeze the actual codec, resolution, frame rate, and hardware-acceleration path.
5. Add WSS/TLS and a backend token endpoint with short-lived, least-privilege grants.
6. Restrict signaling and RTC ports to the trusted LAN.
7. Test transparent recovery from Wi-Fi interruption, packet loss, and LiveKit restart.
8. Run a sustained test with speech and vision models resident.

Until gates 1–3 pass, keep prerecorded video as the demo fallback.
