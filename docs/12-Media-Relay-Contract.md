# Media Relay Contract

This document defines the transport between the Media Gateway and its media consumers — today the Vision Service and the Speech Service. It is the wire format for the `mediaWorker → sampled video` and `mediaWorker → audio` edges in the [Architecture Diagram](10-Architecture-Diagram.md).

**This is not the canonical observation contract.** Nothing here is a memory, an observation, or trusted state. [Data Contract and Memory Semantics](06-Data-Contract.md) remains the only canonical event and response format. A relay message carries decoded media and the metadata needed to interpret it; it never carries an object, a location, or a confidence.

Protocol version: `media-relay/1.0`. The executable form of this document is `packages/media-contract`; when the two disagree, the package is wrong and must be corrected.

## Boundary

The Media Gateway holds the **only inference subscription**. Vision, Speech, Memory, and other processing consumers do not join the room, do not hold LiveKit tokens, and do not depend on WebRTC. This keeps inference decode cost and token surface at one, keeps the bounded sampler and dimension guard in one implementation, and lets the transport be replaced without touching inference or memory — a requirement carried over from the [S01 spike](09-Spike-Plan.md).

The operator console is the sole carve-out. It may join an existing session with a short-lived `viewer` grant minted by `POST /v1/sessions/{session_id}/viewer` behind the internal bearer check. That grant can subscribe but cannot publish media or data. No inference, memory, or reply path may depend on the viewer being present.

Video publishers on the production glasses path use one 1280×720 layer with simulcast disabled. SG-B measured an operator joining an unpinned simulcast room collapsing gateway ingest from 720p to 320×180. SG-C then measured that explicit gateway-high/viewer-low requests were insufficient: 96/121 gateway frames arrived at a rejected lower dimension after viewer join. A single-layer publisher admitted 120/120. The console still requests low when a development publisher offers simulcast, but the production inference guarantee comes from there being no lower layer to select.

The media plane itself — ingress, codecs, jitter buffering, ICE, return audio — stays entirely on LiveKit. The relay begins only after a frame is decoded, sampled, and dimension-checked.

## Framing

One logical message is exactly one WebSocket **binary** frame:

```text
magic       4 bytes   b"VMA1"
header_len  4 bytes   uint32 big-endian
header      N bytes   UTF-8 JSON, discriminated on "type"
payload     rest      raw bytes; empty for control messages
```

Header and payload travel together rather than as two WebSocket messages. Two messages would force every consumer to pair them, and any interleaving bug would corrupt data silently. Because control messages use the same framing with an empty payload, **control and media are strictly ordered**: an `epoch_started` can never be observed after a frame belonging to the epoch it starts.

Headers are JSON rather than a binary encoding because they are a few hundred bytes at the relay's sampled rate, they keep the Pydantic models the single source of truth, and they can be read with ordinary tooling while debugging. `header_len` is capped at 65536 bytes and checked before any allocation.

Consumers **must ignore unknown header fields**. Additive fields are a minor version bump and must not break a pinned consumer.

## Message types

| `type` | Payload | Meaning |
|---|---|---|
| `stream_hello` | none | First message on every connection. Describes what is already in flight. |
| `session_started` | none | A session began. |
| `session_ended` | none | A session ended, with a reason. |
| `epoch_started` | none | **Reset per-track state now.** |
| `epoch_ended` | none | The epoch's track went away, with a reason. |
| `video_frame` | encoded image | One sampled, dimension-guarded frame. |
| `audio_chunk` | PCM | Coalesced audio. Never dropped. |
| `lifecycle_signal` | none | In-band copy of what the gateway sends to Memory. |
| `keepalive` | none | Sent while idle so silence is distinguishable from a dead socket. |

Every message carries `protocol_version`. All timestamps are UTC ISO-8601 with a `Z` suffix and millisecond precision, per [Recommended Architecture](01-Recommended-Architecture.md).

## Media epochs

`epoch_id` **is the LiveKit track SID.**

The S01 spike established this: across three deliberate rejoin cycles with an unchanged participant identity, every rejoin produced new track SIDs. Identity is therefore not a usable boundary, and the SID is. This matches the rule in [Hackathon Stack](03-Hackathon-Stack.md) — *"treat a changed LiveKit camera track SID as a new tracking epoch"* — and makes [Model Landscape](02-Model-Landscape.md)'s *"tracker IDs are scoped to a media epoch and must not survive reconnects"* mechanically enforceable.

On `epoch_started` a consumer **must** discard tracker state, identity assignments, and any partially accumulated temporal event for that stream. `sequence` restarts at 0 for each epoch.

Video and audio are separate LiveKit tracks and therefore have **different** `epoch_id` values. They are correlated by `session_id`.

## Video

Frames are sampled to a configured rate (8 FPS by default) using a latest-wins slot of size one, so a slow consumer never applies backpressure to LiveKit ingest. Frames whose dimensions do not match the guard are counted and discarded **before** they reach the sampler; the spike observed transient 8×8 frames making up roughly a quarter of all frames during simulcast adaptation and track teardown, and forwarding one to a detector would be a silent correctness bug.

`dropped_since_previous` reports how many frames the latest-wins slot evicted since the frame before it, so a consumer can measure its own sampling gaps without polling `/v1/status`.

Default encoding is **JPEG, quality 92, 4:4:4 subsampling**. At 320×180 raw RGBA would be harmless, but real glasses at 720p would be 3.7 MB per frame and buffering that for multiple consumers is pure waste. 4:4:4 preserves chroma edges for segmentation. `image/jpeg` is already the canonical evidence media type in [Data Contract](06-Data-Contract.md). The gateway encodes once and fans the same immutable bytes to every subscriber.

`encoding: "rgba_raw"` is available per-connection for pixel-exact work; it sends the LiveKit RGBA buffer untouched.

`sha256` covers the payload. It makes byte-exact fixtures possible and matches the evidence-digest discipline used elsewhere.

## Audio

Payload is raw interleaved PCM: `s16le`, mono, 48 kHz by default. The gateway coalesces five 20 ms LiveKit frames into one 100 ms message to cut message rate from 50/s to 10/s.

`pts_samples` is the cumulative sample count since the epoch began, so a gap is detectable by arithmetic rather than inference.

**Audio is never dropped.** Each subscriber has a bounded FIFO of about two seconds. On overflow the gateway **closes that subscriber's socket** with code `1011` and reason `audio_backpressure`. Silently dropping audio would corrupt transcription invisibly; a loud failure is the correct behaviour.

## Return audio

Media also flows the other way. The gateway joins each room as a participant and publishes one audio track — `assistant-tts` by default — so synthesized speech reaches the device. The gateway owns the track and its pacing; the Speech Service only supplies samples.

```text
WS   /v1/return-audio/{session_id}      binary int16 PCM at the configured rate
POST /v1/return-audio/{session_id}/tone dev stand-in until Speech exists
```

This direction deliberately has **no framing**. It is a single-purpose one-way feed with no interleaved control messages, so the `VMA1` envelope would add ceremony without adding meaning. A text frame is refused rather than ignored, because a producer sending text is speaking a different protocol and silently playing nothing would be worse than failing.

Backpressure comes from the SDK: the capture call waits until the outbound queue has room, so a producer that outruns real time is slowed rather than buffering without limit. A trailing partial frame is padded with silence rather than dropped, so the end of an utterance is not clipped.

## Lifecycle signals

`lifecycle_signal` carries the same envelope the gateway posts to the Memory Service, relayed in band so consumers observe it in order with the media.

The envelope deliberately is **not** an observation. It has no `object`, `location`, `confidence`, or `evidence`, because the gateway observes none of those. `scope` carries the blast radius: a signal scoped by `media_epoch_id` applies to every object whose in-transit state originated in that epoch.

> **Status: accepted.** The `scope` block resolves the contradiction between docs/06 naming the gateway as a `track_lost` emitter and its reducer treating `track_lost` as per-object. The Memory Service accepts this envelope at `POST /v1/lifecycle` and fans an epoch-scoped signal out to every object whose in-transit state began in that epoch. See [Data Contract § Lifecycle signals](06-Data-Contract.md#lifecycle-signals).

## Reconnection

`MediaClient` reconnects with exponential backoff. After every reconnection the gateway re-sends `stream_hello` followed by a synthetic `epoch_started` for each still-active epoch, so a consumer that dropped mid-epoch still resets correctly rather than resuming against stale tracker state.

A consumer must treat a reconnect as at least as strong a reset as an epoch change.

## Consuming the stream

```python
from visual_memory_media_contract import MediaClient
from visual_memory_media_contract.protocol import EpochEnded, EpochStarted, VideoFrame

async for message in MediaClient("ws://localhost:8080/v1/stream/video", token=...):
    match message:
        case EpochStarted():
            tracker.reset(message.epoch_id)
        case VideoFrame():
            tracker.step(message.rgb, message.captured_at)
        case EpochEnded():
            tracker.finalize(message.reason)
```

### Testing a consumer with no gateway

`packages/media-contract` ships recorded fixtures and a replay server, so a consumer can be exercised end to end with no gateway, no LiveKit, and no hardware:

```python
from visual_memory_media_contract.testing import assert_matches_fixture, replay_server

async with replay_server("video_session_basic") as url:
    received = [message async for message in MediaClient(url, reconnect=False)]

assert_matches_fixture(received, "video_session_basic")
```

`video_session_basic` covers a session, two epochs across a rejoin with an unchanged participant identity, a reported frame drop, a lifecycle signal, a keepalive, and a clean shutdown. `audio_session_basic` carries three seconds of 48 kHz mono with one deliberate 500 ms gap where `pts_samples` jumps while `sequence` stays contiguous — a consumer that counts messages instead of reading `pts_samples` will miss it.

`flaky_replay_server` drops the first connection mid-stream to exercise reconnect handling.

Per [Team Split](05-Team-Split.md), this interface is complete only when the gateway's provider fixture passes inside the consumer's test harness; both sides assert against the same fixture files, so a provider change that breaks a consumer fails on both sides.

### Inspecting a live stream

```text
uv run python -m visual_memory_media_contract.tap ws://127.0.0.1:8080/v1/stream/video
```

Prints one line per message with ids, dimensions, payload sizes, and digests — never payload bytes, transcripts, or tokens.

## Device HUD events

The glasses do not subscribe to the decoded-media relay or open a Speech/Agent socket.
The Agent already owns one STT socket per active session, so it pushes each completed
transcript and each guarded hands-free reply to the Gateway:

```text
POST /v1/device/{session_id}/events    internal bearer; Agent producer
WS   /v1/device/{session_id}/events    owning device credential or internal operator
```

This JSON channel is deliberately separate from `VMA1`. It carries no media and cannot be
used to publish LiveKit data. Transcript events preserve epoch and sample positions. Reply
events preserve question, reply, `answer_status`, object ID, guard verdict, and latency;
a guard veto is a successful refusal event, not a socket failure.

Fan-out queues are bounded per subscriber. A slow HUD is closed with
`event_backpressure` rather than growing memory without limit. Events are session-scoped,
not replayed, and no inference or audio path depends on a HUD subscriber being present.

The owning device may also `POST /v1/device/{session_id}/manual-trigger`. This arms one
consume-once, 15-second turn. The Agent consumes it when the next transcript arrives and
still requires the same bounded where-question shape; it only bypasses the wake prefix.
This gives the X3 touchpad/UI a deterministic room-noise fallback without granting the
device access to the Agent or allowing arbitrary text injection.

## Versioning

Additive optional fields are a minor bump and must not break pinned consumers. Removing a field, renaming one, changing a type, or changing the meaning of an enum value is a major bump and requires updating the provider, every consumer, and the shared fixtures together.

## Related

- [Recommended Architecture](01-Recommended-Architecture.md) — where the Media Gateway sits
- [Data Contract and Memory Semantics](06-Data-Contract.md) — the canonical observation format this document is *not*
- [Privacy and Security](07-Privacy-and-Security.md) — retention, redaction, and port policy
- [Spike Plan](09-Spike-Plan.md) — S01, which produced the sampler, the dimension guard, and the epoch rule
