# Read this before following SKILL.md

`SKILL.md` is vendored unchanged from [livekit/agent-skills](https://github.com/livekit/agent-skills)
(MIT). It is useful for LiveKit API discipline. It is also written for a
different deployment model than this project's, and following its recommended
path would violate constraints we have already committed to.

| SKILL.md recommends | This project |
|---|---|
| LiveKit Cloud (`wss://*.livekit.cloud`) | **Self-hosted** `livekit/livekit-server` v1.13.4, loopback or trusted LAN |
| LiveKit Inference for AI models | **No cloud model APIs.** Speech and vision run on the GN100 |
| Credentials from a Cloud project | Generated locally; the gateway's validator refuses well-known dev values |

[Privacy and Security](../../../docs/07-Privacy-and-Security.md) puts cloud
speech, signaling, TURN relay, telemetry, and model APIs outside the trust
boundary. An integration test enforces it:
`test_the_gateway_talks_to_nothing_off_this_machine` sweeps for established
sockets to non-local peers and fails if it finds any. Pointing anything at
LiveKit Cloud turns that red, correctly.

SKILL.md says as much itself, in one line near the top that is easy to miss:

> This skill is for LiveKit Cloud developers. If you're self-hosting LiveKit,
> some recommendations (particularly around LiveKit Inference) won't apply
> directly.

## What does carry over

- **Never trust model memory for LiveKit APIs.** The SDK moves faster than
  training data. This is the single most valuable rule in the document and it
  applies regardless of hosting.
- **Test agent behaviour rather than eyeballing it.** Same reasoning as the
  rest of this repo.
- **Latency and context discipline** for anything conversational.

## Where our own LiveKit decisions are written down

- [Media Relay Contract](../../../docs/12-Media-Relay-Contract.md) — the wire
  format, and why the gateway holds the only subscription
- [S01 spike results](../../../docs/spikes/livekit-media-gateway/RESULTS.md) —
  what was actually measured, including the track-SID epoch rule
- [tools/dev-livekit/README.md](../../../tools/dev-livekit/README.md) — running
  a server locally, and why tunnels are not an option

## Provenance

`SKILL.md` is byte-identical to `livekit/agent-skills@main:skills/livekit-agents/SKILL.md`,
verified by sha256 `1d8f9c6e…`. Note that this does **not** match the
`computedHash` recorded in `skills-lock.json`; that value appears to be
computed by a different scheme, since the local file matches upstream exactly.
Re-verify against upstream rather than against the lock.
