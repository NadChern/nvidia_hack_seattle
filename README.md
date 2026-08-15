# Visual Memory Assistant

A privacy-friendly assistant for people with memory difficulties. Smart glasses capture first-person audio and video; a local workstation turns important visual events into searchable memories.

> *“Where did I leave my keys?”*
> *“On the living-room coffee table at 10:42 — but they were picked up afterward and I have not confirmed a new location since.”*

That second sentence is the hard part. Being usefully uncertain matters more than sounding confident.

Built for the NVIDIA Spark Hackathon, Seattle. The default path runs locally: no cloud speech, hosted model APIs, or first-person video leaving the machine. A temporary external language-model evaluation profile exists only through explicit opt-in and is labeled in the console.

## Start here

**New to the project? Open [`docs/onboarding.html`](docs/onboarding.html) in a browser.** It explains what the Media Gateway does, why it is built this way, and gets you consuming real media — with no glasses and no special hardware.

Working with a coding agent? Point it at [`docs/13-Dev-Onboarding.md`](docs/13-Dev-Onboarding.md), the same material written for machines. Repository-wide rules live in [`AGENTS.md`](AGENTS.md).

See the whole thing running in one command:

```bash
./scripts/dev_stack.sh
```

That is LiveKit, all five services and the console. It works on Linux and on Apple Silicon, choosing the model runtime this machine can safely run and saying which one it chose. On the measured 8 GB WSL laptop, the launcher uses real Speech STT but stub TTS, fixture Vision, and the stub Agent for a local LLM; an explicit external profile supplies a real Agent without local model pressure. Reply-audio acceptance requires another profile because WSL terminated at that handoff on this laptop. Open the URL it prints and press **Publish** — your own camera and microphone stand in for the glasses.

Just the gateway, for work on the relay itself:

```bash
cd services/media-gateway && ./scripts/dev_up.sh
```

## What exists today

| Component | State |
|---|---|
| **Media Gateway** (`services/media-gateway`) | Working. Subscribes to the glasses, samples video, filters bad frames, relays to consumers, carries speech back. |
| **Media contract** (`packages/media-contract`) | Working. Wire models, client, and recorded fixtures every consumer builds against. |
| **Vision** (`services/vision-worker`) | Working. Detects, tracks, decides an object was placed, and records it as a memory. |
| **Memory** (`services/application-memory`) | Working. Stores events and evidence, answers "where did I leave it" with an explicit uncertainty. |
| **Speech** (`services/speech`) | Working. Parakeet and Kokoro on CUDA or MLX; stub backends elsewhere, reported as such. |
| **Console** (`apps/console`) | Working. Publish a camera, watch detections land on it live, exercise memory, speech, and the Assistant. |
| **Agent / Query orchestration** (`services/agent`) | Working. ADK calls the bounded Memory tool, a deterministic guard prevents unsupported location claims, and replies return through the console or glasses audio path. |

## Layout

```text
services/          deployable services, each owning its lockfile and Dockerfile
packages/          shared libraries; media-contract is what consumers depend on
tools/dev-livekit/ local media server: config, downloader, listener check
docs/              architecture, contracts, privacy, onboarding
deploy/            release configuration
compose.dev.yaml   local development dependencies
compose.yaml       the GN100 release topology
compose.gpu.yaml   overlay: the real models, on a machine with a GPU
```

## Documentation

Start with [Overview](docs/00-Overview.md), which indexes everything. The ones you are most likely to need:

- [Dev Onboarding](docs/onboarding.html) — from a fresh clone to media flowing
- [Media Relay Contract](docs/12-Media-Relay-Contract.md) — how Vision and Speech receive media
- [Data Contract](docs/06-Data-Contract.md) — what counts as a memory, and the state reducer
- [Privacy and Security](docs/07-Privacy-and-Security.md) — the trust boundary and what may never cross it
- [Team Split](docs/05-Team-Split.md) — ownership and integration milestones
- [Agent Laptop Testing](docs/14-Agent-Laptop-Testing.md) — safe MiniCPM and OpenRouter free-model test profiles

## House rules

- **Nothing leaves the machine on the default path.** No cloud speech, hosted model APIs, or tunnels are selected automatically. External LLM evaluation requires `VMA_ALLOW_EXTERNAL_LLM=true`, is visibly labeled, and follows the disclosure in [Privacy and Security](docs/07-Privacy-and-Security.md#optional-external-language-model-evaluation).
- **No secrets in the repository** — not in code, logs, images, or Compose files.
- **No raw media, transcripts, or tokens in logs.** Log a size and a digest.
- **Python 3.11 and uv.** Every service owns a `uv.lock`; no `requirements.txt`, Poetry, or a second Python version.

```bash
python3 .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
```

## Event

[NVIDIA Spark Hack: Seattle](https://luma.com/spark-hack-seattle?tk=8XBLkn) · [DGX Spark](https://build.nvidia.com/spark?utm_source=luma) · [NVIDIA Build model endpoints](https://build.nvidia.com/models?utm_source=luma) · [VSS Spark Playbook](https://build.nvidia.com/spark/vss?utm_source=luma)
