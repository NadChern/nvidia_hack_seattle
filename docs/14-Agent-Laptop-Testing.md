# Agent Laptop Testing

This guide tests the complete laptop query path without loading a local LLM:

```text
browser microphone → Speech STT → Agent → Memory tool → guard → Speech TTS
```

The default project path remains local. Both profiles below are explicit external-LLM evaluations: transcript text and the complete Memory `QueryResponse` leave the workstation. Audio, images, video, and evidence bytes are not sent to the provider. See [Privacy and Security](07-Privacy-and-Security.md#optional-external-language-model-evaluation).

## Profile for the 8 GB WSL laptop

Keep `VMA_ALLOW_RESOURCE_OVERSUBSCRIPTION` disabled: MoGe warmup beside Speech has exceeded this laptop's safe local-model profile. The measured profile uses real CUDA Parakeet STT, real CUDA Kokoro TTS, and fixture Vision. An earlier shutdown attributed to reply-audio handoff was traced to a Console reload bug, and Parakeet plus Kokoro have since passed together. Use `--no-sync` after dependencies are installed so testing does not download model runtimes.

Stop any existing stack with `Ctrl-C` before starting one of the following options.

## Option 1 — ModelBest MiniCPM

ModelBest publishes a shared evaluation key in the upstream [MiniCPM API guide](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md). Copy the current key from that page; do not put it in source, `.env`, a command argument, or a commit.

```bash
cd "$HOME/projects/nvidia-spark-hack-seattle"  # or your actual clone directory

printf 'ModelBest API key: ' >&2
IFS= read -rs VMA_LLM_API_KEY
printf '\n' >&2
export VMA_LLM_API_KEY

VMA_ALLOW_EXTERNAL_LLM=true \
VMA_LLM_BASE_URL='https://api.modelbest.cn/v1' \
VMA_LLM_MODEL='openai/MiniCPM-V-4.5-9B' \
VMA_DETECTOR_KIND=fixture \
VMA_LLM_TIMEOUT_S=120 \
./scripts/dev_stack.sh --no-sync
```

This profile has passed the fixture-backed ADK tool-call and deterministic-guard integration test.

## Option 2 — OpenRouter free Nemotron

Create an OpenRouter key at [OpenRouter Keys](https://openrouter.ai/settings/keys). The documented test model is [NVIDIA Nemotron 3.5 Lightning (free)](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free). OpenRouter currently advertises `tools` and `tool_choice` for the free route, which the Agent requires.

```bash
cd "$HOME/projects/nvidia-spark-hack-seattle"  # or your actual clone directory

printf 'OpenRouter API key: ' >&2
IFS= read -rs VMA_LLM_API_KEY
printf '\n' >&2
export VMA_LLM_API_KEY

VMA_ALLOW_EXTERNAL_LLM=true \
VMA_LLM_BASE_URL='https://openrouter.ai/api/v1' \
VMA_LLM_MODEL='openrouter/nvidia/nemotron-3.5-lightning:free' \
VMA_DETECTOR_KIND=fixture \
VMA_LLM_TIMEOUT_S=120 \
./scripts/dev_stack.sh --no-sync
```

Free routes may be rate-limited, temporarily unavailable, or subject to provider routing and retention policies. They are evaluation options, never automatic fallbacks. The exact Nemotron route has been validated against OpenRouter's model metadata; the automated LiteLLM test validates provider routing only, not route existence or capabilities. A live integration run still requires an OpenRouter credential.

Keep the `:free` suffix. The paid `nvidia/nemotron-3.5-lightning` route currently does not advertise tool support, so dropping the suffix is not a safe rate-limit workaround. A route without tools may still answer general-assistant questions, but it cannot reliably answer personal visual-memory requests and therefore does not satisfy the deployment profile.

## Confirm the selected provider

In a second terminal:

```bash
curl -s http://127.0.0.1:8086/v1/status | jq
```

Expect `backend` to be `external` and verify the model and endpoint host match the option selected above.

Test a direct query:

```bash
curl -s http://127.0.0.1:8086/v1/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"text":"Where did I leave my keys?"}' | jq
```

With no seeded memory, a truthful unknown answer is expected. A provider or tool failure must return an error; it must not invent a location.

## Browser voice round trip

1. Open `http://localhost:5173`.
2. Select **Glasses**, then **Publish**.
3. Wait for Speech to report `listening`.
4. Select **Assistant**.
5. Hold the ask button, say “Where did I leave my keys?”, and release.
6. Confirm the transcript, text reply, `answer_status`, guard verdict, and external-provider warning appear.

A completed transcript must arrive within 12 seconds after release. If none arrives, the turn disarms and displays an error; later unrelated speech will not trigger it. In this 8 GB profile, `/v1/status` must report real STT and real TTS, and the guarded reply should be audible. Keep resource oversubscription disabled so Vision remains on fixtures.

## Stop

Press `Ctrl-C` in the stack terminal. Verify ports are clear before changing providers:

```bash
ss -ltn '( sport = :8080 or sport = :8081 or sport = :8082 or sport = :8085 or sport = :8086 or sport = :5173 )'
```
