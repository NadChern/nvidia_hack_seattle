# Agent Service

The conversational boundary for the Visual Memory Assistant. It accepts a
question, calls the deterministic Memory query API, and supervises any wording
before returning it. Port: **8086**.

The default backend is a Google ADK 2.6 `Runner` with one `LlmAgent`, one
LiteLLM model, and two bounded tools: `where_is` and `start_registration`. If the
model omits `where_is` for a deterministically recognized location question, the
runner performs that same bounded Memory call and uses Memory's canonical answer;
unsupported conversation still cannot reach Memory. `StubLlm` remains available
for deterministic, fully offline development; it recognizes bounded “where”
questions and returns Memory's `spoken_answer` unchanged.

## What it refuses to do

- It does not answer from model knowledge or raw vision output.
- It exposes only `where_is` and `start_registration`; there is no open chat or history tool.
- It never lets a model create or choose a `session_id`.
- It does not send transcripts to a non-loopback model endpoint unless
  `VMA_ALLOW_EXTERNAL_LLM=true` is explicitly configured.
- It never logs transcripts, replies, API keys, or tokens.
- It does not implement barge-in or interruption; a reply plays to completion.

## Truthfulness guard

A model rewrite is untrusted. Rules run in order:

1. No Memory result produces a fixed unknown response. A recognized location
   question whose model-selected tool call was omitted first receives the bounded
   deterministic Memory fallback described above.
2. An unknown Memory answer may name no place.
3. `last_confirmed_only` must preserve an explicit uncertainty marker.
4. `ambiguous_object` must name every candidate and preserve ambiguity.
5. Location/content tokens must be grounded in the Memory response.
6. Speech must be non-empty and bounded.

A veto is a successful safety outcome, not an endpoint error. The response uses
`spoken_answer` byte-for-byte and reports `guard: "vetoed:<rule>"`. Registration
never passes model prose through this guard: its prompt and terminal lines come
from a fixed vocabulary and carry `registration:prompt|succeeded|failed` verdicts.

## API

```text
POST /v1/agent/query
  {"text": "Where did I leave my keys?", "session_id": "sess_..."}

GET /v1/status
GET /health/live
GET /health/ready
```

When `VMA_INTERNAL_API_TOKEN` is configured, `/v1/status` and the query endpoint
require `Authorization: Bearer ...`. Deploy mode refuses to start without it.
Health endpoints remain unauthenticated for orchestration.

## Configuration

All variables use the `VMA_` prefix.

| Setting | Default |
|---|---|
| `AGENT_BACKEND` | `llm` (`stub` for offline development) |
| `MEMORY_BASE_URL` | `http://127.0.0.1:8081` |
| `MEMORY_API_TOKEN` | unset |
| `HANDS_FREE_ENABLED` | `false` |
| `GATEWAY_BASE_URL` | `http://127.0.0.1:8080` |
| `SPEECH_BASE_URL` | `http://127.0.0.1:8085` |
| `VISION_BASE_URL` | `http://127.0.0.1:8082` |
| `REGISTRATION_CAPTURE_SECONDS` | `6.0` |
| `REGISTRATION_TIMEOUT_S` | `20.0` |
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` |
| `LLM_MODEL` | `openai/qwen3:4b` |
| `LLM_API_KEY` | unset |
| `ALLOW_EXTERNAL_LLM` | `false` |
| `LLM_TIMEOUT_S` | `30` |
| `REQUEST_TIMEOUT_S` | `30` |
| `MAX_TURNS_KEPT` | `6` |

External providers are explicit, opt-in evaluation profiles. The complete
copy/paste and browser instructions for both supported laptop options are in
[Agent Laptop Testing](../../docs/14-Agent-Laptop-Testing.md).

### Laptop MiniCPM API profile

The resource-safe laptop profile uses ModelBest's OpenAI-compatible API instead
of loading another model into the 8 GB GPU. On the measured WSL laptop,
`dev_stack.sh` forces fixture Vision while keeping real CUDA STT and TTS. MoGe
warmup beside Speech exhausted the local-model profile; the earlier attribution
of a later shutdown to Kokoro was incorrect—the observed failure was a Console
reload bug, and Parakeet plus Kokoro have since been validated together. On
2026-08-11, `MiniCPM-V-4.5-9B` passed a basic completion, emitted the required
`where_is({"label":"keys"})` tool call, and passed the Agent's opt-in real-model
integration test in 7.35 seconds using fixture Memory data. A subsequent full
ADK → Memory fixture → guard run called the tool once, preserved
`answer_status="confirmed"`, and passed the guard with Memory's grounded reply.

From the repository root, read the API key without putting it in shell history,
then start the stack:

```bash
printf 'ModelBest API key: ' >&2
IFS= read -rs VMA_LLM_API_KEY
printf '\n' >&2
export VMA_LLM_API_KEY

VMA_ALLOW_EXTERNAL_LLM=true \
VMA_LLM_BASE_URL=https://api.modelbest.cn/v1 \
VMA_LLM_MODEL=openai/MiniCPM-V-4.5-9B \
VMA_DETECTOR_KIND=fixture \
VMA_LLM_TIMEOUT_S=120 \
./scripts/dev_stack.sh --no-sync
```

This is an explicit external-egress profile, not a fallback. It sends the
transcribed question and the complete Memory tool response to ModelBest; it
does not send audio, images, or evidence media. Use the public trial key from
the upstream [MiniCPM API guide](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md)
only for evaluation, do not commit it, and expect that a shared key may expire
or be rate-limited.

### Laptop OpenRouter free-model profile

The second laptop option uses
[NVIDIA Nemotron 3.5 Lightning (free)](https://openrouter.ai/nvidia/nemotron-3.5-lightning:free).
OpenRouter currently advertises `tools` and `tool_choice` for this route, and
LiteLLM resolves the documented model string to its OpenRouter provider.
Create a key at [OpenRouter Keys](https://openrouter.ai/settings/keys), then run:

```bash
printf 'OpenRouter API key: ' >&2
IFS= read -rs VMA_LLM_API_KEY
printf '\n' >&2
export VMA_LLM_API_KEY

VMA_ALLOW_EXTERNAL_LLM=true \
VMA_LLM_BASE_URL=https://openrouter.ai/api/v1 \
VMA_LLM_MODEL='openrouter/nvidia/nemotron-3.5-lightning:free' \
VMA_DETECTOR_KIND=fixture \
VMA_LLM_TIMEOUT_S=120 \
./scripts/dev_stack.sh --no-sync
```

The free route may be rate-limited or temporarily unavailable and remains
subject to OpenRouter/provider data policies. It is never an automatic
fallback. OpenRouter metadata confirms the exact free route currently supports
tools; the automated LiteLLM test checks provider routing only, not route
existence or capabilities. The live Agent integration test remains opt-in
because this repository has no OpenRouter key.

Do not drop `:free` as a rate-limit workaround: the paid
`nvidia/nemotron-3.5-lightning` route currently does not advertise tools. A
route that omits the `where_is` call causes guard rule 1 to replace each reply;
watch `metrics.guard_vetoed["vetoed:1"]` on `/v1/status` for that signal.

### GN100 Cosmos3 switch boundary

For the hackathon GN100, `nvidia/Cosmos3-Nano` (16B, BF16) is the larger
Cosmos3 candidate because it leaves substantially more of the 128 GB unified
memory budget for detection, speech, media, and operating-system headroom than
`Cosmos3-Super` (64B). Cosmos3 is evaluated first as the selective video event
verifier, where its physical and temporal reasoning matches the task. It is not
yet the Agent default: its ADK tool calling, guarded final answer, ARM64 runtime,
and complete-workload memory use still need the physical GN100 gate.

The Agent's model boundary remains an OpenAI-compatible base URL plus model ID,
so a validated local Cosmos vLLM endpoint can replace the MiniCPM endpoint
without changing the Agent, Memory API, or response contract. Do not make that
switch until the Cosmos tool-call and guard tests pass. The verifier requires a
separate OpenAI-compatible adapter because its current implementation uses the
Ollama `/api/chat` wire format.

## Hands-free loop

When enabled, the service polls the authenticated Gateway session list and
opens Speech's existing STT WebSocket only for sessions with a publisher. Every
completed non-empty transcript is forwarded to Nemotron, whose system
instruction owns find-versus-register-versus-unsupported intent selection. There
is no listener regex, wake-prefix requirement, or manual-trigger requirement;
“where are my keys?” works as a bare utterance. The glasses button remains a UI
convenience but is not an authorization gate.

This deliberately trades the previous ambient-speech filter for model-owned
intent routing. Unsupported conversation receives no Memory authority: without
a Memory result, the deterministic guard still prevents a location claim.

Every completed transcript is also posted to the Gateway's bounded device-event channel.
A guarded answer is posted there with `answer_status`, object ID, guard verdict, and
latency, then synthesized through Speech's WAV endpoint, validated and resampled to the
Gateway's 48 kHz signed-16-bit PCM contract, and streamed to
`/v1/return-audio/{session_id}`. Event-delivery failure is isolated from the guarded audio
reply. Audio, evidence, and frames never enter the model prompt.

Barge-in is explicitly out of scope. The current reply plays to completion.

## Development and checks

```bash
cd services/agent
uv sync --frozen --all-groups
uv run uvicorn agent.main:app --port 8086

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

| Test area | Guarantee |
|---|---|
| `test_guard.py` | Every guard rule vetoes unsafe text and accepts grounded text |
| `test_config.py` | External egress is opt-in and secrets remain redacted |
| `test_tools_memory.py` / `test_tools_register.py` | Complete query state and bounded registration polling are preserved |
| `test_api_query.py` | Offline end-to-end query and no-tool behavior |
| `test_agent_litellm.py` | Fake tool-calling completion, bounded ADK sessions, opt-in real model |
| `test_listener.py` | Wake-prefix/intent gate prevents unwanted model calls and duplicate registration audio |
| `test_workflow.py` | Registration always reaches a fixed prompt and terminal success/failure line |
| `test_reply.py` | WAV validation, resampling, and PCM-only return transport |
| `test_health.py` / `test_status.py` | Operational and trust-boundary reporting |
| `test_logging.py` | Structured logging redacts secrets and binary payloads |

## Measured model run

On the WSL development box, a warm local Ollama `qwen3-vl:4b` round trip took
**11.5 seconds**. It called `where_is` correctly but emitted a reasoning
preamble, so guard rule 5 vetoed it and returned Memory's canonical answer.
That is safe but too slow for the target voice UX. OpenRouter measurement is
pending because no credential is configured on this machine.
