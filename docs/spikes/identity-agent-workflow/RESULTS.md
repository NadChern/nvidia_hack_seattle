# Agent registration workflow Phase-4 results

## Decision

**Adopt the tested workflow shell.** ADK exposes `start_registration` beside `where_is`; the
model routes intent, while a background task owns all registration side effects and speech. The
wake gate now admits only bounded where or `remember/scan/learn my X` shapes. Registration never
reuses the where-answer rewrite path and cannot speak model-generated progress text.

## Automated gates

- Fake LiteLLM emits `start_registration({"label":"keys"})`; ADK calls it once with the
  authenticated session and returns `registration_started=true`.
- Wake-prefixed remember/scan/learn utterances reach the backend; unsupported speech still stops
  before any model call.
- Listener sends no duplicate model reply for registration; the workflow exclusively owns audio.
- Workflow success: fixed prompt then fixed confirmation, 1/1 terminal completion.
- Honest weak-footage and dependency-failure fixtures: fixed prompt then fixed failure, 2/2
  terminal completions.
- Overlapping registration in one session is refused before a second side effect.
- Full pre-existing guard suite passes unchanged. Registration fixed vocabulary passes byte-for-
  byte with typed `registration:prompt|succeeded|failed` verdicts; non-scripted text is replaced.
- Agent status reports Vision host, capture/timeout configuration, and registration
  started/succeeded/failed counters without credentials.

## Pending measurements

The real-model three-way intent confusion matrix and trigger-to-TTS/capture-done-to-confirmation
p95 require the selected live model plus Speech/Vision services. The workflow implementation is
fixture-demoable now; the headline voice loop still requires a live video session and Phase-3
capture. No new LiveKit SDK API was introduced, and LiveKit documentation MCP was unavailable in
this implementation session.
