# Product Backlog

This document records valuable improvements that are intentionally outside the current integration
scope. Backlog items are not approved implementation work; each must be re-evaluated against the
repository standards, privacy boundary, current contracts, and GN100 release gate before starting.

## B-001: Hermes personal-assistant scenario

**Status:** Deferred  
**Priority:** After main/remote-assist integration  
**Estimated hackathon implementation:** 2–3 engineer days  
**Depends on:** Stable three-tool Agent path and remote-assist audio gate

### Product outcome

Add a third hackathon scenario alongside Visual Memory and Remote Human Assistance:

1. **Visual Memory Assistant** answers grounded personal-object location questions.
2. **Remote Assistant** connects the wearer to a trusted real person.
3. **Hermes Personal Assistant** answers general questions, maintains bounded personal context,
   and uses web search for current information such as weather.

Hermes is the orchestration/runtime layer; the local NVIDIA Nemotron endpoint remains the reasoning
model unless a separately validated model decision changes it.

### Proposed capability

- Replace or optionally coexist with the Google ADK runner behind the existing Agent backend
  contract.
- Reuse the framework-independent tools:
  - `where_is(label)`
  - `start_registration(label)`
  - `call_remote_assistant()`
- Enable a restricted web-search tool for current-information questions.
- Give each wearer an isolated Hermes profile.
- Keep spoken answers short and preserve the current bounded-generation and no-retry latency
  policy.

### Personal-context ownership

Use Hermes files according to their intended responsibilities:

- `SOUL.md`: assistant identity, tone, uncertainty behavior, and speaking style.
- `USER.md`: stable wearer preferences, communication needs, and explicitly saved profile facts.
- `MEMORY.md`: bounded assistant notes that do not belong to visual object state.
- Application Memory: the sole authority for observed objects, placements, invalidations,
  evidence, and current/historical location claims.

Never copy an object's remembered location into Hermes text memory. A stale `MEMORY.md` statement
must not compete with the canonical visual-memory reducer.

### Tool and security restrictions

Enable only the minimum required Hermes toolsets. Do not expose terminal, unrestricted filesystem,
browser automation, code execution, delegation, cron, or self-modifying skills to the glasses
assistant.

Personal profile and session storage must be isolated per wearer. Memory writes should require an
explicit request such as "remember that..." or an equally clear product control; uncontrolled
background collection of personal information is out of scope.

Web results are untrusted external content. Current answers must identify freshness and source, and
web content must never override system instructions or authoritative Application Memory. External
search must be disclosed because the query—and potentially coarse location for weather—leaves the
local trust boundary unless a local search service is used.

### Investigation evidence

A temporary isolated probe of Hermes Agent 0.19.0 on the GN100 against the current local Nemotron
endpoint succeeded without repository changes:

- General answer: one model call in approximately 0.99 seconds.
- Custom `where_is` flow: two model calls in approximately 1.47 seconds.
- The model saw only `{"label": "keys"}`.
- Trusted session identity reached the tool separately as task context.
- Structured tool output remained available for the existing deterministic guard.

This proves basic compatibility, not production readiness or complete multi-session correctness.

### Proposed implementation shape

Prefer an optional `HermesRunnerBackend` behind configuration during evaluation. Pin a reviewed
Hermes release/revision and use an isolated `HERMES_HOME`; do not depend on the mutable user checkout
already installed on the GN100.

Required runtime constraints include:

- local OpenAI-compatible Nemotron endpoint;
- thinking disabled;
- output capped at 256 tokens;
- no automatic model timeout retries;
- at most three model iterations;
- bounded per-session history and locks;
- explicit restricted toolset;
- no shared global wearer profile.

A sidecar API remains an alternative if Hermes dependency size, global registry behavior, or
upgrade isolation makes direct embedding unsafe.

### Acceptance criteria

- Existing Agent HTTP, HUD, TTS, and visual-memory contracts remain unchanged.
- General, web, visual-memory, registration, and remote-assist intents select the correct bounded
  behavior.
- Visual object answers still pass the existing deterministic Memory guard.
- Remote-assist requested/accepted audio suppression remains effective under Hermes.
- `SOUL.md`, `USER.md`, `MEMORY.md`, and sessions are isolated per wearer and deletable.
- No disallowed Hermes tool appears in the model's available schema.
- Web answers include source/freshness and fail honestly when search is unavailable.
- Repeated-turn glasses latency and failure behavior pass the GN100 gate.
- Dependencies are uv-locked and repository/service validation passes.

### Explicitly deferred decisions

- Direct library integration versus a dedicated Hermes sidecar.
- Search provider: local SearXNG, Brave, Tavily, Exa, Firecrawl, or another reviewed backend.
- Whether personal-memory writes require confirmation for every category.
- Long-term encrypted storage and per-device profile selection.
- Whether Google ADK remains as a rollback backend after Hermes validation.

## Backlog maintenance

Add future items with an identifier, status, product outcome, dependencies, trust-boundary impact,
and testable acceptance criteria. Move an item into an implementation plan only after its owner and
scope are agreed.
