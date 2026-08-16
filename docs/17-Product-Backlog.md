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

## B-002: Automatic enrollment image extraction

**Status:** Deferred; manual operator-guided enrollment is the demo baseline

**Priority:** Post-demo registration improvement
**Depends on:** A representative physical-capture evaluation set, a durable pending-view lifecycle,
and GN100 latency measurements

### Product outcome

Automatically propose high-quality personal-object reference crops from a bounded glasses capture,
while preserving the current operator-guided flow as a reliable fallback. Automatic extraction must
reduce onboarding effort; it must not make uncertain model output authoritative.

The current Console workflow remains the baseline: freeze the POV, draw a crop, inspect several
angles, and explicitly confirm. The existing automatic capture API is experimental and is not the
demo enrollment path. Live testing showed that independent and temporal Cosmos prompts could
repeatedly ground laptop keyboards, screen content, accessories, or background for the ambiguous
label `keys`; recursive grounding on masked crops also converted five to seven usable first-pass
candidates into zero or one. Those failures must become evaluation fixtures rather than prompt-only
regressions.

### Proposed capability

Use a hierarchical extraction pipeline rather than running the same single-image prompt over a few
sparse frames:

1. Retain the complete bounded capture at the Media Gateway's actual sampled relay rate. The current
   Vision input is approximately 8 FPS, not the glasses' native 24–30 FPS; obtaining native-rate
   frames would require a separate reviewed transport change.
2. Run coarse temporal search over ordered frame batches to find intervals where the deliberately
   presented physical object persists or rotates.
3. Return to the original relay frames around those intervals for batched image grounding or a
   separately evaluated localization adapter.
4. Preserve each source frame with its proposed box so operators and tests can inspect localization
   failures; do not validate one crop and embed a different broader crop.
5. Apply blur, scale, occlusion, semantic-validity, and temporal-consistency checks.
6. Embed qualifying crops with C-RADIOv4-H, remove near-duplicates, and rank approximately five to
   eight diverse pending suggestions.
7. Let the operator enlarge and remove individual suggestions before explicit confirmation.
8. Activate the object and its references only after confirmation; otherwise roll back the empty
   object and all pending evidence.

An IPLOC-ID-style localization strategy, a dedicated grounding/segmentation model, improved Cosmos
prompting, or a hybrid may be evaluated. This backlog item does not approve an additional model,
runtime exception, or contract change. Any selected adapter must preserve the Linux ARM64/CUDA gate
and be recorded in the Vision model manifest.

### Safety and trust-boundary requirements

- Automatic suggestions are pending evidence, never active identity references.
- Manual crop selection remains available when automatic extraction abstains or produces weak
  suggestions.
- For `keys`, a valid reference must show the physical portable key set clearly enough to identify
  it; computer keyboard keys, piano keys, keypad buttons, screen images, hands, cords, rings, tags,
  and fobs alone are invalid.
- Full captures remain bounded Vision evidence. Only explicitly confirmed crops become durable
  object-scoped Memory evidence.
- Failure and deletion must invalidate both Memory rows and Vision's gallery cache immediately.
- Session identity and object identifiers remain trusted service context and are not inferred or
  selected by a model.
- The registration path must not change the canonical observation or personal-location contract.

### Evaluation and acceptance criteria

- Build a reviewed fixture set from consented captures covering portable keys near laptop
  keyboards, reflective metal, partial occlusion, hands, rings/fobs, screen images, cluttered desks,
  motion blur, and multiple viewing angles.
- Evaluate localization and crop validity separately; report candidate precision, capture-level
  success rate, abstention rate, and end-to-end extraction latency.
- Every accepted automated registration yields at least two enlarged, operator-approved views of
  the actual physical target.
- No unconfirmed suggestion appears in C-RADIO's active gallery or can resolve a live identity.
- Discard, failure, service restart, and partial Memory-write tests leave no object, view, crop file,
  embedding, or stale gallery entry.
- The Console supports source-frame/crop inspection, per-image removal, explicit confirmation, and
  immediate manual fallback.
- Extraction completes within an agreed demo latency budget on the GN100 without starving Cosmos
  event reasoning, C-RADIO matching, Speech, or Agent inference.
- Provider, consumer, fixture, API, Console, and physical GN100 tests pass, along with repository
  standards validation.

### Explicitly deferred decisions

- Whether automatic extraction reuses Cosmos 3 Nano or introduces a dedicated localization model.
- Whether the Gateway remains at its sampled relay rate or adds a bounded enrollment-only burst
  path.
- The pending-view storage location and expiration contract.
- Numerical precision/latency release thresholds after the evaluation set exists.
- Whether automatic extraction is exposed to voice registration before Console evaluation passes.

## B-003: Operator-guided registration in the smart-glasses app

**Status:** Deferred; Console registration is the demo baseline

**Priority:** Post-demo onboarding improvement
**Depends on:** Stable Android media session, a reviewed client authorization design, and the
confirmed-crop registration contract

### Product outcome

Let a wearer register a personal object directly from the RayNeo X3 Pro without requiring an Admin
Console operator. This is the smart-glasses equivalent of the working Console flow, not a return to
unreviewed automatic localization.

A wearer starts registration for an allowlisted label, centers the object inside an on-glasses
capture guide, records several distinct angles, reviews or removes pending views, and explicitly
confirms or cancels. Only confirmed views are eligible for C-RADIO quality/diversity processing and
durable registration.

### Proposed wearer flow

1. Start from an explicit command or control such as “register my keys.”
2. Show the canonical label and a bounded center guide in the glasses HUD.
3. Capture one wearer-approved crop or guided center region at a time.
4. Prompt the wearer to rotate the object and collect two to eight angles.
5. Show pending-view count and a review/remove control appropriate to the X3 interaction model.
6. Require an explicit final **Confirm registration** action; **Cancel** clears all pending images.
7. Submit the confirmed batch for geometric quality, C-RADIO embedding, and diversity selection.
8. Report the exact stored-view count or an honest retry reason through HUD and existing return
   audio.

The current no-barge-in limitation still applies. Registration prompts and confirmation audio must
use the existing return-audio suppression rules so spoken guidance cannot recursively trigger the
Agent.

### Architecture and security requirements

- Reuse the same server-side confirmed-crop semantics as Console manual enrollment; do not create a
  second incompatible object/view contract.
- Do not embed `VMA_INTERNAL_API_TOKEN`, Memory credentials, or another long-lived service secret in
  the Android application.
- Choose and document either a Gateway-authenticated registration facade or a short-lived,
  session-bound upload capability for Vision. The Gateway may authorize and relay bytes but must
  not perform object inference or become an object-state authority.
- Bind uploads to the active glasses session, canonical allowlisted label, bounded image count,
  content size, expiration, and one registration attempt. The client cannot choose an existing
  `object_id` to overwrite.
- Pending crops remain local to the app or in explicitly bounded server staging. They are not active
  C-RADIO references and are deleted on cancel, expiration, disconnect, or failed confirmation.
- Application Memory remains the sole durable owner of confirmed object identity references and
  location history.
- Keep full camera video on the existing LiveKit/Gateway path; registration must not introduce an
  unrestricted Android-to-Memory upload route.

### Interaction decisions to evaluate

- Touchpad, button, gesture, or voice controls for capture, remove, confirm, and cancel.
- Whether the X3 HUD can provide sufficient enlarged crop review or needs a paired-phone/Console
  review option for accessibility.
- A fixed center crop versus an adjustable on-glasses rectangle.
- Local crop encoding versus a session-bound server extraction from the already-relayed frame.
- How capture guidance remains usable while the wearer holds and rotates an object with one hand.

### Acceptance criteria

- A wearer can register an allowlisted object using only the physical glasses and trusted backend.
- At least two distinct wearer-approved views are required; no crop becomes active before final
  confirmation.
- The HUD shows label, pending count, capture feedback, remove/cancel state, and final stored-view
  count without claiming success early.
- Cancel, timeout, disconnect, duplicate confirmation, app restart, and partial upload leave no
  empty object, crop file, embedding, or stale Vision gallery entry.
- An unauthorized client, expired capability, wrong session, wrong label, oversized image, or ninth
  image is rejected before persistence.
- Console and glasses registrations produce the same Memory view schema and C-RADIO embedder
  provenance.
- Android unit/instrumentation tests, server provider/consumer tests, return-audio tests, and a
  physical X3/GN100 registration-and-recognition run pass.

### Explicitly deferred decisions

- Final X3 interaction controls and crop-adjustment UX.
- Gateway facade versus short-lived direct Vision capability.
- Whether a paired helper may approve crops on behalf of the wearer.
- Whether automatic extraction from B-002 is offered as an optional suggestion mode after its
  evaluation gate passes.

## Backlog maintenance

Add future items with an identifier, status, product outcome, dependencies, trust-boundary impact,
and testable acceptance criteria. Move an item into an implementation plan only after its owner and
scope are agreed.
