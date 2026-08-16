# Main Integration and Remote Assist Plan

## Status

Implemented on `integration/personal-memory-remote-assist`; automated validation is complete.
Physical GN100, glasses, and helper-app testing remains pending, and the integration branch must
not be merged into `main` until that manual gate is completed by the release owner. Hermes is
explicitly out of scope and is tracked in the [Product Backlog](17-Product-Backlog.md).

## Goal

Integrate `feature/personal-object-identity` with the remote-assist work merged into `main`, then
add a third Agent tool, `call_remote_assistant`, without weakening visual-memory grounding or
allowing wearer/helper speech to reach the language model while assistance is pending or active.

The resulting glasses experience has three bounded Agent actions:

1. `where_is(label)` reads authoritative Application Memory.
2. `start_registration(label)` starts the scripted personal-object enrollment workflow.
3. `call_remote_assistant()` creates a remote-human assistance request.

General questions may still receive concise direct answers from the local model. Hermes, web
search, `SOUL.md`, and long-term personal-profile memory are deferred.

## Non-goals

- Replacing Google ADK with Hermes.
- Changing the canonical observation, object-state, or Memory query-response contracts.
- Sending helper audio to Speech, the Agent, Vision, or Application Memory.
- Allowing the model to provide a session ID, helper identity, authorization credential, or
  arbitrary request payload.
- Adding barge-in.
- Adding an LLM-routed voice hang-up command while Agent audio ingestion is suppressed.

## Current branch integration evidence

The branches share base commit `8fdf616`. At the time this plan was written:

- `feature/personal-object-identity` has 28 feature-only commits.
- `origin/main` has 13 remote-assist-only commits from PR #1.
- A dry `git merge-tree` reports one textual conflict.
- Only three files were changed on both sides.

| File | Expected merge result |
|---|---|
| `apps/console/src/App.tsx` | Resolve the import conflict by retaining both `AssistPanel` and `EnrollmentPanel`; retain both tabs. |
| `apps/console/src/lib/contracts.ts` | Auto-merge the Assist types with overlay schema 1.4, identity, registration, and enrollment types; run formatting and review the combined unions. |
| `services/agent/src/agent/listener.py` | Retain model-owned intent routing, return-audio echo suppression, and registration behavior; add remote-assist suppression described below. |

There are also semantic documentation conflicts that Git cannot detect:

- The incoming media-relay documentation still describes the older wake/manual-trigger Agent
  gate, while the current feature forwards every completed transcript to Nemotron.
- The remote-assist documentation describes a server-enforced microphone-only helper grant, while
  the latest PR code temporarily uses an unrestricted publish grant and relies on the helper app
  not enabling its camera.
- The privacy documentation must describe the paired-helper authorization added by PR #1.

These must be reconciled before the integration PR is considered complete.

## Integration strategy

Do not rebase or force-push the deployed feature branch. Preserve it as a known-good rollback
point.

1. Fetch `origin/main` and verify both working trees are clean.
2. Create `integration/personal-memory-remote-assist` from
   `feature/personal-object-identity`.
3. Merge `origin/main` into the integration branch with a merge commit.
4. Resolve `App.tsx` by keeping both feature imports.
5. Review the automatic merges in `contracts.ts` and `listener.py` rather than accepting them only
   because Git reports no conflict.
6. Run the existing checks before adding the third tool. This separates merge regressions from
   new-tool regressions.
7. Implement the tool and audio gate in small tested commits.
8. Deploy the integration commit to the GN100 and pass the physical-glasses acceptance sequence.
9. Open one integration PR into `main`; do not merge until CI, GN100, helper-app, and rollback gates
   pass.

## Remote-assist state and Agent-audio gate

### State authority

The Media Gateway remains the authority for remote-assist lifecycle. The logical lifecycle is:

```text
idle -> requested -> accepted -> ended -> idle
          |                        ^
          +------ expired --------+
```

The request registry already owns `requested`, `accepted`, expiration, and helper-disconnect end
handling. Extend session summaries additively with an `assist_state` field:

```text
null | requested | accepted
```

`ended` remains a lifecycle/HUD event; after cleanup, session status returns `null`. Preserve the
existing `assist_active` field for compatibility with existing consumers. An older Gateway that
omits `assist_state` must still be handled by falling back to `assist_active`.

The request registry must sweep expired pending requests before reporting state. A pending request
that expires therefore reopens Agent audio just like an ended call.

### Audio policy

"Audio blocked" means no newly completed wearer transcript is submitted to the Agent backend or
language model. Closing the Agent's STT WebSocket also stops unnecessary transcription work. The
Gateway may continue receiving the wearer's encrypted LiveKit microphone track because the room
and remote-human call still need media.

| Assist state | Agent STT subscription | Transcript sent to model | Agent TTS |
|---|---:|---:|---:|
| `idle` | Open | Yes | Allowed |
| `requested` | Closed | No | Only one fixed request acknowledgement is allowed |
| `accepted` | Closed | No | No new Agent reply |
| ended/expired (`null`) | Reopened | Yes | Allowed |

Helper media remains out of the inference relay. The Gateway already admits inference tracks only
from the session's publisher identity, so the helper's microphone is heard in LiveKit but does not
reach Speech or the Agent. The Agent gate additionally prevents the wearer's side of the human
conversation from reaching the model.

### Race-free local suppression

Polling alone leaves a short interval between a successful request and the next session-list poll.
The Agent listener therefore needs a local, fail-closed suppression latch per session.

1. When `call_remote_assistant` returns a successful `requested` result, latch that session before
   any next transcript can call the backend.
2. Permit one fixed acknowledgement: "I've sent a request to your remote assistant."
3. End the current STT-consumer loop so queued transcripts are not processed.
4. The next Gateway poll confirms `requested` or `accepted` and keeps the session excluded.
5. Do not clear the latch because of a polling error, timeout, task restart, or model error.
6. Clear it only after an authoritative Gateway session summary reports no pending/active assist
   state. Then create a fresh STT subscription.

If an assist request is created from the Console rather than the Agent tool, the session poll sets
the same latch and cancels any in-flight listener task. Before publishing any ordinary model reply,
the listener must re-check suppression so a response already in flight cannot speak over a newly
started human-assist interaction.

Once the Gateway confirms the request side effect, that side effect wins even if the model's
post-tool finalization fails. The backend must return the fixed acknowledgement state rather than
retrying or reporting a generic LLM failure that could invite a duplicate request.

## Third tool: `call_remote_assistant`

### Tool adapter

Add a framework-independent bounded adapter at:

```text
services/agent/src/agent/tools/assist.py
```

It will:

- accept the trusted `session_id` from Agent request state;
- reject an absent session without making a network call;
- call `POST /v1/assist/{session_id}/request` through the configured Gateway base URL;
- authenticate with the internal service token;
- use the existing bounded request timeout;
- validate `request_id`, `session_id`, `state`, and expiry timestamps;
- map transport/authentication failures to an explicit Agent dependency error;
- never log transcript text, tokens, helper identity, or response bodies.

The Gateway endpoint is already idempotent while a request remains pending, so repeated tool calls
must return the existing request rather than extending its expiration or minting another ID.

### Model-facing tool

Register the ADK function as:

```text
call_remote_assistant()
```

It has no model-visible arguments. Session identity comes from the same trusted request state used
by `where_is` and `start_registration`.

Update the instruction so Nemotron calls it exactly once only for an explicit request for a remote
person, human helper, caregiver, or remote assistant. Ordinary requests containing the word
"help"—for example, "help me find my keys"—must not trigger a human call.

The tool result is authoritative only about request state. `requested` does not mean connected.
The wearer-facing acknowledgement is fixed rather than model-authored:

```text
I've sent a request to your remote assistant.
```

No name, arrival estimate, or connection claim may be added unless a future trusted contract
provides it.

### Backend result

Extend the internal draft result with explicit remote-assist state rather than inferring the side
effect from generated prose. Both the HTTP query path and hands-free listener must consume that
field consistently. The stub backend may use a bounded phrase matcher only for deterministic
offline tests; production intent routing remains Nemotron-owned.

## Expected implementation areas

### Media Gateway

- Expose current `assist_state` in session summaries without removing `assist_active`.
- Sweep request expiration when state is read.
- Preserve helper-disconnect cleanup and the `ended` HUD/event notification.
- Add provider/API tests for requested, accepted, ended, expired, and backward-compatible summary
  fields.

### Agent

- Add `AssistTool` and response validation.
- Register `call_remote_assistant` beside the two existing tools.
- Pass session identity out-of-band through trusted tool context.
- Parse the tool response into explicit draft state.
- Use the fixed acknowledgement.
- Add the immediate local suppression latch.
- Exclude both requested and accepted sessions from STT discovery.
- Drop queued transcripts and suppress in-flight ordinary replies.
- Reopen only after authoritative ended/expired state.
- Add metrics for requests, suppressed transcripts, and audio-gate transitions without transcript
  content.

### Console and helper app

- Retain both Assist and Enrollment tabs.
- Confirm the helper app receives an Agent-created request exactly as it receives a Console-created
  request.
- Confirm helper disconnect causes the Gateway to end the assist state.
- Keep camera disabled in the helper client. The temporary unrestricted server grant remains a
  documented security debt and must not be presented as server-enforced microphone-only access.

### Documentation

- Update `services/agent/README.md` from two tools to three.
- Reconcile `docs/12-Media-Relay-Contract.md` with current hands-free routing, helper authorization,
  actual publish-grant behavior, `assist_state`, and the requested/accepted audio gate.
- Update `docs/07-Privacy-and-Security.md` for helper credential scope and remote-human audio.
- Update Console/helper documentation and test commands as needed.
- Do not change `docs/06-Data-Contract.md`: remote-assist state is not an observation, evidence,
  object state, or Memory answer.

## Test plan

### Merge checks before new behavior

- Console typecheck and tests with both tabs present.
- Existing Agent suite, including all-transcript routing, registration, guard, thought filtering,
  bounded generation, and echo suppression.
- Existing Gateway and helper-app suites.
- Speech, Memory, and Vision checks to detect integration regressions.

### Agent tool tests

- Explicit remote-human request produces exactly one empty-argument tool call.
- Ordinary "help" and visual-memory questions do not call it.
- The model cannot provide or override `session_id`.
- Missing session returns a bounded failure and creates no request.
- Gateway authentication, timeout, invalid JSON, and unavailable failures are explicit.
- A repeated pending request remains idempotent.
- A successful tool side effect followed by model-finalization failure still produces the fixed
  acknowledgement and suppression state.

### Audio-gate tests

- `idle`: transcripts reach the backend.
- `requested`: the triggering transcript may create the request; subsequent and already-queued
  transcripts do not reach the backend.
- `accepted`: no STT listener exists and no Agent reply is published.
- Console-created request suppresses an already-running Agent listener.
- A poll failure does not reopen audio.
- Pending expiration reopens a fresh STT listener.
- Helper disconnect emits `ended`, clears Gateway state, and reopens a fresh STT listener.
- Return-audio echo cooldown still works after assist ends.
- Reconnects do not retain a stale local suppression latch after authoritative ended state.

### Contract and security tests

- Session summaries expose additive `assist_state` and preserve `assist_active`.
- Helper audio never enters the inference relay.
- Device/helper/internal credentials retain their intended scopes.
- Logs contain no transcripts, tokens, or raw request bodies.
- The helper camera remains disabled client-side; the unrestricted publish-grant debt remains
  visible through its existing strict xfail/security test.

## Physical GN100 acceptance sequence

1. Deploy the merged integration commit with the existing local Nemotron, Cosmos, C-RADIO,
   Parakeet, and Kokoro configuration.
2. Connect the glasses and helper app through the real LiveKit/TURN path.
3. Ask a general question and a `where_is` question to establish normal Agent audio.
4. Say, "Call my remote assistant."
5. Confirm one fixed acknowledgement and one incoming helper request.
6. Speak again while the request is ringing; confirm no new transcript reaches Nemotron and no
   Agent reply is produced.
7. Accept from the helper app; confirm `assist_state=accepted` and no Agent STT subscription.
8. Hold a two-way conversation; confirm neither wearer nor helper speech appears in Agent/Speech
   query logs, HUD transcripts, or TTS.
9. Disconnect the helper; confirm `ended`, then `assist_state=null`.
10. Ask another general question; confirm a fresh STT subscription and normal reply.
11. Repeat with an unanswered request and wait for expiration; confirm listening resumes.
12. Run a reconnect cycle and verify no stale suppression or duplicate request.

## Required validation

For each affected Python service:

```text
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Also run Console and helper-app installation, typecheck, and test commands; then run:

```text
python .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
```

The integration is not complete until the physical GN100/glasses/helper sequence passes. A mocked
or CPU-only test does not prove LiveKit, TURN, audio suppression, or ARM64 behavior.

## Rollback

- Preserve `feature/personal-object-identity` and the current GN100 commit as the known-good visual
  memory deployment.
- Preserve the previous Compose/environment files and database/evidence volumes.
- If remote-assist integration fails, deploy the known-good feature commit without deleting Memory
  or model caches.
- Do not partially enable `call_remote_assistant` if the Gateway cannot expose authoritative
  requested/accepted/end state; disable the tool instead.

## Completion criteria

- The combined branch has no unresolved textual or semantic documentation conflicts.
- All three tools receive session identity only from trusted request state.
- A successful remote-assist request immediately blocks further model-bound audio.
- Requested and accepted sessions remain blocked across polling errors and reconnects.
- Helper disconnect or pending expiration authoritatively reopens Agent audio.
- Helper audio never enters inference.
- Visual Memory remains authoritative for personal-object locations.
- Full repository, service, UI, GN100, glasses, and helper-app gates pass.
