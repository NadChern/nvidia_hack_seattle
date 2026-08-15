# Memory Service (`application-memory`)

The trusted source of truth. It decides what the assistant is allowed to claim.

The demo is impressive not because the system finds keys, but because of what it says when it cannot:

> *I last confirmed the keys on the living room coffee table at 10:42, but they were picked up afterward and I have not confirmed a new location.*

Producing that sentence — rather than confidently repeating a stale location — is this service's entire job.

## See it work

```bash
cd services/application-memory && uv run python scripts/seed_demo.py
```

No camera, no vision model, no network. Validated observations go in, the reducer runs, the answer comes out. This is **fallback levels 4 and 5** from [Hackathon Stack](../../docs/03-Hackathon-Stack.md), rehearsable today.

The script exits non-zero if the answer comes back `confirmed`, because the scenario ends with the keys being picked up — so it is a rehearsal and a regression test at once. `--confirmed` runs the undisturbed scenario, which may legitimately answer `confirmed`.

## Run it

```bash
cd services/application-memory && uv run uvicorn application_memory.main:app --port 8081
```

Run `uv` from **inside this directory**; from the repository root it resolves a different environment and fails with `ModuleNotFoundError`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/observations` | Vision submits what it saw |
| `POST /v1/query` | "Where are my keys?" |
| `POST /v1/lifecycle` | The gateway reports a track or session ending |
| `POST /v1/evidence` · `GET /v1/evidence/{id}` | Frames, verified by digest |
| `POST /v1/sessions` · `DELETE /v1/sessions/{id}` | Registration and deletion |
| `GET /v1/status` | Counts and the promotion thresholds |

## How it works

**Observations are immutable; state is derived.** Every write replays that object's whole timeline in `(occurred_at, id)` order rather than patching a stored state. Duplicate delivery, out-of-order arrival, and replay after a restart therefore produce identical state *by construction* — two of the nine required reducer tests pass because of this choice rather than because of code that handles those cases.

It also means deletion works: there is no second place a deleted memory can survive.

**The reducer is a pure function.** No database, no clock, no I/O — `tests/test_domain_isolation.py` asserts that mechanically. The rules that decide what the assistant may claim run in a fraction of a second with no engine and no migration.

**A confirmed answer requires evidence that can be loaded, not merely referenced.** Retention deletes files while rows survive, and a row pointing at a deleted frame looks exactly like a valid one. The query path checks the filesystem and downgrades to `last_confirmed_only` when the frame is gone — making [docs/04](../../docs/04-Evaluation-Plan.md)'s *unsupported confident answer* impossible rather than merely measured.

**Identity resolution is deliberately dumb**: exact label match within `(session_id, media_epoch_id, track_id)`. The epoch is part of the key, so a tracker that restarts its numbering after a reconnect cannot silently merge two objects. An honest `ambiguous_object` beats a confident wrong merge.

## Boundary decisions

**SQLite, not a database container.** [docs/08](../../docs/08-Development-and-Deployment.md) permits combining logical services *"when hackathon simplicity or latency justifies it"*. Dropping the container removes a volume, credentials, and a readiness dependency from day one on the GN100 — the only day that hardware exists. SQLAlchemy 2 and Alembic are used as [docs/11](../../docs/11-Engineering-Standards.md) requires, so switching to Postgres is a URL change plus a migration run.

**Wording is generated from a template, not by a model.** This is the last place a wrong claim can escape, and a model cannot be constrained to preserve an invalidation clause. The conversational layer may shorten the text but must keep `answer_status`, the uncertainty, and any invalidation.

**`--workers 1`.** The reducer recomputes a timeline on write; two workers would interleave recomputations against one SQLite file.

## Checks

```bash
cd services/application-memory && uv sync --frozen --all-groups && uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest
```

The reducer alone, with no infrastructure:

```bash
cd services/application-memory && uv run pytest tests/test_reducer.py
```

Schema changes:

```bash
cd services/application-memory && uv run alembic revision --autogenerate -m "what changed"
```

## Related

- [Data Contract](../../docs/06-Data-Contract.md) — the canonical contract this service implements
- [Evaluation Plan](../../docs/04-Evaluation-Plan.md) — the reducer cases and the metrics
- [Privacy and Security](../../docs/07-Privacy-and-Security.md) — retention and deletion scope
- `packages/memory-contract` — what Vision depends on to produce observations
