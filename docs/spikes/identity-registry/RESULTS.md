# Personal-object registry Phase-1 results

## Decision

**Adopt.** The session-independent registry, float32 view storage, monotonic cache version, and
four cross-session repairs pass against SQLite. Continue to fixture-backed Vision identity.

## Record

- Date: 2026-08-15
- Owner: Visual Memory Assistant maintainers
- Branch: `feature/personal-object-identity`
- Machine: WSL2 Linux x86_64, local SQLite
- Scope: `packages/memory-contract` and `services/application-memory`

## Correctness gates

- `test_registered_object_survives_session_delete_and_answers_in_a_new_session`: pass.
- `test_preresolved_object_id_still_receives_lifecycle_signals`: pass.
- Vector round trip: little-endian float32 bytes compare exactly after SQLite `LargeBinary`.
- Registry create and view writes are idempotent; deletion removes rows and crop files.
- `alembic upgrade head` followed by `alembic check`: no new upgrade operations detected.
- Registered crops use their own root and are not reachable from the session retention sweeper.

## Demo-scale gallery benchmark

Command, from `services/application-memory`:

```bash
uv run python scripts/benchmark_registry.py
```

Result at 30 objects x 4 views x two 1,152-float vectors, 25 reads:

```text
objects=30 views=120 dim=1152 iterations=25
gallery latency ms p50=26.24 p95=31.89; vector bytes/object=36864
```

The p95 gate is `<50 ms`; measured p95 is 31.89 ms. Vector storage is 36,864 bytes per object,
plus four bounded JPEG crops and row metadata. This local set is sufficient for the demo-scale
cache gate, not a multi-user database benchmark.
