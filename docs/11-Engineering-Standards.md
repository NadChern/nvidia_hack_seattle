# Engineering Standards

This document is the source of truth for repository-wide implementation conventions. The repository skill at `.agents/skills/visual-memory-repo-standards/` turns these rules into an agent workflow, service scaffold, and validator.

## Standard stack

| Area | Required default |
|---|---|
| Language | CPython 3.11, expressed as `>=3.11,<3.12` |
| Project and dependency management | uv with a committed `uv.lock` per deployable service |
| HTTP and health APIs | FastAPI and Uvicorn |
| Validation and settings | Pydantic v2 and `pydantic-settings` |
| HTTP client | httpx with explicit timeouts |
| Testing | pytest with unit, contract, and fixture coverage as applicable |
| Formatting and linting | Ruff formatter and linter |
| Type checking | Pyright |
| Database access | SQLAlchemy 2 and Alembic where relational persistence is owned |
| Logging | Structured, redacted logs with request, session, and observation identifiers |
| Deployment | Linux ARM64 Dockerfile, non-root process, frozen uv install, health checks, and graceful shutdown |

Use current compatible package releases resolved into the lockfile. Review dependency upgrades intentionally; do not depend on floating installations during startup or deployment.

## Repository layout

```text
services/
  <service-name>/
    .python-version
    pyproject.toml
    uv.lock
    Dockerfile
    service.toml
    src/<package_name>/
    tests/
packages/
  contracts/
.agents/
  skills/visual-memory-repo-standards/
```

Every deployable service is an independent uv project and owns its lockfile. This avoids forcing incompatible MLX, PyTorch, CUDA, FlashInfer, or vendor dependencies into a single universal resolution.

Shared packages may contain contracts, generated clients, and small infrastructure utilities. They must not import service internals or model runtimes.

## Service rules

Every Python service must:

- use the `src/` package layout;
- expose `/health/live` and `/health/ready`;
- validate configuration at startup;
- use typed request, response, and internal boundary models;
- return explicit unavailable, invalid, unauthorized, and ambiguous errors;
- set timeouts and bounds for network calls, queues, media buffers, and model work;
- emit structured logs without raw media, transcripts, secrets, tokens, or arbitrary evidence paths;
- support deterministic startup, readiness, and graceful shutdown;
- keep business logic outside FastAPI route handlers;
- include unit tests and shared contract fixtures for every owned interface.

Use FastAPI for HTTP, control, and health surfaces. Streaming media remains on LiveKit/WebRTC; FastAPI does not replace the media plane.

## Model and platform boundary

Application code must depend on a model adapter interface, not MLX, CUDA, operating-system paths, or checkpoint layouts.

The following variation is expected:

- Apple-silicon development may use MLX-backed adapters.
- Windows development may use native PyTorch/CUDA-backed adapters.
- GN100 deployment uses tested Linux ARM64/CUDA adapters.
- Heavy inference services may need separate Python or package constraints when the upstream runtime makes Python 3.11 impossible.

An inference exception requires explicit approval and a `standards-exception.toml` containing:

```toml
rules = ["python-version"]
reason = "Upstream runtime constraint and evidence"
owner = "Named owner"
approved_by = "Named approver"
expires = "YYYY-MM-DD"
```

An exception changes only the minimum necessary runtime rule. It does not waive uv locking, API contracts, tests, health behavior, privacy controls, or the physical GN100 release gate.

## Required commands

Run from each affected service:

```text
uv sync --frozen --all-groups
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Run from the repository root:

```text
python .agents/skills/visual-memory-repo-standards/scripts/validate_repo.py
```

CI runs the same checks. A service is not complete when it passes locally with an uncommitted or stale lockfile.

## Agent adoption

`AGENTS.md` requires agents to load the repository skill before implementation work. The skill provides:

- the required reading and decision workflow;
- a service generator based on the standard template;
- deterministic validation of service structure and dependencies;
- exception handling rules;
- completion checks.

This is guidance and automation, not a security boundary. CI is the final enforcement point for changes created by agents or humans.

## Ownership

The release owner owns the common standard, CI workflow, and scaffold. Each service owner owns compliance inside their service.

Changes to the standard require review from the release owner and each affected service owner. Changes to data contracts additionally require provider and consumer fixtures as defined in [Team Split](05-Team-Split.md).
