---
name: visual-memory-repo-standards
description: Enforce the Visual Memory Assistant repository engineering standard. Use for any task that creates or modifies Python code, a service, FastAPI route, dependency, pyproject.toml, uv.lock, test, Dockerfile, Compose configuration, CI workflow, model adapter, health endpoint, shared contract, or repository scaffold.
---

# Visual Memory Repo Standards

Apply the repository's golden path while preserving platform-specific inference adapters and the GN100 deployment gate.

## Required reading

Before editing:

1. Read `../../../11-Engineering-Standards.md` completely.
2. For APIs, observations, memory, or evidence, read `../../../06-Data-Contract.md`.
3. For Docker, CI, Compose, model packaging, or releases, read `../../../08-Development-and-Deployment.md`.
4. For a model or unresolved runtime, read `../../../02-Model-Landscape.md` and `../../../09-Spike-Plan.md`.
5. Obey the nearest `AGENTS.md` for any nested scope.

Do not continue from memory or a copied summary when one of these documents governs the change.

## Classify the change

- Use the default Python 3.11/FastAPI template for application APIs, memory/query services, control APIs, and ordinary workers.
- Keep LiveKit media on WebRTC; use FastAPI only for control and health surfaces.
- Isolate heavy model dependencies in the owning inference service and its own uv lock.
- Preserve one adapter contract across MLX, Windows PyTorch/CUDA, and GN100 Linux ARM64/CUDA implementations.
- Treat spikes as disposable evidence. Do not copy a spike's `requirements.txt`, environment, or shortcuts into a service.

## Create a service

Prefer the generator:

```text
python .agents/skills/visual-memory-repo-standards/scripts/new_service.py <service-name> --kind application --owner <owner>
```

Use `--kind worker` for a worker with a FastAPI control/health surface and `--kind inference` for a model adapter service. For inference, replace the default base image only after the GN100 runtime is selected and record the model manifest.

Do not hand-create an alternative layout when the template covers the service.

## Modify an existing service

1. Preserve its public contracts or update providers, consumers, fixtures, and documentation together.
2. Add dependencies with `uv add`; add development tools with `uv add --dev`.
3. Commit the resulting `pyproject.toml` and `uv.lock`.
4. Keep model imports behind adapters.
5. Keep route handlers thin and bounded.
6. Add or update unit and contract tests.

Do not use `pip install`, `requirements.txt`, Poetry, Pipenv, or Conda for production project services.

## Handle an exception

Do not create or expand a standards exception without explicit user approval.

For an approved upstream model/runtime constraint:

1. Minimize the exception to named validator rule IDs.
2. Create `standards-exception.toml` with `rules`, `reason`, `owner`, `approved_by`, and `expires`.
3. Preserve uv locking, APIs, tests, health checks, privacy controls, and physical GN100 validation.
4. Report the exception prominently in the final response.

## Validate

Run from every affected service:

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

For deployment work, also run the tests and physical GN100 gates required by `08-Development-and-Deployment.md`. Do not claim ARM64/CUDA compatibility from an x86 build or mocked CI result.

## Completion report

State:

- affected services and contracts;
- dependency and lockfile changes;
- checks run and their results;
- any approved exception;
- whether physical GN100 validation remains pending.
