# Repository Agent Instructions

These instructions apply to the entire repository.

Before creating or modifying Python code, services, APIs, dependencies, tests, Dockerfiles, Compose configuration, CI, model adapters, or service documentation:

1. Read `.agents/skills/visual-memory-repo-standards/SKILL.md` completely.
2. Follow that skill for the remainder of the task.
3. Read every project document the skill identifies for the affected subsystem.
4. Run the repository standards validator and the affected service checks before reporting completion.

The repository default is Python 3.11, uv, FastAPI, Pydantic v2, Ruff, Pyright, and pytest. Every deployable Python service owns its `pyproject.toml`, `uv.lock`, tests, health endpoints, and Dockerfile.

Do not introduce `requirements.txt`, Poetry, Pipenv, an unpinned `pip install` workflow, another Python minor version, or another HTTP framework for project services. Existing files under `docs/spikes/` are historical experimental artifacts and do not establish production conventions.

Inference services may require platform-specific packages or an isolated Python/runtime version. Do not create or expand an exception without explicit user approval. Record an approved exception in the service's `standards-exception.toml`, keep the public service contract unchanged, and preserve the Linux ARM64/CUDA deployment gate.

Do not change the canonical observation or response contract without updating `docs/06-Data-Contract.md`, its fixtures, and both provider and consumer tests.
