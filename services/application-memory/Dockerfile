# Build context is the REPOSITORY ROOT, not this directory, because the service
# depends on ../../packages/memory-contract by relative path. Build with:
#   docker build -f services/application-memory/Dockerfile .
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser

COPY --from=uv /uv /uvx /bin/

# The local path dependency must sit at the same relative offset as in
# pyproject.toml: /app/services/application-memory/../../packages/memory-contract.
COPY packages/memory-contract /app/packages/memory-contract

WORKDIR /app/services/application-memory
COPY services/application-memory/pyproject.toml services/application-memory/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/application-memory/src ./src
COPY services/application-memory/alembic.ini ./
RUN uv sync --frozen --no-dev

# The database and evidence live on mounted volumes, so the image itself needs
# nothing writable and can run with a read-only root filesystem.
USER appuser
EXPOSE 8081

# The venv's uvicorn directly, not `uv run`. uv wants a writable cache even
# with --no-sync, which fails outright under a read-only root filesystem, and
# nothing about uv is needed once the environment is built.
#
# --workers 1: the reducer recomputes an object's timeline on write, and two
# workers would interleave those recomputations against one SQLite file.
CMD ["/app/services/application-memory/.venv/bin/uvicorn", "application_memory.main:app", \
     "--host", "0.0.0.0", "--port", "8081", \
     "--workers", "1", "--timeout-graceful-shutdown", "20"]
