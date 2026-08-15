# Build context is the REPOSITORY ROOT, not this directory, because the service
# depends on ../../packages/media-contract by relative path. Build with:
#   docker build -f services/media-gateway/Dockerfile .
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser

COPY --from=uv /uv /uvx /bin/

# The local path dependency must sit at the same relative offset as in
# pyproject.toml: /app/services/media-gateway/../../packages/media-contract.
COPY packages/media-contract /app/packages/media-contract

WORKDIR /app/services/media-gateway
COPY services/media-gateway/pyproject.toml services/media-gateway/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/media-gateway/src ./src
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 8080

# The venv's uvicorn directly, not `uv run`. uv wants a writable cache
# directory even with --no-sync, which fails outright under a read-only root
# filesystem -- and a read-only root is the correct way to run a service that
# writes nothing. Nothing about uv is needed once the environment is built.
#
# --workers 1 is load-bearing: a second worker would open a second LiveKit
# subscription on the same room, doubling decode and splitting relay state.
#
# --no-access-log: a browser cannot set headers on a WebSocket, so the console
# authenticates in the query string, and an access log records the request line
# verbatim. docs/07 forbids tokens in logs. The redaction filter also strips
# `?token=`, but not emitting the line at all is the boundary that does not
# depend on a regex staying correct. The service's own structured logs already
# carry request outcomes.
CMD ["/app/services/media-gateway/.venv/bin/uvicorn", "media_gateway.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log", \
     "--workers", "1", "--loop", "asyncio", "--timeout-graceful-shutdown", "20"]
