# Build context is the REPOSITORY ROOT, not this directory, because the service
# depends on ../../packages/{media,memory,vision}-contract by relative path.
# Build with: docker build -f services/vision-worker/Dockerfile .
#
# CPU-only, and deliberately so: this image runs detector_kind=fixture, which
# needs no GPU, and stays small enough that a teammate without one can still
# bring the whole stack up. The real models live in `Dockerfile.cuda`, selected
# by `compose.gpu.yaml` -- the two images are not interchangeable.
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser

COPY --from=uv /uv /uvx /bin/

# Each local path dependency must sit at the same relative offset as in
# pyproject.toml: /app/services/vision-worker/../../packages/<name>.
COPY packages/media-contract /app/packages/media-contract
COPY packages/memory-contract /app/packages/memory-contract
COPY packages/vision-contract /app/packages/vision-contract

WORKDIR /app/services/vision-worker
COPY services/vision-worker/pyproject.toml services/vision-worker/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/vision-worker/src ./src
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 8082

# The venv's uvicorn directly, not `uv run`. uv wants a writable cache
# directory even with --no-sync, which fails outright under a read-only root
# filesystem -- and a read-only root is the correct way to run a service that
# writes nothing durable. Nothing about uv is needed once the environment is
# built.
#
# --workers 1 is load-bearing, for the same class of reason it is in
# media-gateway: Pipeline holds one relay connection and all of one epoch's
# state (TrackRegistry, EvidenceRing, the tracker) in-process as a singleton.
# A second worker would open a second relay connection and independently
# process the same video stream, producing duplicate candidates and
# duplicate writes to Memory.
#
# --no-access-log: the overlay socket authenticates in the query string because
# a browser cannot set headers on a WebSocket, and an access log would record
# that token verbatim. docs/07 forbids tokens in logs.
CMD ["/app/services/vision-worker/.venv/bin/uvicorn", "vision_worker.main:app", \
     "--host", "0.0.0.0", "--port", "8082", "--no-access-log", \
     "--workers", "1", "--loop", "asyncio", "--timeout-graceful-shutdown", "20"]
