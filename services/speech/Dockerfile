# Build context is the REPOSITORY ROOT, not this directory, because the service
# depends on ../../packages/media-contract by relative path. Build with:
#   docker build -f services/speech/Dockerfile .
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser

COPY --from=uv /uv /uvx /bin/

# The local path dependency must sit at the same relative offset as in
# pyproject.toml: /app/services/speech/../../packages/media-contract.
COPY packages/media-contract /app/packages/media-contract

WORKDIR /app/services/speech
COPY services/speech/pyproject.toml services/speech/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/speech/src ./src
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 8080

# The venv's uvicorn directly, not `uv run`. uv wants a writable cache
# directory even with --no-sync, which fails outright under a read-only root
# filesystem -- and a read-only root is the correct way to run a service that
# writes nothing.
#
# --workers 1: `mlx`/model backends are never installed in this image (Darwin
# only -- see pyproject.toml's `mlx` group), so this container always runs the
# stub STT/TTS backends. A second worker would just double that for no
# benefit; once a real Linux/CUDA backend exists, model memory is the actual
# reason to keep this at 1.
CMD ["/app/services/speech/.venv/bin/uvicorn", "speech.main:app", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-graceful-shutdown", "20"]
