# Build from the repository root because the service has an editable path
# dependency on packages/memory-contract:
#   docker build -f services/agent/Dockerfile .
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser
COPY --from=uv /uv /uvx /bin/

# Preserve the relative source mapping in pyproject.toml.
COPY packages/memory-contract /app/packages/memory-contract
WORKDIR /app/services/agent
COPY services/agent/pyproject.toml services/agent/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/agent/src ./src
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 8086

# Run the installed executable directly so a read-only container needs no uv
# cache. One worker keeps the in-memory conversational session store coherent.
CMD ["/app/services/agent/.venv/bin/uvicorn", "agent.main:app", \
     "--host", "0.0.0.0", "--port", "8086", \
     "--workers", "1", "--timeout-graceful-shutdown", "20"]
