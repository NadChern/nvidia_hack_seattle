FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.11.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

USER appuser
EXPOSE 8080

CMD ["uv", "run", "--no-sync", "uvicorn", "__PACKAGE_NAME__.main:app", "--host", "0.0.0.0", "--port", "8080"]
