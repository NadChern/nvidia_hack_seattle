# __SERVICE_NAME__

Repository-standard `__SERVICE_KIND__` service for the Visual Memory Assistant.

## Development

```text
uv sync --frozen --all-groups
uv run uvicorn __PACKAGE_NAME__.main:app --reload
```

## Checks

```text
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

The generator records the owning person or team in `service.toml`.
