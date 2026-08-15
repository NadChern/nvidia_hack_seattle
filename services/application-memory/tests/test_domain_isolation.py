"""The domain layer must not depend on the database or the web framework.

Keeping the reducer free of SQLAlchemy and FastAPI is what lets the rules that
decide what the assistant may claim be tested in milliseconds, with no engine,
no migration, and no ASGI app. That is easy to break with one convenient
import, so it is asserted rather than trusted.

The media gateway carries the same test for the same reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DOMAIN = Path(__file__).resolve().parent.parent / "src" / "application_memory" / "domain"
FORBIDDEN = ("sqlalchemy", "alembic", "fastapi", "starlette", "httpx")


def imported_modules(source: Path) -> set[str]:
    """Return the top-level module names a file imports."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def domain_sources() -> list[Path]:
    return sorted(path for path in DOMAIN.rglob("*.py") if "__pycache__" not in path.parts)


def test_there_is_a_domain_layer_to_protect() -> None:
    """Guards against this suite passing vacuously if the layout changes."""
    assert domain_sources(), f"no domain modules found under {DOMAIN}"


@pytest.mark.parametrize("source", domain_sources(), ids=lambda path: path.name)
def test_domain_modules_import_no_infrastructure(source: Path) -> None:
    offenders = imported_modules(source) & set(FORBIDDEN)

    assert not offenders, (
        f"{source.name} imports {', '.join(sorted(offenders))}. The domain layer "
        "must stay testable without a database or an ASGI app."
    )
