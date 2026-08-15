"""This package must stay usable by a verifier that has no GPU and no torch.

Person 2's contract test harness imports `visual_memory_vision_contract` to
build fixtures and assert against `VerifierResult`. If this package pulled in
torch, numpy, or websockets, every consumer of a candidate event would inherit
them just to read a JSON envelope -- the same reasoning that keeps
`memory-contract` and `media-contract` independent of each other. Asserted
rather than trusted, the same way `services/media-gateway/tests/
test_domain_isolation.py` asserts the gateway's domain layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "visual_memory_vision_contract"
FORBIDDEN = (
    "torch",
    "numpy",
    "websockets",
    "fastapi",
    "starlette",
    "sqlalchemy",
    "cv2",
    "ultralytics",
)


def imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def package_sources() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if "__pycache__" not in path.parts)


def test_there_is_a_package_to_protect() -> None:
    assert package_sources(), f"no source modules found under {SRC}"


@pytest.mark.parametrize("source", package_sources(), ids=lambda path: path.name)
def test_no_module_imports_model_or_media_infrastructure(source: Path) -> None:
    offenders = imported_modules(source) & set(FORBIDDEN)

    assert not offenders, (
        f"{source.name} imports {', '.join(sorted(offenders))}. This package must "
        "stay importable by a verifier with no GPU and no media stack."
    )
