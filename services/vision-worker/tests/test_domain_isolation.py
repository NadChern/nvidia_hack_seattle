"""The domain layer must not depend on a model runtime, the media stack, or
the web framework.

Keeping `domain/` free of torch, fastapi, and friends is what lets the rules
that decide whether an object is at rest or moving be tested in milliseconds,
with no GPU, no relay connection, and no ASGI app. That is easy to break with
one convenient import, so it is asserted rather than trusted -- the same
discipline `services/media-gateway` and `services/application-memory` each
carry for their own domain layers.

Per the plan (`docs/08-Development-and-Deployment.md`:30 -- "code outside a
model adapter must not depend on MLX, PyTorch, CUDA, operating-system paths,
or a checkpoint layout"), this extends beyond `domain/`: `detect/`, `depth/`,
`track/`, and `pose/` are the only packages in this service allowed to import
a model runtime. Everything else -- `api/`, `consume/`, `emit/`, `evidence/`,
`verify/` -- must stay importable on a machine with no GPU, which is what
makes the `ci` and `dev-macos` profiles real rather than aspirational.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "vision_worker"
DOMAIN = SRC / "domain"

#: Packages allowed to import a model runtime or a media/vision library --
#: everything else in this service must stay free of them.
MODEL_ADAPTER_PACKAGES = {"detect", "depth", "track", "pose"}

DOMAIN_FORBIDDEN = ("torch", "numpy", "cv2", "ultralytics", "fastapi", "starlette", "websockets")
SERVICE_FORBIDDEN = ("torch", "cv2", "ultralytics")


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


def non_adapter_sources() -> list[Path]:
    """Every module in this service outside the four model-adapter packages."""
    sources = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC)
        if relative.parts and relative.parts[0] in MODEL_ADAPTER_PACKAGES:
            continue
        sources.append(path)
    return sorted(sources)


def test_there_is_a_domain_layer_to_protect() -> None:
    """Guards against this suite passing vacuously if the layout changes."""
    assert domain_sources(), f"no domain modules found under {DOMAIN}"


@pytest.mark.parametrize("source", domain_sources(), ids=lambda path: path.name)
def test_domain_modules_import_no_infrastructure(source: Path) -> None:
    offenders = imported_modules(source) & set(DOMAIN_FORBIDDEN)

    assert not offenders, (
        f"{source.name} imports {', '.join(sorted(offenders))}. The domain layer "
        "must stay testable without a GPU, a relay connection, or an ASGI app."
    )


@pytest.mark.parametrize(
    "source", non_adapter_sources(), ids=lambda path: str(path.relative_to(SRC))
)
def test_non_adapter_modules_import_no_model_runtime(source: Path) -> None:
    """`detect/`, `depth/`, `track/`, and `pose/` are the only packages allowed
    to depend on a model runtime. Everything else must stay importable on a
    machine with no GPU -- the `ci` and `dev-macos` profiles depend on it."""
    offenders = imported_modules(source) & set(SERVICE_FORBIDDEN)

    assert not offenders, (
        f"{source.relative_to(SRC)} imports {', '.join(sorted(offenders))} outside "
        f"{sorted(MODEL_ADAPTER_PACKAGES)}. Only the model-adapter packages may "
        "depend on a model runtime."
    )
