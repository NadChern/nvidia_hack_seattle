"""The domain layer must not depend on LiveKit.

Keeping the sampler, dimension guard, and epoch rules free of the SDK is what
lets them be tested without a server or a network. That is easy to break by
adding one convenient import, so it is asserted rather than trusted.
"""

import ast
import sys
from pathlib import Path

import pytest

DOMAIN = Path(__file__).resolve().parent.parent / "src" / "media_gateway" / "domain"
FORBIDDEN = ("livekit", "av", "fastapi")


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


@pytest.mark.parametrize("source", sorted(DOMAIN.glob("*.py")), ids=lambda p: p.name)
def test_domain_module_avoids_transport_dependencies(source: Path) -> None:
    offending = imported_modules(source).intersection(FORBIDDEN)

    assert not offending, (
        f"{source.name} imports {sorted(offending)}; the domain layer stays "
        "transport-independent so it can be tested without a LiveKit server"
    )


def test_importing_the_domain_does_not_pull_in_livekit() -> None:
    """Catches a transitive import the AST scan would miss."""
    for name in list(sys.modules):
        if name.startswith(("media_gateway.domain", "livekit")):
            del sys.modules[name]

    import media_gateway.domain.epoch  # noqa: F401
    import media_gateway.domain.metrics  # noqa: F401
    import media_gateway.domain.sampling  # noqa: F401
    import media_gateway.domain.session  # noqa: F401

    assert not [name for name in sys.modules if name.startswith("livekit")]
