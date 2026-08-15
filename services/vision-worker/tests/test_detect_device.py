"""Which torch device YOLOE loads onto.

Unmarked, unlike the rest of `test_detect_yoloe.py`: `_select_device` takes
the torch module as an argument rather than importing it, so the whole
decision is testable against fakes with no `models` extra, no checkpoint and
no GPU. That seam is the point -- the branch that matters most is the Apple
Silicon one, which no machine in this project's CI or on this developer's desk
can execute.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vision_worker.detect.yoloe import _select_device


def _torch(*, cuda: bool, mps: bool | None) -> Any:
    """A stand-in for the torch module.

    `mps=None` models a build with no `torch.backends.mps` attribute at all,
    which is a different failure from one that has it and reports False.
    """
    backends = SimpleNamespace()
    if mps is not None:
        backends.mps = SimpleNamespace(is_available=lambda: mps)
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=backends,
    )


def test_cuda_wins_when_present() -> None:
    assert _select_device(_torch(cuda=True, mps=False)) == "cuda"


def test_cuda_wins_over_mps() -> None:
    # Not a real machine today, but the ordering is deliberate: CUDA is the
    # deploy target and the only path with measured numbers behind it.
    assert _select_device(_torch(cuda=True, mps=True)) == "cuda"


def test_mps_when_there_is_no_cuda() -> None:
    assert _select_device(_torch(cuda=False, mps=True)) == "mps"


def test_cpu_when_metal_is_unavailable() -> None:
    # An Intel Mac, or a macOS too old for Metal: the attribute exists and
    # answers False.
    assert _select_device(_torch(cuda=False, mps=False)) == "cpu"


def test_cpu_when_torch_has_no_mps_backend_at_all() -> None:
    # A torch built without it. Reading the attribute directly would raise
    # AttributeError here and fail startup on a machine that simply has no
    # Metal, which is not an error.
    assert _select_device(_torch(cuda=False, mps=None)) == "cpu"


def test_an_explicit_device_overrides_detection() -> None:
    # The escape hatch for MPS falling back to CPU on an op Metal has no
    # kernel for. It must win even where CUDA is present and working, or it
    # is not an override.
    assert _select_device(_torch(cuda=True, mps=True), "cpu") == "cpu"
    assert _select_device(_torch(cuda=False, mps=False), "mps") == "mps"
