"""The Phase-0 identity probe's balanced protocol and model smoke test."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_identity.py"
_SPEC = importlib.util.spec_from_file_location("probe_identity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe_identity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe_identity
_SPEC.loader.exec_module(probe_identity)

RadioEmbedder = probe_identity.RadioEmbedder
Trial = probe_identity.Trial
choose_threshold = probe_identity.choose_threshold
discover_dataset = probe_identity.discover_dataset
gate_metrics = probe_identity.gate_metrics
roc_auc = probe_identity.roc_auc


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def test_flat_keys_dataset_is_discovered_as_one_same_class(tmp_path: Path) -> None:
    for instance, color in (("keys_1", (255, 0, 0)), ("keys_2", (0, 255, 0))):
        _image(tmp_path / instance / "reference" / "1.jpg", color)
        _image(tmp_path / instance / "reference" / "2.jpg", color)
        _image(tmp_path / instance / "query_realistic" / "1.jpg", color)

    objects = discover_dataset(tmp_path)

    assert [item.instance_id for item in objects] == ["keys_1", "keys_2"]
    assert {item.label for item in objects} == {"keys"}


def test_a_single_instance_cannot_fake_same_class_rejection(tmp_path: Path) -> None:
    _image(tmp_path / "wallet_1" / "reference" / "1.jpg", (255, 0, 0))
    _image(tmp_path / "wallet_1" / "reference" / "2.jpg", (255, 0, 0))
    _image(tmp_path / "wallet_1" / "query_clean" / "1.jpg", (255, 0, 0))

    with pytest.raises(ValueError, match="at least 2 physical instances"):
        discover_dataset(tmp_path)


def test_threshold_selection_prioritizes_the_negative_accuracy_floor() -> None:
    trials = (
        Trial("keys_1", "keys_1", Path("p1"), "all", True, 0.90),
        Trial("keys_2", "keys_2", Path("p2"), "all", True, 0.70),
        Trial("keys_1", "keys_2", Path("n1"), "all", False, 0.80),
        Trial("keys_2", "keys_1", Path("n2"), "all", False, 0.60),
    )

    threshold = choose_threshold(trials)
    metrics = gate_metrics(trials, threshold)

    assert threshold > 0.80
    assert metrics.negative_accuracy == 1.0
    assert metrics.positive_accuracy == 0.5
    assert roc_auc(trials) == pytest.approx(0.75)


@pytest.mark.models
def test_radio_model_embeds_one_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.jpg"
    _image(image_path, (128, 64, 32))

    embedder = RadioEmbedder(device="cuda", batch_size=1)
    vectors, latencies = embedder.embed((image_path,))

    assert vectors[image_path].ndim == 1
    assert np.linalg.norm(vectors[image_path]) == pytest.approx(1.0, abs=1e-5)
    assert len(latencies) == 1
    assert latencies[0] > 0.0
