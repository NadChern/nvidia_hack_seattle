#!/usr/bin/env python3
"""Probe same-instance identity separation before integrating an embedder.

The dataset is intentionally local and gitignored. Each physical instance owns
``reference`` and one or both of ``query_clean``/``query_realistic`` folders::

    clips/identity-probe/keys_1/reference/*.HEIC
    clips/identity-probe/keys_1/query_clean/*.HEIC
    clips/identity-probe/keys_1/query_realistic/*.HEIC

Nested ``<label>/<instance>/...`` layouts are supported too. For every positive
query, the probe pairs one query from a different physical instance of the same
label. This is the balanced IPLoc-ID accept/reject protocol: an always-accept
classifier scores F1=0.667, so generic class similarity cannot look successful.

Images are square-padded rather than center-cropped. The enrollment pipeline
will eventually supply segmented crops; this probe deliberately preserves the
whole user-collected image and should therefore be treated as a conservative,
dataset-specific risk check rather than a final threshold calibration.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

RADIO_MODEL = "nvidia/C-RADIOv4-H"
RADIO_REVISION = "0057b339059c0b9e1b4ba996f975410ebbfdfcc8"
CLIP_MODEL = "openai/clip-vit-base-patch32"
_IMAGE_SUFFIXES = frozenset({".heic", ".heif", ".jpeg", ".jpg", ".png", ".webp"})
_SPLITS = ("clean", "realistic", "all")

Vector = NDArray[np.float32]
Split = Literal["clean", "realistic", "all"]


@dataclass(frozen=True, slots=True)
class ObjectImages:
    """Reference and query images for one physical object instance."""

    label: str
    instance_id: str
    references: tuple[Path, ...]
    clean_queries: tuple[Path, ...]
    realistic_queries: tuple[Path, ...]

    def queries(self, split: Split) -> tuple[Path, ...]:
        if split == "clean":
            return self.clean_queries
        if split == "realistic":
            return self.realistic_queries
        return self.clean_queries + self.realistic_queries

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return self.references + self.clean_queries + self.realistic_queries


@dataclass(frozen=True, slots=True)
class Trial:
    """One balanced same-instance (positive) or other-instance (negative) score."""

    target_instance: str
    query_instance: str
    query_path: Path
    split: Split
    expected_match: bool
    score: float


@dataclass(frozen=True, slots=True)
class GateMetrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def positive_total(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_total(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 0.0 if denominator == 0 else 2 * self.true_positive / denominator

    @property
    def positive_accuracy(self) -> float:
        return _ratio(self.true_positive, self.positive_total)

    @property
    def negative_accuracy(self) -> float:
        return _ratio(self.true_negative, self.negative_total)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model_name: str
    trials: Mapping[Split, tuple[Trial, ...]]
    positive_margins: tuple[float, ...]
    latencies_ms: tuple[float, ...]


class Embedder(Protocol):
    model_name: str

    def embed(self, paths: Sequence[Path]) -> tuple[Mapping[Path, Vector], tuple[float, ...]]:
        """Return one normalized vector per path and per-image forward latencies."""


class RadioEmbedder:
    """Small probe-only adapter for the pinned transformers-native C-RADIOv4."""

    model_name = RADIO_MODEL

    def __init__(self, *, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel

        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model = AutoModel.from_pretrained(
            RADIO_MODEL,
            revision=RADIO_REVISION,
            trust_remote_code=True,
        ).eval()
        # Keep the input conditioner and weights in their checkpoint dtype.
        # C-RADIO's conditioner intentionally returns float32; converting only
        # the module to fp16 causes a float/half linear mismatch. CUDA autocast
        # below gives fp16 kernels without violating that internal contract.
        self._model.to(device=self._device)
        switch_to_deploy = getattr(self._model, "switch_to_deploy", None)
        if callable(switch_to_deploy):
            switch_to_deploy()

    def embed(self, paths: Sequence[Path]) -> tuple[Mapping[Path, Vector], tuple[float, ...]]:
        vectors: dict[Path, Vector] = {}
        latencies: list[float] = []
        for batch_paths in _batches(paths, self._batch_size):
            images = np.stack([_radio_pixels(path) for path in batch_paths])
            tensor = self._torch.from_numpy(images).to(self._device)
            if self._device.type == "cuda":
                self._torch.cuda.synchronize(self._device)
            started = time.perf_counter()
            autocast = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
                if self._device.type == "cuda"
                else nullcontext()
            )
            with self._torch.inference_mode(), autocast:
                output = self._model(tensor)
            if self._device.type == "cuda":
                self._torch.cuda.synchronize(self._device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            backbone = output["backbone"] if isinstance(output, dict) else output
            summary = backbone.summary.float()
            feature_dim = int(backbone.features.shape[2])
            if summary.shape[1] != feature_dim:
                summary = summary.reshape(summary.shape[0], -1, feature_dim).mean(dim=1)
            batch_vectors = _normalize(summary.cpu().numpy())
            for path, vector in zip(batch_paths, batch_vectors, strict=True):
                vectors[path] = vector.astype(np.float32, copy=False)
                latencies.append(elapsed_ms / len(batch_paths))
        return vectors, tuple(latencies)


class ClipEmbedder:
    """CLIP baseline used for the PeKit-style backbone ablation."""

    model_name = CLIP_MODEL

    def __init__(self, *, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoProcessor, CLIPVisionModelWithProjection

        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._processor = AutoProcessor.from_pretrained(CLIP_MODEL)
        self._model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).eval()
        if self._device.type == "cuda":
            self._model.to(device=self._device, dtype=torch.float16)
        else:
            self._model.to(device=self._device)

    def embed(self, paths: Sequence[Path]) -> tuple[Mapping[Path, Vector], tuple[float, ...]]:
        vectors: dict[Path, Vector] = {}
        latencies: list[float] = []
        for batch_paths in _batches(paths, self._batch_size):
            images = [_load_rgb(path) for path in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            if self._device.type == "cuda":
                pixel_values = pixel_values.to(dtype=self._torch.float16)
                self._torch.cuda.synchronize(self._device)
            started = time.perf_counter()
            with self._torch.inference_mode():
                output = self._model(pixel_values=pixel_values)
            if self._device.type == "cuda":
                self._torch.cuda.synchronize(self._device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            batch_vectors = _normalize(output.image_embeds.float().cpu().numpy())
            for path, vector in zip(batch_paths, batch_vectors, strict=True):
                vectors[path] = vector.astype(np.float32, copy=False)
                latencies.append(elapsed_ms / len(batch_paths))
        return vectors, tuple(latencies)


def discover_dataset(root: Path) -> tuple[ObjectImages, ...]:
    """Discover both flat ``keys_1`` and nested ``keys/keys_1`` layouts."""
    if not root.is_dir():
        raise ValueError(f"dataset does not exist or is not a directory: {root}")

    instances: list[ObjectImages] = []
    for reference_dir in sorted(path for path in root.rglob("reference") if path.is_dir()):
        instance_dir = reference_dir.parent
        label = (
            _label_from_instance(instance_dir.name)
            if instance_dir.parent == root
            else instance_dir.parent.name
        )
        item = ObjectImages(
            label=label,
            instance_id=instance_dir.name,
            references=_images_in(reference_dir),
            clean_queries=_images_in(instance_dir / "query_clean"),
            realistic_queries=_images_in(instance_dir / "query_realistic"),
        )
        if len(item.references) < 2:
            raise ValueError(f"{item.instance_id} needs at least 2 reference images")
        if not item.clean_queries and not item.realistic_queries:
            raise ValueError(f"{item.instance_id} needs at least one query image")
        instances.append(item)

    if not instances:
        raise ValueError(f"no instance/reference directories found below {root}")
    duplicate_ids = _duplicates(item.instance_id for item in instances)
    if duplicate_ids:
        raise ValueError(f"instance directory names must be unique: {sorted(duplicate_ids)}")

    per_label: defaultdict[str, int] = defaultdict(int)
    for item in instances:
        per_label[item.label] += 1
    singleton_labels = sorted(label for label, count in per_label.items() if count < 2)
    if singleton_labels:
        raise ValueError(
            "every label needs at least 2 physical instances for same-class negatives: "
            f"{singleton_labels}"
        )
    return tuple(instances)


def evaluate_embedder(embedder: Embedder, objects: Sequence[ObjectImages]) -> ProbeResult:
    paths = tuple(dict.fromkeys(path for item in objects for path in item.all_paths))
    vectors, latencies = embedder.embed(paths)
    galleries = {
        item.instance_id: np.stack([vectors[path] for path in item.references]) for item in objects
    }

    trials = {
        split: tuple(_balanced_trials(objects, galleries, vectors, split)) for split in _SPLITS
    }
    margins: list[float] = []
    for target in objects:
        others = [item for item in objects if item.label == target.label and item != target]
        for query_path in target.queries("all"):
            own_score = _gallery_score(vectors[query_path], galleries[target.instance_id])
            runner_up = max(
                _gallery_score(vectors[query_path], galleries[item.instance_id]) for item in others
            )
            margins.append(own_score - runner_up)

    return ProbeResult(
        model_name=embedder.model_name,
        trials=trials,
        positive_margins=tuple(margins),
        latencies_ms=latencies,
    )


def gate_metrics(trials: Sequence[Trial], threshold: float) -> GateMetrics:
    tp = fp = tn = fn = 0
    for trial in trials:
        accepted = trial.score >= threshold
        if trial.expected_match and accepted:
            tp += 1
        elif trial.expected_match:
            fn += 1
        elif accepted:
            fp += 1
        else:
            tn += 1
    return GateMetrics(threshold, tp, fp, tn, fn)


def choose_threshold(trials: Sequence[Trial], *, min_negative_accuracy: float = 0.8) -> float:
    """Choose the best F1 threshold while respecting the demo's reject floor."""
    if not trials:
        raise ValueError("cannot choose a threshold without trials")
    scores = sorted({trial.score for trial in trials})
    candidates = [scores[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(scores, scores[1:], strict=False))
    candidates.append(scores[-1] + 1e-6)
    metrics = [gate_metrics(trials, threshold) for threshold in candidates]
    constrained = [
        metric for metric in metrics if metric.negative_accuracy >= min_negative_accuracy
    ]
    pool = constrained or metrics
    best = max(
        pool,
        key=lambda metric: (metric.f1, metric.negative_accuracy, metric.positive_accuracy),
    )
    return best.threshold


def roc_auc(trials: Sequence[Trial]) -> float:
    positives = [trial.score for trial in trials if trial.expected_match]
    negatives = [trial.score for trial in trials if not trial.expected_match]
    if not positives or not negatives:
        return math.nan
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _balanced_trials(
    objects: Sequence[ObjectImages],
    galleries: Mapping[str, NDArray[np.float32]],
    vectors: Mapping[Path, Vector],
    split: Split,
) -> Iterable[Trial]:
    """Yield one positive and one rotating same-label negative per query."""
    for target in objects:
        positive_queries = target.queries(split)
        negative_queries = [
            (other.instance_id, path)
            for other in objects
            if other.label == target.label and other.instance_id != target.instance_id
            for path in other.queries(split)
        ]
        if not positive_queries or not negative_queries:
            continue
        gallery = galleries[target.instance_id]
        for index, positive_path in enumerate(positive_queries):
            yield Trial(
                target_instance=target.instance_id,
                query_instance=target.instance_id,
                query_path=positive_path,
                split=split,
                expected_match=True,
                score=_gallery_score(vectors[positive_path], gallery),
            )
            negative_instance, negative_path = negative_queries[index % len(negative_queries)]
            yield Trial(
                target_instance=target.instance_id,
                query_instance=negative_instance,
                query_path=negative_path,
                split=split,
                expected_match=False,
                score=_gallery_score(vectors[negative_path], gallery),
            )


def _gallery_score(query: Vector, gallery: NDArray[np.float32]) -> float:
    return float(np.max(gallery @ query))


def _normalize(values: NDArray[np.floating]) -> NDArray[np.float32]:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _radio_pixels(path: Path, resolution: int = 512) -> NDArray[np.float32]:
    image = _square_pad(_load_rgb(path)).resize((resolution, resolution), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(pixels, (2, 0, 1))


def _load_rgb(path: Path) -> Image.Image:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:  # pragma: no cover - actionable model-profile failure
            raise RuntimeError("HEIC input requires the models extra with pillow-heif") from exc
        register_heif_opener()
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _square_pad(image: Image.Image) -> Image.Image:
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), color=(127, 127, 127))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def _images_in(path: Path) -> tuple[Path, ...]:
    if not path.is_dir():
        return ()
    return tuple(sorted(item for item in path.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES))


def _label_from_instance(instance_id: str) -> str:
    label = re.sub(r"[_-]?\d+$", "", instance_id).strip("_-")
    return label or instance_id


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _batches(values: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _fmt_rate(value: float, numerator: int, denominator: int) -> str:
    return f"{value:.3f} ({numerator}/{denominator})"


def _print_dataset(objects: Sequence[ObjectImages]) -> None:
    print("Dataset")
    print("instance  label  references  clean  realistic")
    for item in objects:
        print(
            f"{item.instance_id:<9} {item.label:<6} {len(item.references):>10} "
            f"{len(item.clean_queries):>6} {len(item.realistic_queries):>10}"
        )
    print(
        "NOTE: keys-only demo set; thresholds are exploratory and the set is too small "
        "for a reliability claim."
    )


def _print_result(
    result: ProbeResult, default_threshold: float, *, peak_memory_mib: float | None
) -> float:
    all_trials = result.trials["all"]
    chosen = choose_threshold(all_trials)
    print(f"\n{result.model_name}")
    print("split      threshold  F1     positive accuracy  negative accuracy  ROC-AUC")
    for split in _SPLITS:
        trials = result.trials[split]
        if not trials:
            print(f"{split:<10} no balanced trials")
            continue
        threshold = chosen
        metric = gate_metrics(trials, threshold)
        positive_rate = _fmt_rate(
            metric.positive_accuracy, metric.true_positive, metric.positive_total
        )
        negative_rate = _fmt_rate(
            metric.negative_accuracy, metric.true_negative, metric.negative_total
        )
        print(
            f"{split:<10} {threshold:>9.4f}  {metric.f1:.3f}  "
            f"{positive_rate:>18}  {negative_rate:>18}  {roc_auc(trials):.3f}"
        )

    positives = [trial.score for trial in all_trials if trial.expected_match]
    negatives = [trial.score for trial in all_trials if not trial.expected_match]
    metric = gate_metrics(all_trials, chosen)
    margins = result.positive_margins
    default_metric = gate_metrics(all_trials, default_threshold)
    print(
        f"starting min_cosine={default_threshold:.4f}; F1={default_metric.f1:.3f}; "
        f"positive={default_metric.true_positive}/{default_metric.positive_total}; "
        f"negative={default_metric.true_negative}/{default_metric.negative_total}"
    )
    print(
        f"chosen min_cosine={chosen:.4f}; F1={metric.f1:.3f}; "
        f"positive={metric.true_positive}/{metric.positive_total}; "
        f"negative={metric.true_negative}/{metric.negative_total}"
    )
    failures = [
        trial for trial in all_trials if (trial.score >= chosen) is not trial.expected_match
    ]
    for trial in failures:
        expected = "positive" if trial.expected_match else "negative"
        print(
            f"failure expected={expected} target={trial.target_instance} "
            f"query={trial.query_path.name} score={trial.score:.4f}"
        )
    positive_mean = statistics.mean(positives)
    positive_std = statistics.pstdev(positives)
    negative_mean = statistics.mean(negatives)
    negative_std = statistics.pstdev(negatives)
    print(
        f"same cosine mean±std={positive_mean:.4f}±{positive_std:.4f}; "
        f"different={negative_mean:.4f}±{negative_std:.4f}"
    )
    negative_p90 = _percentile(negatives, 90)
    positive_p10 = _percentile(positives, 10)
    escalation_low = min(chosen, positive_p10)
    escalation_high = max(chosen, negative_p90)
    print(
        f"empirical escalation band=[{escalation_low:.4f}, {escalation_high:.4f}] "
        "(positive p10 / negative p90 around the chosen gate)"
    )
    print(
        f"suggested min_margin={max(0.0, _percentile(margins, 10)):.4f}; "
        f"positive margin mean±std={statistics.mean(margins):.4f}±{statistics.pstdev(margins):.4f}"
    )
    if result.latencies_ms:
        print(
            f"embed latency ms/image p50={_percentile(result.latencies_ms, 50):.1f} "
            f"p95={_percentile(result.latencies_ms, 95):.1f} "
            f"(N={len(result.latencies_ms)}, includes first forward)"
        )
    if peak_memory_mib is not None:
        print(f"peak CUDA memory allocated={peak_memory_mib:.1f} MiB")
    return chosen


def _reset_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _release_models() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-cosine", type=float, default=0.75)
    parser.add_argument(
        "--backbone",
        choices=("radio", "clip", "both"),
        default="both",
        help="run C-RADIO, CLIP, or the ablation pair",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    objects = discover_dataset(args.dataset.resolve())
    _print_dataset(objects)

    factories: dict[str, Callable[[], Embedder]] = {
        "radio": lambda: RadioEmbedder(device=args.device, batch_size=args.batch_size),
        "clip": lambda: ClipEmbedder(device=args.device, batch_size=args.batch_size),
    }
    names = ("radio", "clip") if args.backbone == "both" else (args.backbone,)
    chosen: dict[str, float] = {}
    for name in names:
        print(f"\nLoading {name}...")
        _reset_peak_memory()
        embedder = factories[name]()
        result = evaluate_embedder(embedder, objects)
        chosen[name] = _print_result(result, args.min_cosine, peak_memory_mib=_peak_memory_mib())
        del embedder
        _release_models()

    if "radio" in chosen and "clip" in chosen:
        print("\nBackbone ablation")
        print(
            "Compare the F1 rows above at each model's chosen threshold; this tiny set is "
            "reported as evidence, not a benchmark claim."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
