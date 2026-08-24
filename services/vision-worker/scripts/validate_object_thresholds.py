"""Validate per-object identity thresholds on the real same-class keyrings.

Spike 4 of the enrollment redesign. The live gate used a single global cosine
(`identity_min_cosine=0.8334`) tuned once for the three-keyring probe. The
identity-probe RESULTS warned it "won't survive a second same-class instance":
a query from keyring A can score above 0.8334 against keyring B's references and
be *misidentified* as B. Per-object thresholds fix that by raising each object's
bar to just above its closest confusable sibling.

This harness embeds the `clips/identity-probe` keyrings with the same
C-RADIOv4 adapter the probe used, wraps the vectors as production `GalleryView`s,
and runs queries through the **actual** `score_gallery` / `object_thresholds`
code (summary_weight=1.0, since the probe embedder emits only the summary
channel -- the concept transfers to the blended channel). It reports, for a
global threshold vs. a per-object sweep, how many queries are correctly
identified, misidentified as the wrong keyring, or rejected.

    cd services/vision-worker
    uv run python scripts/validate_object_thresholds.py --split realistic

Data is the local, gitignored `clips/identity-probe`; no gateway or glasses.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the probe's dataset discovery and C-RADIO adapter verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_identity import (  # noqa: E402
    ObjectImages,
    RadioEmbedder,
    Split,
    discover_dataset,
)

from vision_worker.identity.base import EmbeddingVectors  # noqa: E402
from vision_worker.identity.gallery import (  # noqa: E402
    GalleryView,
    object_thresholds,
    score_gallery,
)

GLOBAL_COSINE = 0.8334  # config.py identity_min_cosine default


def _view(instance_id: str, index: int, vector: np.ndarray) -> GalleryView:
    return GalleryView(
        object_id=instance_id,
        label="keys",
        view_id=f"{instance_id}-ref-{index}",
        embedder_id="c-radio-probe",
        pooling="summary",
        dim=int(vector.shape[0]),
        summary=vector.astype(np.float32, copy=False),
        pooled_spatial=vector.astype(np.float32, copy=False),
        crop_reference=f"probe://{instance_id}/{index}",
    )


def _query(vector: np.ndarray) -> EmbeddingVectors:
    return EmbeddingVectors(
        embedder_id="c-radio-probe",
        pooling="summary",
        summary=vector.astype(np.float32, copy=False),
        pooled_spatial=vector.astype(np.float32, copy=False),
    )


@dataclass
class Outcome:
    correct: int = 0
    misidentified: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.misidentified + self.rejected

    def line(self) -> str:
        t = self.total or 1
        return (
            f"correct {self.correct:>3}/{self.total} ({self.correct / t:.0%})   "
            f"misidentified {self.misidentified:>2} ({self.misidentified / t:.0%})   "
            f"rejected {self.rejected:>2} ({self.rejected / t:.0%})"
        )


def _evaluate(
    views: Sequence[GalleryView],
    queries: Sequence[tuple[str, np.ndarray]],
    *,
    floor: float,
    confusion_margin: float,
    global_threshold: float | None,
) -> Outcome:
    """Run every query through score_gallery; accept on the winner's bar.

    When ``global_threshold`` is set the accept bar is that constant (floor and
    margin are passed as 0 so the per-object threshold stays 0 and does not
    interfere); otherwise the accept bar is the winner's per-object threshold.
    """
    outcome = Outcome()
    for true_id, vector in queries:
        if global_threshold is not None:
            score = score_gallery(views, [_query(vector)], label="keys", summary_weight=1.0)
            bar = global_threshold
        else:
            score = score_gallery(
                views,
                [_query(vector)],
                label="keys",
                summary_weight=1.0,
                floor=floor,
                confusion_margin=confusion_margin,
            )
            bar = score.threshold if score is not None else floor
        if score is None or score.score < bar:
            outcome.rejected += 1
        elif score.object_id == true_id:
            outcome.correct += 1
        else:
            outcome.misidentified += 1
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../../clips/identity-probe"))
    parser.add_argument("--split", choices=("clean", "realistic", "all"), default="realistic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.60, 0.70, 0.75, 0.80])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.02, 0.04, 0.06])
    args = parser.parse_args()

    split: Split = args.split
    objects: tuple[ObjectImages, ...] = discover_dataset(args.dataset.resolve())
    print(f"dataset: {len(objects)} instances  split={split}")

    embedder = RadioEmbedder(device=args.device, batch_size=args.batch_size)
    all_paths = [p for obj in objects for p in obj.all_paths]
    vectors, _ = embedder.embed(all_paths)

    views: list[GalleryView] = []
    queries: list[tuple[str, np.ndarray]] = []
    for obj in objects:
        for i, ref in enumerate(obj.references):
            views.append(_view(obj.instance_id, i, vectors[ref]))
        for q in obj.queries(split):
            queries.append((obj.instance_id, vectors[q]))
    print(
        f"gallery: {len(views)} reference views across "
        f"{len({v.object_id for v in views})} objects; {len(queries)} queries\n"
    )

    baseline = _evaluate(
        views, queries, floor=0.0, confusion_margin=0.0, global_threshold=GLOBAL_COSINE
    )
    print(f"GLOBAL  cosine={GLOBAL_COSINE:.4f}   {baseline.line()}\n")

    print("PER-OBJECT sweep (floor x margin):")
    best: tuple[float, tuple[float, float], Outcome] | None = None
    for floor in args.floors:
        for margin in args.margins:
            thr = object_thresholds(views, summary_weight=1.0, floor=floor, confusion_margin=margin)
            outcome = _evaluate(
                views, queries, floor=floor, confusion_margin=margin, global_threshold=None
            )
            thr_str = "  ".join(f"{oid}={thr[oid]:.3f}" for oid in sorted(thr))
            print(f"  floor={floor:.2f} margin={margin:.2f}   {outcome.line()}   [{thr_str}]")
            # Prefer fewest misidentifications, then most correct.
            key = (-outcome.misidentified, outcome.correct)
            if best is None or key > (-best[2].misidentified, best[2].correct):
                best = (floor, (floor, margin), outcome)

    if best is not None:
        floor, (f, m), outcome = best
        print(
            f"\nbest per-object: floor={f:.2f} margin={m:.2f}   {outcome.line()}"
            f"\nvs global:                          {baseline.line()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
