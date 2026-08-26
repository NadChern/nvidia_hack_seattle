"""Validate per-sighting pooling on the real keyrings (graduates gate item 1b).

Reproduces the reject-headroom result of Spike 9b against the **actual**
production `score_gallery` -- which now pools queries by median (my pipeline
change) -- on the real `clips/identity-probe` keyrings. A keyring's realistic
query set is treated as one sighting. Genuine sightings (object enrolled) must
accept; intruder sightings (object held out of the gallery, so it is not one of
the enrolled objects) must reject against the enrolled ones. We compare deciding
**per frame** against **pooling the sighting by median**, and report the
separation gap each gives -- the number Spike 9b measured as +0.011 per frame vs
+0.054 per sighting.

    cd services/vision-worker
    uv run python scripts/validate_sighting_pooling.py --split realistic

Data is the local, gitignored `clips/identity-probe`; no gateway or glasses.
The probe embedder emits only the summary channel, so summary_weight=1.0; the
median-pooling concept transfers to the blended channel unchanged.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_identity import (  # noqa: E402
    ObjectImages,
    RadioEmbedder,
    Split,
    discover_dataset,
)

from vision_worker.identity.base import EmbeddingVectors  # noqa: E402
from vision_worker.identity.gallery import GalleryView, score_gallery  # noqa: E402

FLOOR = 0.75  # config.py identity_min_cosine
MARGIN = 0.02  # config.py identity_min_margin


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
class Event:
    name: str
    genuine: bool
    n: int
    sighting_score: float  # median over the sighting's frames (my score_gallery)
    sighting_object: str
    sighting_threshold: float
    frame_scores: list[float]  # each frame decided alone
    frame_correct: list[bool]  # per frame: winner is the true object


def _gallery(views: list[GalleryView], exclude: str | None) -> list[GalleryView]:
    return [v for v in views if v.object_id != exclude]


def _evaluate_event(
    name: str,
    *,
    genuine: bool,
    true_id: str,
    frame_vectors: list[np.ndarray],
    views: list[GalleryView],
) -> Event:
    pooled = score_gallery(
        views,
        [_query(v) for v in frame_vectors],
        label="keys",
        summary_weight=1.0,
        floor=FLOOR,
        confusion_margin=MARGIN,
    )
    assert pooled is not None
    frame_scores: list[float] = []
    frame_correct: list[bool] = []
    for v in frame_vectors:
        one = score_gallery(views, [_query(v)], label="keys", summary_weight=1.0, floor=FLOOR)
        assert one is not None
        frame_scores.append(one.score)
        frame_correct.append(one.object_id == true_id)
    return Event(
        name=name,
        genuine=genuine,
        n=len(frame_vectors),
        sighting_score=pooled.score,
        sighting_object=pooled.object_id,
        sighting_threshold=pooled.threshold,
        frame_scores=frame_scores,
        frame_correct=frame_correct,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../../clips/identity-probe"))
    parser.add_argument("--split", choices=("clean", "realistic", "all"), default="realistic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    split: Split = args.split
    objects: tuple[ObjectImages, ...] = discover_dataset(args.dataset.resolve())
    embedder = RadioEmbedder(device=args.device, batch_size=args.batch_size)
    vectors, _ = embedder.embed([p for obj in objects for p in obj.all_paths])

    all_views: list[GalleryView] = []
    for obj in objects:
        for i, ref in enumerate(obj.references):
            all_views.append(_view(obj.instance_id, i, vectors[ref]))

    events: list[Event] = []
    for obj in objects:
        frames = [vectors[q] for q in obj.queries(split)]
        if not frames:
            continue
        # Genuine: the object is enrolled; the full gallery is present.
        events.append(
            _evaluate_event(
                obj.instance_id,
                genuine=True,
                true_id=obj.instance_id,
                frame_vectors=frames,
                views=all_views,
            )
        )
        # Intruder: hold this object out, so its frames are an unenrolled object
        # presented against the others -- it must be rejected.
        events.append(
            _evaluate_event(
                f"INTRUDER {obj.instance_id}",
                genuine=False,
                true_id=obj.instance_id,
                frame_vectors=frames,
                views=_gallery(all_views, exclude=obj.instance_id),
            )
        )

    print(f"dataset: {len(objects)} keyrings  split={split}  floor={FLOOR} margin={MARGIN}\n")
    print("== per sighting: median over the sighting's frames (production score_gallery) ==")
    print(f"  {'event':>20} {'n':>3} {'median':>7} {'thr':>6} {'attributed':>11}  verdict")
    for e in events:
        accepted = e.sighting_score >= e.sighting_threshold
        ok = (accepted and e.genuine and e.sighting_object.endswith(e.name.split()[-1])) or (
            not accepted and not e.genuine
        )
        verdict = ("accept" if accepted else "reject") + ("  ok" if ok else "  ** WRONG **")
        print(
            f"  {e.name:>20} {e.n:>3} {e.sighting_score:>7.3f} {e.sighting_threshold:>6.3f} "
            f"{e.sighting_object:>11}  {verdict}"
        )

    genuine = [e for e in events if e.genuine]
    intruder = [e for e in events if not e.genuine]

    # Separation: lowest accept-side vs highest reject-side, per frame and per sighting.
    lo_gen_frame = min(min(e.frame_scores) for e in genuine)
    hi_int_frame = max(max(e.frame_scores) for e in intruder)
    lo_gen_sight = min(e.sighting_score for e in genuine)
    hi_int_sight = max(e.sighting_score for e in intruder)

    events_right = sum(
        1 for e in events if ((e.sighting_score >= e.sighting_threshold) == e.genuine)
    )
    print(
        f"\n  events decided correctly (accept genuine / reject intruder): "
        f"{events_right}/{len(events)}"
    )
    print("\n== separation: lowest genuine vs highest intruder ==")
    print(
        f"  per frame    lowest genuine {lo_gen_frame:.3f}  highest intruder {hi_int_frame:.3f}  "
        f"gap {lo_gen_frame - hi_int_frame:+.3f}"
    )
    print(
        f"  per sighting lowest genuine {lo_gen_sight:.3f}  highest intruder {hi_int_sight:.3f}  "
        f"gap {lo_gen_sight - hi_int_sight:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
