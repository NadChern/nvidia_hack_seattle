#!/usr/bin/env python3
"""Measure the Phase-1 gallery read at the 30-object demo scale."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

from visual_memory_memory_contract.protocol import ObjectViewQuality, ObjectViewUpload

from application_memory.config import Settings
from application_memory.store import repository
from application_memory.store.engine import create_all, create_db_engine, create_session_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=int, default=30)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--dim", type=int, default=1152)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--p95-limit-ms", type=float, default=50.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quality = ObjectViewQuality(
        detection_confidence=0.9,
        box_area_fraction=0.3,
        sharpness_score=1.0,
        mask_box_ratio=0.8,
        quality_score=0.9,
    )
    vector = tuple(float(index % 17) / 17.0 for index in range(args.dim))

    with tempfile.TemporaryDirectory(prefix="vma-registry-benchmark-") as directory:
        database = Path(directory) / "registry.db"
        settings = Settings(
            environment="ci",
            database_url=f"sqlite+pysqlite:///{database}",
            registry_max_views_per_object=max(args.views, 2),
            registry_max_embedding_dim=args.dim,
        )
        engine = create_db_engine(settings)
        create_all(engine)
        sessions = create_session_factory(engine)
        with sessions() as db:
            for object_index in range(args.objects):
                enrolled, _ = repository.create_enrolled_object(
                    db,
                    label="keys",
                    idempotency_key=f"benchmark/object/{object_index}",
                )
                for view_index in range(args.views):
                    upload = ObjectViewUpload(
                        view_index=view_index,
                        quality=quality,
                        embedder_id="fixture-1152",
                        pooling="summary+spatial-v1",
                        dim=args.dim,
                        summary=vector,
                        pooled_spatial=vector,
                        crop_sha256=f"{object_index * args.views + view_index:064x}",
                        crop_base64="",
                    )
                    repository.put_object_view(
                        db,
                        object_id=enrolled.object_id,
                        view_id=f"view_{object_index}_{view_index}",
                        upload=upload,
                        crop_relative_path=f"{enrolled.object_id}/{view_index}.bin",
                        max_views=args.views,
                        max_dim=args.dim,
                    )
            db.commit()

            latencies_ms: list[float] = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                gallery = repository.list_gallery(db)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            assert len(gallery.objects) == args.objects
            assert len(gallery.views) == args.objects * args.views
        engine.dispose()

    ordered = sorted(latencies_ms)
    p95 = ordered[max(0, round(0.95 * len(ordered)) - 1)]
    vector_bytes_per_object = args.views * args.dim * 4 * 2
    print(
        f"objects={args.objects} views={args.objects * args.views} dim={args.dim} "
        f"iterations={args.iterations}"
    )
    print(
        f"gallery latency ms p50={statistics.median(latencies_ms):.2f} "
        f"p95={p95:.2f}; vector bytes/object={vector_bytes_per_object}"
    )
    return 0 if p95 < args.p95_limit_ms else 1


if __name__ == "__main__":
    raise SystemExit(main())
