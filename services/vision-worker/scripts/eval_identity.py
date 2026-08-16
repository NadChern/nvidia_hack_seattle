#!/usr/bin/env python3
"""Print the fixed personal-identity evaluation table from labeled predictions."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSON object with a `samples` list",
    )
    parser.add_argument("--status-url", help="optional Vision /v1/status URL")
    parser.add_argument("--token", help="optional bearer token")
    return parser.parse_args()


def rate(numerator: int, denominator: int) -> str:
    value = numerator / denominator if denominator else math.nan
    rendered = f"{value:.3f}" if denominator else "n/a"
    return f"{rendered} ({numerator}/{denominator})"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def evaluate(samples: list[dict[str, Any]]) -> dict[str, str]:
    resolved = [sample for sample in samples if sample.get("resolved_object_id") is not None]
    correct = [
        sample
        for sample in resolved
        if sample.get("resolved_object_id") == sample.get("expected_object_id")
    ]
    registered_present = [
        sample
        for sample in samples
        if sample.get("registered", False) and sample.get("present", True)
    ]
    negatives = [sample for sample in samples if sample.get("expected_object_id") is None]
    false_identity = [
        sample for sample in negatives if sample.get("resolved_object_id") is not None
    ]
    escalated = [sample for sample in samples if sample.get("escalated", False)]
    switches = [sample for sample in samples if sample.get("identity_switch", False)]
    latencies = [float(sample["latency_ms"]) for sample in samples if "latency_ms" in sample]
    track_counts = Counter(str(sample.get("track_id", "")) for sample in samples)
    resolution_count = sum(int(sample.get("resolution_count", 1)) for sample in samples)
    return {
        "identity resolution precision": rate(len(correct), len(resolved)),
        "identity resolution recall": rate(len(correct), len(registered_present)),
        "false-identity rate": rate(len(false_identity), len(negatives)),
        "identity switch rate": rate(len(switches), len(samples)),
        "VLM escalation rate": rate(len(escalated), len(samples)),
        "resolutions per track": (
            f"{resolution_count / len(track_counts):.3f} ({resolution_count}/{len(track_counts)})"
            if track_counts
            else "n/a (0/0)"
        ),
        "identity latency p50/p95 ms": (
            f"{percentile(latencies, 0.50):.1f}/{percentile(latencies, 0.95):.1f} "
            f"(N={len(latencies)})"
            if latencies
            else "n/a (N=0)"
        ),
    }


def fetch_status(url: str, token: str | None) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10.0) as response:
        parsed = json.load(response)
    if not isinstance(parsed, dict):
        raise ValueError("status endpoint did not return an object")
    return cast("dict[str, Any]", parsed)


def main() -> int:
    args = parse_args()
    payload = json.loads(args.predictions.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError("predictions must be an object containing a samples list")
    samples = cast("list[dict[str, Any]]", payload["samples"])
    results = evaluate(samples)
    print("metric | value")
    print("--- | ---")
    for name, value in results.items():
        print(f"{name} | {value}")

    if args.status_url:
        identity = fetch_status(args.status_url, args.token).get("identity", {})
        print("\nruntime identity counters")
        print(json.dumps(identity, indent=2, sort_keys=True))
    print("\nNOTE: rates include numerator/denominator; state when N is too small for reliability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
