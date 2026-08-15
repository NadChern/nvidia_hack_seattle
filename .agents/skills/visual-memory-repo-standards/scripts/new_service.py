#!/usr/bin/env python3
"""Create a repository-standard Python service from the bundled template."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
OWNER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@/-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Lowercase service name using hyphens")
    parser.add_argument(
        "--kind",
        choices=("application", "worker", "inference"),
        default="application",
    )
    parser.add_argument("--owner", required=True, help="Owning person or team")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination parent; defaults to <repository>/services",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip uv lock generation; intended only for testing the generator",
    )
    return parser.parse_args()


def render_template(
    template_root: Path, target: Path, replacements: dict[str, str]
) -> None:
    for source in sorted(template_root.rglob("*")):
        relative = Path(
            *[
                replacements.get(part, part)
                for part in source.relative_to(template_root).parts
            ]
        )
        destination = target / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            content = content.replace(old, new)
        destination.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    if not NAME_PATTERN.fullmatch(args.name):
        raise SystemExit("Service name must match ^[a-z][a-z0-9-]*$.")
    if not OWNER_PATTERN.fullmatch(args.owner):
        raise SystemExit("Owner contains unsupported characters.")

    script_path = Path(__file__).resolve()
    skill_root = script_path.parents[1]
    repository_root = script_path.parents[4]
    output_root = (args.output_root or repository_root / "services").resolve()
    target = output_root / args.name

    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing path: {target}")

    package_name = args.name.replace("-", "_")
    replacements = {
        "__PACKAGE_NAME__": package_name,
        "__SERVICE_NAME__": args.name,
        "__SERVICE_KIND__": args.kind,
        "__SERVICE_OWNER__": args.owner,
    }

    template_root = skill_root / "assets" / "python-service"
    target.mkdir(parents=True)
    render_template(template_root, target, replacements)

    if args.kind == "inference":
        (target / "model-manifest.toml").write_text(
            (
                'logical_model = "replace-me"\n'
                'source = "replace-me"\n'
                'revision = "replace-me"\n'
                'runtime = "replace-me"\n'
                'precision = "replace-me"\n'
                'license = "review-required"\n'
            ),
            encoding="utf-8",
            newline="\n",
        )

    if not args.no_lock:
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit(
                f"Created {target}, but uv is unavailable. Install uv and run `uv lock` there."
            )
        subprocess.run([uv, "lock"], cwd=target, check=True)

    print(f"Created {args.kind} service: {target}")
    print("Next: implement the service contract and run the required checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
