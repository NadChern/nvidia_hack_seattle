#!/usr/bin/env python3
"""Validate repository Python services against the engineering standard."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

DEFAULT_PYTHON = ">=3.11,<3.12"
REQUIRED_RUNTIME = {"fastapi", "pydantic", "pydantic-settings", "uvicorn"}
REQUIRED_DEV = {"httpx", "pytest", "pyright", "ruff"}
REQUIRED_FILES = {
    ".python-version",
    "Dockerfile",
    "README.md",
    "pyproject.toml",
    "service.toml",
    "uv.lock",
}
REQUIRED_DIRS = {"src", "tests"}
# Skipped at any depth. Vendored files inside a virtualenv or package cache are
# not project dependency files; without this, running `uv sync` before the
# validator reports site-packages content such as livekit/requirements.txt.
IGNORED_DIR_NAMES = {".venv", "venv", "node_modules", "__pycache__", "site-packages"}
VALID_EXCEPTION_RULES = {
    "python-version",
    "fastapi-stack",
    "dev-tools",
    "docker-runtime",
    "health-endpoints",
    "uv-lock",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root; defaults to the root containing this skill",
    )
    return parser.parse_args()


def dependency_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~@\[]", requirement, maxsplit=1)[0].lower()


def load_exceptions(service: Path, errors: list[str]) -> set[str]:
    path = service / "standards-exception.toml"
    if not path.exists():
        return set()

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    rules = set(data.get("rules", []))
    unknown = rules - VALID_EXCEPTION_RULES
    if unknown:
        errors.append(f"{service.name}: unknown exception rules: {sorted(unknown)}")
    for field in ("reason", "owner", "approved_by", "expires"):
        if not str(data.get(field, "")).strip():
            errors.append(f"{service.name}: exception is missing {field}")
    return rules & VALID_EXCEPTION_RULES


def validate_service(service: Path) -> list[str]:
    errors: list[str] = []
    exceptions = load_exceptions(service, errors)

    for filename in sorted(REQUIRED_FILES):
        if filename == "uv.lock" and "uv-lock" in exceptions:
            continue
        if not (service / filename).exists():
            errors.append(f"{service.name}: missing {filename}")
    for dirname in sorted(REQUIRED_DIRS):
        if not (service / dirname).is_dir():
            errors.append(f"{service.name}: missing {dirname}/")
    if (service / "tests").is_dir() and not any((service / "tests").rglob("test_*.py")):
        errors.append(f"{service.name}: tests/ contains no test_*.py files")

    pyproject_path = service / "pyproject.toml"
    if not pyproject_path.exists():
        return errors

    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)

    requires_python = project.get("project", {}).get("requires-python")
    if "python-version" not in exceptions and requires_python != DEFAULT_PYTHON:
        errors.append(
            f"{service.name}: requires-python must be {DEFAULT_PYTHON!r}, "
            f"found {requires_python!r}"
        )

    python_version_path = service / ".python-version"
    if (
        "python-version" not in exceptions
        and python_version_path.exists()
        and python_version_path.read_text(encoding="utf-8").strip() != "3.11"
    ):
        errors.append(f"{service.name}: .python-version must contain 3.11")

    runtime = {
        dependency_name(item)
        for item in project.get("project", {}).get("dependencies", [])
    }
    missing_runtime = REQUIRED_RUNTIME - runtime
    if "fastapi-stack" not in exceptions and missing_runtime:
        errors.append(
            f"{service.name}: missing runtime dependencies: {sorted(missing_runtime)}"
        )

    dev = {
        dependency_name(item)
        for item in project.get("dependency-groups", {}).get("dev", [])
    }
    missing_dev = REQUIRED_DEV - dev
    if "dev-tools" not in exceptions and missing_dev:
        errors.append(
            f"{service.name}: missing dev dependencies: {sorted(missing_dev)}"
        )

    tools = project.get("tool", {})
    if "dev-tools" not in exceptions:
        for tool in ("ruff", "pyright", "pytest"):
            if tool not in tools:
                errors.append(f"{service.name}: missing [tool.{tool}] configuration")

    dockerfile = service / "Dockerfile"
    if "docker-runtime" not in exceptions and dockerfile.exists():
        docker_text = dockerfile.read_text(encoding="utf-8")
        if "USER " not in docker_text:
            errors.append(f"{service.name}: Dockerfile must use a non-root USER")
        if "uv sync --frozen" not in docker_text:
            errors.append(f"{service.name}: Dockerfile must use `uv sync --frozen`")

    if "health-endpoints" not in exceptions:
        source_text = (
            "\n".join(
                path.read_text(encoding="utf-8")
                for path in (service / "src").rglob("*.py")
            )
            if (service / "src").exists()
            else ""
        )
        for endpoint in ("/health/live", "/health/ready"):
            if endpoint not in source_text:
                errors.append(f"{service.name}: missing {endpoint} endpoint")

    service_manifest = service / "service.toml"
    if service_manifest.exists():
        with service_manifest.open("rb") as handle:
            manifest = tomllib.load(handle)
        kind = manifest.get("service", {}).get("kind")
        owner = str(manifest.get("service", {}).get("owner", "")).strip()
        if kind not in {"application", "worker", "inference"}:
            errors.append(f"{service.name}: invalid service.kind {kind!r}")
        if not owner or owner == "replace-me":
            errors.append(f"{service.name}: service.owner must be assigned")
        if kind == "inference" and not (service / "model-manifest.toml").exists():
            errors.append(
                f"{service.name}: inference service needs model-manifest.toml"
            )
        elif kind == "inference":
            with (service / "model-manifest.toml").open("rb") as handle:
                model_manifest = tomllib.load(handle)
            for field in (
                "logical_model",
                "source",
                "revision",
                "runtime",
                "precision",
                "license",
            ):
                value = str(model_manifest.get(field, "")).strip()
                if not value or value == "replace-me":
                    errors.append(
                        f"{service.name}: model-manifest.toml needs a pinned {field}"
                    )

    return errors


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repository_root = (args.repo_root or script_path.parents[4]).resolve()
    errors: list[str] = []

    forbidden_names = {"Pipfile", "poetry.lock"}
    for path in repository_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository_root)
        if relative.parts[0] in {"spikes", ".agents", ".git"} or relative.parts[:2] == (
            "docs",
            "spikes",
        ):
            continue
        if IGNORED_DIR_NAMES.intersection(relative.parts):
            continue
        if path.name in forbidden_names or path.name.startswith("requirements"):
            errors.append(f"forbidden project dependency file: {relative}")

    services_root = repository_root / "services"
    services = (
        sorted(path for path in services_root.iterdir() if path.is_dir())
        if services_root.exists()
        else []
    )
    for service in services:
        errors.extend(validate_service(service))

    if errors:
        print("Repository standards validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository standards validation passed ({len(services)} services).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
