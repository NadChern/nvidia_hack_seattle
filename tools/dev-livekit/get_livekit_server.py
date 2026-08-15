#!/usr/bin/env python3
"""Fetch a pinned LiveKit server binary, for teammates without Docker.

Path A (`compose.dev.yaml`) is the primary way to run a local server. This is
the fallback: same version, same config file, same environment variables, so
the two are behaviourally interchangeable.

    python3 tools/dev-livekit/get_livekit_server.py
    python3 tools/dev-livekit/get_livekit_server.py --print-run-command

Downloads to a gitignored `.tools/` and verifies the archive against the
checksums committed below before extracting anything. docs/07-Privacy-and-Security.md
forbids trusting code fetched at runtime; a pinned version with a pinned
digest is the difference between a reproducible dev setup and an unverified
binary from the internet.

Stdlib only, on purpose: this runs before anyone has a virtualenv.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

#: The version the S01 spike validated, and the version compose.dev.yaml runs.
#: Changing it here without changing it there puts the two paths out of step.
VERSION = "1.13.4"

RELEASE_URL = "https://github.com/livekit/livekit/releases/download/v{version}/{asset}"

#: SHA-256 of each release archive, from the release's own `checksums.txt`.
#: The windows_amd64 entry matches the one the spike's setup.ps1 verified
#: independently, which is a useful cross-check that this table is right.
#:
#: LiveKit publishes no macOS build for this release -- see PLATFORM_HELP.
CHECKSUMS = {
    "linux_amd64": (
        "livekit_1.13.4_linux_amd64.tar.gz",
        "549bcbe07a92685e45dfd98d8e7cbafd0e1c91d3502fd417079162e1a3f18d17",
    ),
    "linux_arm64": (
        "livekit_1.13.4_linux_arm64.tar.gz",
        "691d34c0d0095a3d5c6dfb9d7e9353a0600a3423d498136037001626d281ad64",
    ),
    "linux_armv7": (
        "livekit_1.13.4_linux_armv7.tar.gz",
        "a81b785b3951780f4f7e3fd62c02cefac827c40a4763513834e5ebff7b9d8a39",
    ),
    "windows_amd64": (
        "livekit_1.13.4_windows_amd64.zip",
        "a326e025de516e93dfb3719bcd28e5a4ac16f21bcf1ef562499403ca98cc65fe",
    ),
    "windows_arm64": (
        "livekit_1.13.4_windows_arm64.zip",
        "fa9e4174915f8635ee98124459b42630b063ef5680ee054a0cc10209bc60df17",
    ),
}

PLATFORM_HELP = """\
LiveKit publishes no macOS binary for v{version}, so this fallback cannot help
on a Mac. Use one of:

  docker compose -f compose.dev.yaml up -d livekit    (preferred; same config)
  brew install livekit                                (unpinned version)

Both read tools/dev-livekit/livekit.dev.yaml and the same LIVEKIT_KEYS.
"""

MACHINES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "armv7l": "armv7",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def platform_key() -> str:
    """Name this machine the way the release assets are named."""
    system = platform.system().lower()
    machine = MACHINES.get(platform.machine().lower())

    if system == "darwin":
        raise SystemExit(PLATFORM_HELP.format(version=VERSION))
    if system not in {"linux", "windows"} or machine is None:
        raise SystemExit(
            f"unsupported platform {platform.system()}/{platform.machine()}; "
            f"run the server with `docker compose -f compose.dev.yaml up -d livekit`"
        )
    return f"{system}_{machine}"


def digest_of(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download(url: str, destination: Path) -> None:
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        destination.unlink(missing_ok=True)
        raise SystemExit(f"could not download the LiveKit server: {exc.reason}") from exc


def extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
        return
    with tarfile.open(archive) as bundle:
        # `filter="data"` refuses absolute paths, parent traversal, links, and
        # device nodes. Without it a tarball can write outside `into`.
        bundle.extractall(into, filter="data")


def binary_in(directory: Path) -> Path:
    for name in ("livekit-server", "livekit-server.exe"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"no livekit-server binary found under {directory}")


def run_command(binary: Path, root: Path) -> str:
    config = root / "tools" / "dev-livekit" / "livekit.dev.yaml"
    if platform.system().lower() == "windows":
        return (
            f'$env:LIVEKIT_KEYS = "$env:VMA_LIVEKIT_API_KEY: $env:VMA_LIVEKIT_API_SECRET"\n'
            f'& "{binary}" --config "{config}"'
        )
    return (
        f'export LIVEKIT_KEYS="$VMA_LIVEKIT_API_KEY: $VMA_LIVEKIT_API_SECRET"\n'
        f"{binary} --config {config}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-run-command",
        action="store_true",
        help="print how to start the server and exit, without downloading",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the archive is present"
    )
    args = parser.parse_args(argv)

    root = repo_root()
    tools = root / ".tools"
    target = tools / f"livekit-{VERSION}"

    if args.print_run_command and target.exists():
        print(run_command(binary_in(target), root))
        return 0

    key = platform_key()
    asset, expected = CHECKSUMS[key]
    archive = tools / asset

    tools.mkdir(parents=True, exist_ok=True)
    if args.force or not archive.exists():
        download(RELEASE_URL.format(version=VERSION, asset=asset), archive)

    actual = digest_of(archive)
    if actual != expected:
        # Delete it: leaving an archive that failed verification invites
        # someone to extract it by hand.
        archive.unlink(missing_ok=True)
        print(
            f"checksum mismatch for {asset}\n  expected {expected}\n  actual   {actual}\n"
            "The archive has been deleted. Re-run to try again.",
            file=sys.stderr,
        )
        return 1
    print(f"verified sha256 {actual[:16]}...")

    if target.exists():
        shutil.rmtree(target)
    extract(archive, target)

    binary = binary_in(target)
    binary.chmod(0o755)
    print(f"\nLiveKit {VERSION} is at {binary}\n")
    print("Start it with:\n")
    print(run_command(binary, root))
    print(
        "\nThen confirm what it is listening on:\n"
        "  python3 tools/dev-livekit/check_listeners.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
