"""Gating and the one-shot exercise fixture.

The suite is opt-in twice over: every test carries `@pytest.mark.livekit`
(deselected by the default `addopts`), and the fixture skips unless
`VMA_TEST_LIVEKIT_URL` names a server. Either alone would be enough; both means
neither a stray `-m livekit` nor a stray env var can hang the standards CI loop
waiting on a server that is not there.
"""

from __future__ import annotations

import asyncio
import os
import secrets

import pytest

from media_gateway.config import Settings

from .roundtrip import Roundtrip, exercise

#: Long enough to satisfy the startup validator's minimum secret length.
INTERNAL_TOKEN = "integration-internal-token-0123456789"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"{name} is not set; see tools/dev-livekit/README.md")
    return value


@pytest.fixture(scope="module")
def livekit_settings() -> Settings:
    """Point the gateway at the server under test.

    The credentials are the ones the server was started with, so they come from
    the environment rather than the repo -- and the startup validator still
    refuses the spike's and LiveKit's well-known dev values.
    """
    url = _required("VMA_TEST_LIVEKIT_URL")
    key = os.getenv("VMA_TEST_LIVEKIT_API_KEY") or _required("VMA_LIVEKIT_API_KEY")
    secret = os.getenv("VMA_TEST_LIVEKIT_API_SECRET") or _required("VMA_LIVEKIT_API_SECRET")

    return Settings(
        environment="ci",
        media_source="livekit",
        livekit_url=url,
        livekit_api_key=key,
        livekit_api_secret=secret,  # type: ignore[arg-type]
        internal_api_token=INTERNAL_TOKEN,  # type: ignore[arg-type]
        # The spike's configuration exactly: 320x180 with a strict guard, so a
        # transition frame of any other size is counted and discarded.
        expected_video_width=320,
        expected_video_height=180,
        dimension_guard_mode="strict",
        sample_fps=2.0,
        # Each run gets its own room namespace, so a leftover room from a
        # crashed run cannot make the next one pass or fail spuriously.
        room_prefix=f"vma-it-{secrets.token_hex(3)}",
    )


@pytest.fixture(scope="module")
def roundtrip(livekit_settings: Settings) -> Roundtrip:
    """Run the whole exercise once for the entire module.

    Synchronous on purpose: `asyncio.run` here avoids having to widen anyio's
    function-scoped `anyio_backend` fixture to module scope just to share one
    expensive result.
    """
    return asyncio.run(exercise(livekit_settings, INTERNAL_TOKEN))
