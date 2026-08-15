import os

import pytest
from fastapi import FastAPI

from media_gateway.config import Settings
from media_gateway.main import create_app

#: Reserved for the integration suite, which reads them to find the server it
#: should run against. Everything else must not reach a Settings instance.
TEST_ENV_PREFIX = "VMA_TEST_"


@pytest.fixture(autouse=True)
def isolate_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own configuration out of the tests.

    `Settings` reads `VMA_*` from the environment and from a local `.env`, so a
    developer who has exported `VMA_LIVEKIT_API_SECRET` -- which
    tools/dev-livekit/README.md tells them to do -- silently satisfies the tests
    that assert those credentials are *absent*. The suite then passes or fails
    depending on which shell it was run from, which is worse than either
    outcome on its own.

    Autouse rather than per-test: any test constructing `Settings` is exposed,
    including ones not written yet.
    """
    for name in list(os.environ):
        if name.startswith("VMA_") and not name.startswith(TEST_ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    # A `.env` copied from .env.example is the same hazard by another route.
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    """Settings for a test run: scripted source, so no LiveKit is required."""
    return Settings(
        environment="ci",
        media_source="scripted",
        internal_api_token=None,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)
