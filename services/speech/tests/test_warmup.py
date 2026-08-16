"""Startup model warmup keeps first-use latency off the wearer path."""

from __future__ import annotations

import pytest

from speech.main import _warm_selected_backends

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class WarmBackend:
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    async def initialize(self) -> None:
        self._calls.append(self._name)


async def test_enabled_warmup_initializes_stt_then_tts() -> None:
    calls: list[str] = []

    await _warm_selected_backends(
        enabled=True,
        stt_backend=WarmBackend("stt", calls),
        tts_backend=WarmBackend("tts", calls),
    )

    assert calls == ["stt", "tts"]


async def test_disabled_warmup_loads_nothing() -> None:
    calls: list[str] = []

    await _warm_selected_backends(
        enabled=False,
        stt_backend=WarmBackend("stt", calls),
        tts_backend=WarmBackend("tts", calls),
    )

    assert calls == []


async def test_stub_like_backends_without_initialize_are_ignored() -> None:
    await _warm_selected_backends(enabled=True, stt_backend=object(), tts_backend=object())
