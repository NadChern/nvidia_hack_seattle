from __future__ import annotations

import sys
from types import ModuleType

import pytest

from speech.config import Settings
from speech.kokoro_cuda_backend import KokoroCudaTextToSpeech
from speech.main import _select_stt_backend, _select_tts_backend
from speech.parakeet_cuda_backend import ParakeetCudaSpeechToText
from speech.tts import StubTextToSpeech


def test_explicit_stub_tts_stops_before_any_model_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "speech.main.get_settings",
        lambda: Settings(tts_backend="stub"),
    )

    backend = _select_tts_backend()

    assert isinstance(backend, StubTextToSpeech)


def test_cuda_tts_is_selected_when_mlx_is_absent_and_cuda_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    monkeypatch.setitem(sys.modules, "kokoro", ModuleType("kokoro"))
    monkeypatch.setattr("speech.main._cuda_is_usable", lambda: True)

    backend = _select_tts_backend()

    assert isinstance(backend, KokoroCudaTextToSpeech)


def test_cuda_stt_is_selected_when_mlx_is_absent_and_cuda_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setattr("speech.main._cuda_is_usable", lambda: True)

    backend = _select_stt_backend()

    assert isinstance(backend, ParakeetCudaSpeechToText)
