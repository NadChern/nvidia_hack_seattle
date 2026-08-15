"""Application assembly.

Which real backend serves `/v1/synthesize` and `/v1/stt/{session_id}` is
decided once here, not per request: `KokoroMlxTextToSpeech`/
`ParakeetMlxSpeechToText` if their respective mlx packages are importable,
`StubTextToSpeech`/`StubSpeechToText` otherwise. That is what lets this
module -- and the whole service -- import and start cleanly with no mlx
installed at all (Linux ARM64 / the GN100 deploy target / CI), and still
gives a teammate on a non-Mac machine working, if fake, endpoints to build
against.
"""

from __future__ import annotations

from fastapi import FastAPI

from speech import __version__
from speech.api import health, status, stt, synthesize
from speech.config import get_settings
from speech.kokoro_backend import KokoroMlxTextToSpeech
from speech.kokoro_cuda_backend import KokoroCudaTextToSpeech
from speech.logging import configure_logging
from speech.parakeet_backend import ParakeetMlxSpeechToText
from speech.parakeet_cuda_backend import ParakeetCudaSpeechToText
from speech.stt import SpeechToText, StubSpeechToText
from speech.tts import StubTextToSpeech, TextToSpeech


def _cuda_is_usable() -> bool:
    """Whether there is a CUDA device this process can actually use.

    Importability is not enough. `torch` installs and imports perfectly well
    on a machine with no GPU, or with a driver too old for the build, and in
    both cases it reports `cuda.is_available()` False rather than raising. So
    the probe asks the question that matters instead of assuming the answer
    from a successful import.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    except ModuleNotFoundError:
        return False
    # `bool(...)` because torch is unstubbed here, so this expression is
    # Unknown to the type checker even though it is plainly a boolean.
    return bool(torch.cuda.is_available())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _select_tts_backend() -> TextToSpeech:
    """Pick a voice: explicit stub, MLX, CUDA, or import-fallback stub.

    MLX first, deliberately. On a Mac the `cuda` group is not installed at all,
    so the order only matters on a machine that somehow has both -- and there
    the MLX path is the one that was measured on its hardware.
    """
    if get_settings().tts_backend == "stub":
        return StubTextToSpeech()

    try:
        # Importability probe only -- never used beyond that, so mlx-audio's
        # missing type stubs are the only thing to suppress here, narrowly.
        # `reportMissingImports` too: `mlx_audio` is an optional, Darwin-only
        # dependency group, genuinely uninstalled on CI's Linux pyright run.
        import mlx_audio  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport, reportMissingImports]
    except ModuleNotFoundError:
        pass
    else:
        return KokoroMlxTextToSpeech()

    if _cuda_is_usable():
        try:
            import kokoro  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport, reportMissingImports]
        except ModuleNotFoundError:
            pass
        else:
            return KokoroCudaTextToSpeech()

    return StubTextToSpeech()


def _select_stt_backend() -> SpeechToText:
    """Pick an ear, on the same rules as `_select_tts_backend`.

    Separate probes rather than one shared "is a real backend available":
    the two capabilities depend on unrelated packages, and a machine with a
    working TTS and no STT should get the one it has rather than neither.
    """
    try:
        # Parakeet's own package, not mlx-audio (Kokoro's).
        import parakeet_mlx  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport, reportMissingImports]
    except ModuleNotFoundError:
        pass
    else:
        return ParakeetMlxSpeechToText()

    if _cuda_is_usable():
        try:
            import transformers  # noqa: F401  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport, reportMissingImports]
        except ModuleNotFoundError:
            pass
        else:
            return ParakeetCudaSpeechToText()

    return StubSpeechToText()


settings = get_settings()
configure_logging(level=settings.log_level, service=settings.service_name, version=__version__)
app = FastAPI(title=settings.service_name)
app.state.settings = settings
app.state.tts_backend = _select_tts_backend()
app.state.stt_backend = _select_stt_backend()

for module in (health, status, synthesize, stt):
    app.include_router(module.router)
