"""Service configuration.

All settings come from the environment with a `VMA_` prefix, frozen and
validated at startup so a misconfiguration fails the process rather than
surfacing as odd behaviour once ingestion is already running.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SpeechBackendKind = Literal["auto", "stub"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "speech"
    #: Load selected real STT/TTS models during application startup instead of
    #: making the first wearer query pay model download/load latency. Off for
    #: ordinary development and CI; enabled on the pre-seeded GN100 profile.
    warm_models_on_startup: bool = False
    #: Passed straight to `logging.py`'s `configure_logging` -- Python's
    #: `logging.setLevel` accepts the upper-case name directly.
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    # --- Media Gateway audio relay -----------------------------------------
    #: How this service reaches the gateway's audio relay
    #: (`docs/12-Media-Relay-Contract.md`). Consumed only through
    #: `MediaClient` in `ingest.py`; never hand-parsed.
    gateway_audio_url: str = "ws://127.0.0.1:8080/v1/stream/audio"
    internal_api_token: SecretStr | None = None

    @property
    def gateway_token(self) -> str | None:
        return (
            self.internal_api_token.get_secret_value()
            if self.internal_api_token is not None
            else None
        )

    #: What this service expects the gateway's `audio_sample_rate` to be
    #: configured as. The gateway's own default is 48 kHz
    #: (`services/media-gateway/src/media_gateway/config.py`), but that value
    #: is itself configurable there, so `ingest.py` compares every chunk's
    #: declared `sample_rate` against this and logs a mismatch rather than
    #: assuming. This is *not* a resampling target -- resampling is a later
    #: stage, not this one.
    expected_audio_sample_rate: int = Field(default=48_000, gt=0)

    # --- STT -----------------------------------------------------------
    #: The sample rate `resample.py` converts ingested audio to before a real
    #: STT model would ever see it. Parakeet checkpoints commonly expect
    #: 16 kHz mono, hence the default, but this is a setting rather than a
    #: hardcoded constant precisely because the exact model checkpoint isn't
    #: chosen yet (`role-prompts/Speech.md`, hard rule 7).
    stt_target_sample_rate: int = Field(default=16_000, gt=0)
    #: How long a pause ends an utterance. Below roughly half a second this
    #: starts cutting people off mid-sentence at natural breaths; well above a
    #: second and the assistant feels slow to notice you finished.
    #:
    #: 700 ms was measured against a close desk microphone. On the glasses it
    #: cut wearers off mid-sentence: a head-worn array plus noise suppression
    #: and AGC attenuates inter-word dips enough that Silero reads them as
    #: silence, and a wearer thinking mid-question pauses longer than someone
    #: reading a script. 1000 ms is the compromise -- still inside the "slow to
    #: notice" bound above.
    #:
    #: This is an end-of-speech window, not an initial-listening timeout. Idle
    #: microphone time before the first detected word never consumes it.
    stt_utterance_silence_ms: int = Field(default=1_000, ge=100)
    #: A ceiling, not a target. Without it, audio the VAD reads as unbroken
    #: speech -- sustained noise near the microphone -- would buffer forever
    #: and never reach the model at all.
    #:
    #: 20 s was not survivable on the 8 GB WSL GPU. Measured on the glasses:
    #: the detector found no silence at all, so all 27 utterances in a session
    #: ran to the ceiling, and Parakeet then tried to allocate **10.82 GiB** to
    #: transcribe one of them, against 8 GiB total already shared with
    #: vision-worker. Two transcripts survived out of 27; the rest were CUDA
    #: OOM. A clipped question is a bad demo, but no transcript at all is not a
    #: demo.
    #:
    #: The ceiling starts at the first detected speech frame; idle microphone
    #: time must never make a wearer's first word end the turn immediately.
    #: 8 s is roughly double a real "where did I leave my keys" and bounds the
    #: allocation to something the card can hold. Lower it further if OOM
    #: persists. The ceiling firing at all is a *symptom*: when the detector
    #: finds silence properly, utterances end at ~4 s on their own and this is
    #: never reached.
    stt_utterance_max_seconds: float = Field(default=8.0, gt=0)
    #: Silero's speech probability above which a frame counts as speech.
    stt_vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: How much audio before *detected* speech survives the leading-silence
    #: trim. The detector reports where it grew confident, not where the sound
    #: started, so this has to cover its attack delay or the trim removes the
    #: first phoneme -- see `ingest.PREROLL_SECONDS`. Raise it if the wake
    #: prefix is arriving clipped; the cost is a little extra audio per
    #: utterance, against an assistant that cannot be triggered at all.
    stt_preroll_ms: int = Field(default=600, ge=0, le=3_000)

    # --- STT model (MLX backend, Mac dev) -----------------------------------
    #: Hugging Face repo id for the pinned STT checkpoint. Matches
    #: `model-manifest.toml`'s `source` field, minus its `hf:` scheme prefix
    #: -- `huggingface_hub.hf_hub_download`'s `repo_id` argument expects the
    #: bare id, not the manifest's scheme-qualified form.
    stt_model_source: str = "mlx-community/parakeet-tdt-0.6b-v3"
    #: Pinned revision -- never "latest". Matches `model-manifest.toml`.
    stt_model_revision: str = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    #: MLX has no separate CPU/GPU device selector the way CUDA does -- it
    #: manages Metal placement itself. The closest real, actually-used
    #: equivalent is compute precision, matching `model-manifest.toml`'s
    #: `precision = "fp32"`.
    stt_model_dtype: Literal["float32", "bfloat16"] = "float32"

    # --- TTS model (MLX backend, Mac dev) -----------------------------------
    #: Hugging Face repo id for the pinned TTS checkpoint. Unlike the STT
    #: loader, `mlx_audio.tts.utils.load` accepts a bare repo id plus a
    #: separate `revision=` kwarg directly -- no need to bypass it the way
    #: `parakeet_backend.py` had to.
    tts_model_source: str = "mlx-community/Kokoro-82M-bf16"
    #: Pinned revision -- never "latest". Matches `model-manifest.toml`.
    tts_model_revision: str = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
    #: Kokoro-82M-bf16's weights are natively bf16 on disk, so "bfloat16" is
    #: a no-op cast that matches storage; "float32" is available as an
    #: explicit upcast if ever needed. (Deliberately a different default
    #: than `stt_model_dtype`'s "float32" -- Parakeet's pinned artifact is
    #: natively fp32, Kokoro's is natively bf16; each default matches its
    #: own model's real storage format rather than sharing one guess.)
    tts_model_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    #: Kokoro's language/G2P pipeline selector. "a" is American English --
    #: the only language this service has any use for today.
    tts_lang_code: str = "a"
    #: Kokoro voice preset. `None` lets the model use its own default
    #: (`af_heart`) rather than this service hardcoding one.
    tts_voice: str | None = None
    #: Kokoro's native output is 24 kHz. Kept as a setting rather than
    #: hardcoded because the eventual return-audio rate back through the
    #: gateway isn't decided yet -- `kokoro_backend.py` only resamples
    #: (via `resample.py`) if this differs from the model's native rate.
    tts_output_sample_rate: int = Field(default=24_000, gt=0)

    # --- CUDA backends (Linux + NVIDIA) ---------------------------------
    #: The same *logical* models the MLX path uses, from their original
    #: publishers rather than the mlx-community conversions. Moving between a
    #: Mac and the GN100 changes the runtime, not what the assistant hears.
    #:
    #: NVIDIA publishes v3 with `library_name: transformers`, so this loads
    #: through `AutoModelForCTC` and needs no NeMo -- see
    #: `parakeet_cuda_backend.py`.
    stt_cuda_model_source: str = "nvidia/parakeet-tdt-0.6b-v3"
    #: Pinned revision -- never "latest". Matches `model-manifest.toml`.
    stt_cuda_model_revision: str = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    stt_cuda_model_dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"

    tts_cuda_model_source: str = "hexgrad/Kokoro-82M"
    #: Pinned revision -- never "latest". Matches `model-manifest.toml`.
    #: `KPipeline` takes no `revision=`, so this is recorded for the manifest
    #: and for provenance rather than passed to the loader; pinning it in the
    #: loader needs a `snapshot_download` ahead of the pipeline, which is a
    #: change worth making once the deploy image pre-caches models anyway.
    tts_cuda_model_revision: str = "f3ff3571791e39611d31c381e3a41a3af07b4987"
    #: Kokoro's own voice ids. Unlike the MLX path, `KPipeline` has no "use the
    #: model's default" sentinel -- it requires a voice -- so this is a real
    #: default rather than `None`.
    tts_cuda_default_voice: str = "af_heart"

    # ``stub`` is an explicit resource-safety profile, not an import fallback.
    # It lets a constrained laptop keep real STT without loading a second
    # Speech model when answer synthesis begins.
    tts_backend: SpeechBackendKind = "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
