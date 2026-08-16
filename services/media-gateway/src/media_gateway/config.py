"""Service configuration.

All settings come from the environment with a `VMA_` prefix. Settings are
frozen and validated at startup so a misconfiguration fails the process rather
than surfacing as odd behaviour under load.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "ci", "deploy"]
MediaSourceKind = Literal["livekit", "scripted"]
DimensionGuardMode = Literal["strict", "sustained", "first_frame_wins"]
VideoEncoding = Literal["jpeg", "rgba_raw"]
SubscribeVideoQuality = Literal["low", "medium", "high"]

MIN_SECRET_LENGTH = 32

#: Credentials that must never reach a real deployment. The first is the
#: spike's published value (docs/spikes/livekit-media-gateway); the rest are
#: LiveKit's well-known dev defaults.
INSECURE_KEYS = frozenset({"visual-memory-spike", "devkey"})
INSECURE_SECRET_PREFIXES = ("visual-memory-spike-secret", "secret")


def _env_file() -> str | None:
    """Load a local .env outside deploy only.

    A .env baked into a production image is exactly the secret-handling
    anti-pattern docs/07-Privacy-and-Security.md forbids.
    """
    return None if os.getenv("VMA_ENVIRONMENT") == "deploy" else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        env_file=_env_file(),
        extra="ignore",
        frozen=True,
    )

    service_name: str = "media-gateway"
    environment: Environment = "dev"
    log_level: str = "INFO"

    # --- Media source -----------------------------------------------------
    #: `scripted` drives the relay from an in-process script with no LiveKit at
    #: all, which is what makes the whole service testable in CI.
    media_source: MediaSourceKind = "livekit"
    #: How fast the scripted source produces. Tests turn this down to run
    #: quickly, or up to exercise an idle stream.
    scripted_frame_interval_s: float = Field(default=0.1, gt=0)

    # --- LiveKit ----------------------------------------------------------
    #: How *this process* reaches LiveKit. Under Compose that is a service
    #: name on an internal network.
    livekit_url: str = "ws://127.0.0.1:7880"
    #: How a *device* reaches LiveKit, which is not the same thing once the
    #: gateway and the server are separate containers: `ws://livekit:7880`
    #: resolves inside the Compose network and nowhere else, so handing it to
    #: the glasses gives them an address they cannot connect to. Defaults to
    #: `livekit_url`, which is correct whenever both run on one host.
    livekit_public_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: SecretStr | None = None
    livekit_connect_timeout_s: float = Field(default=5.0, gt=0)
    livekit_probe_interval_s: float = Field(default=5.0, gt=0)
    allow_insecure_dev_credentials: bool = False

    # --- Sessions and authorization ---------------------------------------
    room_prefix: str = "vma"
    max_concurrent_sessions: int = Field(default=2, ge=1)
    session_ttl_s: int = Field(default=3600, ge=1)
    #: How long a session nobody ever joined keeps its slot. Short on purpose:
    #: a device that cannot reach LiveKit mints a new session per retry, and at
    #: the default budget of two, two failed joins lock every later attempt out
    #: with `429 capacity_exhausted` until `session_ttl_s` elapses. Must stay
    #: comfortably longer than a normal join takes. See `SessionRegistry.sweep`.
    unclaimed_session_ttl_s: int = Field(default=90, ge=10, le=3600)
    token_ttl_s: int = Field(default=300, ge=1)
    internal_api_token: SecretStr | None = None
    #: Token minting is the most abuse-sensitive surface here.
    sessions_rate_limit_per_minute: int = Field(default=30, ge=1)
    pairing_code_ttl_s: int = Field(default=120, ge=30, le=900)
    #: A day, not a week. The credential is a stateless HMAC, so there is no
    #: revocation short of rotating `internal_api_token`, which unpairs every
    #: device at once. Expiry is therefore the only bound on a lost pair of
    #: glasses, and it should be shorter than the time to notice they are gone.
    #: Re-pairing is one QR scan. See docs/07-Privacy-and-Security.md.
    device_credential_ttl_s: int = Field(default=86_400, ge=300, le=2_592_000)
    pairing_max_pending: int = Field(default=16, ge=1, le=256)
    pairing_claims_rate_limit_per_minute: int = Field(default=20, ge=1)
    device_event_queue_size: int = Field(default=32, ge=1, le=1_024)
    device_event_max_subscribers: int = Field(default=8, ge=1, le=128)
    manual_trigger_ttl_s: int = Field(default=15, ge=5, le=60)
    #: How long a raised "ask for help" request stays listable before it
    #: expires unanswered. Minutes, not seconds, unlike the manual trigger --
    #: this is a person deciding whether to pick up, not a wake word waiting
    #: on the next transcript.
    assist_request_ttl_s: int = Field(default=180, ge=30, le=600)
    #: `NoDecode` is required, not decorative. Without it pydantic-settings
    #: JSON-decodes complex types straight from the environment, before any
    #: validator runs, so the documented `VMA_DEVICE_ID_ALLOWLIST=glasses-01`
    #: fails to parse and only `["glasses-01"]` works. That would make a
    #: security control unusable exactly as documented.
    device_id_allowlist: Annotated[tuple[str, ...], NoDecode] = ()

    # --- Video sampling ---------------------------------------------------
    expected_video_width: int = Field(default=320, ge=1)
    expected_video_height: int = Field(default=180, ge=1)
    #: `strict` rejects anything but the expected size, matching the spike.
    #: `sustained` latches the first size that holds for `SUSTAINED_RUN`
    #: consecutive frames, and re-latches when a different one does -- what a
    #: publisher at an unknown resolution needs, because an encoder ramps up
    #: to its negotiated size over tens of seconds. `first_frame_wins` latches
    #: the very first frame and is kept only for the spike's exact semantics;
    #: against a ramping publisher it latches the bottom rung and rejects
    #: everything thereafter. See `domain/sampling.DimensionGuard`.
    dimension_guard_mode: DimensionGuardMode = "strict"
    #: Which simulcast layer to request from a publisher that sends several.
    #: Without an explicit request LiveKit delivers the lowest layer, so a
    #: 720p camera would silently reach detection at 320x180. Turn this down
    #: if the workstation cannot keep up with full resolution.
    subscribe_video_quality: SubscribeVideoQuality = "high"
    #: The rate sampled frames are relayed to consumers at. Not the rate the
    #: device captures at -- the sampler's job is to feed consumers less than
    #: LiveKit delivers, with a latest-wins slot so a slow consumer never
    #: applies backpressure to ingest.
    #:
    #: Raised from 2 to 8 once a consumer existed that associates frames with
    #: each other rather than treating each in isolation. The Vision Service's
    #: tracker matches detections between *consecutive* frames by bounding-box
    #: overlap, and its motion estimator correlates consecutive frames for a
    #: global translation; at 2fps a carried object moves far enough between
    #: frames that neither has anything to work with, so tracks fragment and
    #: nothing accumulates the dwell a placement needs. 125ms between frames
    #: is enough for both, at a third of the encode cost of 24.
    sample_fps: float = Field(default=8.0, gt=0)
    video_encoding: VideoEncoding = "jpeg"
    jpeg_quality: int = Field(default=92, ge=1, le=100)
    #: 0 is 4:4:4, which keeps chroma edges intact for segmentation.
    jpeg_subsampling: int = Field(default=0, ge=0, le=2)

    # --- Audio ------------------------------------------------------------
    audio_sample_rate: int = Field(default=48_000, ge=8_000)
    audio_channels: int = Field(default=1, ge=1, le=2)
    audio_frame_ms: int = Field(default=20, ge=1)
    audio_chunk_ms: int = Field(default=100, ge=1)
    audio_queue_chunks: int = Field(default=20, ge=1)

    # --- Return audio -----------------------------------------------------
    return_audio_queue_ms: int = Field(default=200, ge=1)
    return_audio_track_name: str = "assistant-tts"

    # --- Relay ------------------------------------------------------------
    ws_keepalive_s: float = Field(default=10.0, gt=0)
    ws_max_subscribers: int = Field(default=8, ge=1)

    # --- Privacy ----------------------------------------------------------
    #: Only 0 is accepted today. The knob exists so the privacy checklist has
    #: something concrete to assert; a real ring buffer belongs with evidence
    #: capture, which this service does not own.
    raw_buffer_seconds: int = 0

    # --- Downstream services (absent until they exist) --------------------
    session_registry_url: str | None = None
    lifecycle_sink_url: str | None = None
    lifecycle_sink_timeout_s: float = Field(default=2.0, gt=0)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("raw_buffer_seconds")
    @classmethod
    def _no_raw_buffer_yet(cls, value: int) -> int:
        if value != 0:
            raise ValueError("raw_buffer_seconds must be 0; this service retains no raw media")
        return value

    @field_validator("device_id_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> object:
        """Accept a comma-separated string so the env var stays readable."""
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                # Silently admitting this would register one device literally
                # named '["glasses-01"]', so the allowlist would reject the
                # real device and nobody would know why.
                raise ValueError(
                    "device_id_allowlist is comma-separated, not JSON: "
                    'use VMA_DEVICE_ID_ALLOWLIST="glasses-01,glasses-02"'
                )
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def _reject_insecure_credentials(self) -> Self:
        if self.allow_insecure_dev_credentials and self.environment == "dev":
            return self

        if self.livekit_api_key is not None and self.livekit_api_key in INSECURE_KEYS:
            raise ValueError(
                f"livekit_api_key {self.livekit_api_key!r} is a well-known development "
                "credential; generate a unique key, or set "
                "VMA_ALLOW_INSECURE_DEV_CREDENTIALS=true in dev"
            )

        if self.livekit_api_secret is not None:
            revealed = self.livekit_api_secret.get_secret_value()
            if revealed.startswith(INSECURE_SECRET_PREFIXES):
                raise ValueError(
                    "livekit_api_secret is a well-known development credential; "
                    "generate a unique secret, or set "
                    "VMA_ALLOW_INSECURE_DEV_CREDENTIALS=true in dev"
                )
        return self

    @model_validator(mode="after")
    def _secret_is_long_enough(self) -> Self:
        # The opt-in exists so a throwaway `livekit-server --dev` loop works,
        # and LiveKit's own dev secret is short. Enforcing length anyway would
        # make the flag unusable for the only case it is for.
        if self.allow_insecure_dev_credentials and self.environment == "dev":
            return self
        secret = self.livekit_api_secret
        if secret is not None and len(secret.get_secret_value()) < MIN_SECRET_LENGTH:
            raise ValueError(f"livekit_api_secret must be at least {MIN_SECRET_LENGTH} characters")
        return self

    @model_validator(mode="after")
    def _deploy_requires_hardening(self) -> Self:
        if self.environment != "deploy":
            return self
        missing: list[str] = []
        if self.internal_api_token is None:
            missing.append("internal_api_token")
        if not self.device_id_allowlist:
            missing.append("device_id_allowlist")
        if missing:
            raise ValueError(f"environment=deploy requires {', '.join(missing)}")
        if self.allow_insecure_dev_credentials:
            raise ValueError("allow_insecure_dev_credentials cannot be set in deploy")
        return self

    def require_livekit_credentials(self) -> tuple[str, str]:
        """Return the LiveKit key and secret, or explain what is missing.

        Checked at startup rather than at import so the service stays
        importable for tests and for the scripted media source.
        """
        if self.livekit_api_key is None or self.livekit_api_secret is None:
            raise ValueError(
                "media_source=livekit requires VMA_LIVEKIT_API_KEY and "
                "VMA_LIVEKIT_API_SECRET; set VMA_MEDIA_SOURCE=scripted to run "
                "without a LiveKit server"
            )
        return self.livekit_api_key, self.livekit_api_secret.get_secret_value()

    @property
    def client_livekit_url(self) -> str:
        """The LiveKit address to hand to a device in a session grant."""
        return self.livekit_public_url or self.livekit_url

    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()
