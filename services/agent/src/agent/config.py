"""Validated service configuration."""

from __future__ import annotations

import ipaddress
import os
import socket
from functools import lru_cache
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["dev", "ci", "deploy"]
AgentBackendKind = Literal["stub", "llm"]
EndpointScope = Literal["local", "external"]
MIN_SECRET_LENGTH = 32


def _env_file() -> str | None:
    return None if os.getenv("VMA_ENVIRONMENT") == "deploy" else ".env"


def _endpoint_host(url: str) -> str:
    host = urlsplit(url).hostname
    if not host:
        raise ValueError("LLM base URL must include a hostname")
    return host


def _resolves_only_to_local(host: str) -> bool:
    """Return true only when every address is loopback or unspecified-local.

    Requiring every result prevents a hostname with one local and one routable
    address from slipping through the default-local egress gate.
    """
    try:
        address = ipaddress.ip_address(host)
        # Unspecified listener addresses still target this host when used as a
        # client endpoint; treat 0.0.0.0 consistently with the dev launcher.
        return address.is_loopback or address.is_unspecified
    except ValueError:
        pass

    try:
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ValueError(f"LLM endpoint host {host!r} could not be resolved") from exc
    return bool(addresses) and all(
        address.is_loopback or address.is_unspecified for address in addresses
    )


class Settings(BaseSettings):
    """All settings are frozen and use the repository-wide ``VMA_`` prefix."""

    _resolved_endpoint_scope: EndpointScope = PrivateAttr(default="local")

    model_config = SettingsConfigDict(
        env_prefix="VMA_",
        env_file=_env_file(),
        extra="ignore",
        frozen=True,
    )

    service_name: str = "agent"
    environment: Environment = "dev"
    log_level: str = "INFO"

    memory_base_url: str = "http://127.0.0.1:8081"
    memory_api_token: SecretStr | None = None
    internal_api_token: SecretStr | None = None

    # Hands-free transport. Session discovery and HUD events are Gateway
    # control traffic; STT and synthesis stay on Speech's existing API.
    hands_free_enabled: bool = False
    gateway_base_url: str = "http://127.0.0.1:8080"
    speech_base_url: str = "http://127.0.0.1:8085"
    session_poll_interval_s: float = Field(default=2.0, ge=0.25, le=60.0)
    listener_reconnect_s: float = Field(default=1.0, ge=0.1, le=30.0)
    gateway_event_timeout_s: float = Field(default=2.0, gt=0.0, le=10.0)
    gateway_audio_sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    gateway_audio_channels: int = Field(default=1, ge=1, le=2)
    max_synthesis_bytes: int = Field(default=10_000_000, ge=1_024, le=50_000_000)

    # Local ADK/LiteLLM is the default path. ``stub`` remains available for
    # deterministic development and the fully offline endpoint suite.
    agent_backend: AgentBackendKind = "llm"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "openai/qwen3:4b"
    llm_api_key: SecretStr | None = None
    allow_external_llm: bool = False
    llm_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)

    wake_prefix: str = "hey memory"
    # STT commonly renders the same spoken prefix several ways. Keep this list
    # explicit and configurable; matching still requires a bounded where-question
    # after the prefix, which is the false-fire gate.
    wake_prefix_variants: Annotated[tuple[str, ...], NoDecode] = (
        "hay memory",
        "he memory",
        "hey memories",
        "hey mammary",
        # Observed on the glasses, from the HUD transcript of a query that
        # failed to trigger: "Hey may me, where did I leave my ma monitor?".
        # Parakeet splits "memory" into two words on this microphone.
        "hey may me",
        "hey maybe",
        "hey mame",
    )
    #: How long a wake prefix stays live when the utterance carrying it held no
    #: question.
    #:
    #: A wake phrase invites a pause -- people say "Hey memory." and *then* ask.
    #: Any silence longer than `VMA_STT_UTTERANCE_SILENCE_MS` ends the utterance
    #: there, so the prefix and the question arrive as two transcripts and
    #: neither one can fire on its own. Carrying the wake forward briefly is
    #: what makes a natural pause survivable; lengthening the VAD window alone
    #: cannot, because someone will always pause a little longer.
    #:
    #: Both gates still apply, just across two utterances instead of one, and
    #: the carry is single-use. Set to 0 to require prefix and question in the
    #: same breath.
    wake_carry_over_s: float = Field(default=10.0, ge=0.0, le=60.0)
    # Bounds local service HTTP and WebSocket setup. LLM inference has its own
    # timeout because a slow free route must not make Memory or audio look hung.
    request_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    max_turns_kept: int = Field(default=6, ge=1, le=50)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator(
        "memory_base_url",
        "gateway_base_url",
        "speech_base_url",
        "llm_base_url",
    )
    @classmethod
    def _http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an http:// or https:// URL with a hostname")
        return value.rstrip("/")

    @field_validator("wake_prefix")
    @classmethod
    def _wake_prefix_is_not_empty(cls, value: str) -> str:
        normalized = " ".join(value.casefold().split())
        if not normalized:
            raise ValueError("wake_prefix must not be empty")
        return normalized

    @field_validator("wake_prefix_variants", mode="before")
    @classmethod
    def _split_wake_prefix_variants(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @field_validator("wake_prefix_variants")
    @classmethod
    def _normalize_wake_prefix_variants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(" ".join(item.casefold().split()) for item in value)
        if any(not item for item in normalized):
            raise ValueError("wake_prefix_variants must not contain an empty prefix")
        return tuple(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def _external_llm_requires_opt_in(self) -> Self:
        host = _endpoint_host(self.llm_base_url)
        scope: EndpointScope = "local" if _resolves_only_to_local(host) else "external"
        if scope == "external":
            if not self.allow_external_llm:
                raise ValueError(
                    "LLM endpoint resolves outside loopback; set "
                    "VMA_ALLOW_EXTERNAL_LLM=true to permit transcript egress"
                )
            if self.llm_api_key is None or not self.llm_api_key.get_secret_value().strip():
                raise ValueError("external LLM endpoint requires VMA_LLM_API_KEY")
        # DNS is part of startup configuration validation. Cache the resulting
        # trust-boundary label so operational status never turns a temporary
        # resolver outage into a 500 after the service has started.
        self._resolved_endpoint_scope = scope
        return self

    @model_validator(mode="after")
    def _deploy_requires_authentication(self) -> Self:
        if self.environment == "deploy" and self.internal_api_token is None:
            raise ValueError("environment=deploy requires internal_api_token")
        return self

    @model_validator(mode="after")
    def _internal_token_is_long_enough(self) -> Self:
        token = self.internal_api_token
        if token is not None and len(token.get_secret_value()) < MIN_SECRET_LENGTH:
            raise ValueError(f"internal_api_token must be at least {MIN_SECRET_LENGTH} characters")
        return self

    @property
    def resolved_memory_api_token(self) -> SecretStr | None:
        """Use a dedicated Memory token when configured, else the shared token.

        Production may isolate the Memory credential. Native development uses
        one internal service token; silently dropping authentication there
        makes every spoken query fail only after STT has already succeeded.
        """
        return self.memory_api_token or self.internal_api_token

    @property
    def accepted_wake_prefixes(self) -> tuple[str, ...]:
        """Canonical wake prefix followed by unique configured STT variants."""
        return tuple(dict.fromkeys((self.wake_prefix, *self.wake_prefix_variants)))

    @property
    def endpoint_host(self) -> str:
        return _endpoint_host(self.llm_base_url)

    @property
    def endpoint_scope(self) -> EndpointScope:
        return self._resolved_endpoint_scope


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["AgentBackendKind", "EndpointScope", "Environment", "Settings", "get_settings"]
