from __future__ import annotations

import pytest
from litellm import get_llm_provider
from pydantic import SecretStr, ValidationError

from agent.config import Settings


def test_external_endpoint_without_the_flag_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="VMA_ALLOW_EXTERNAL_LLM"):
        Settings(llm_base_url="https://203.0.113.10/v1")


def test_external_endpoint_requires_an_api_key_after_explicit_opt_in() -> None:
    with pytest.raises(ValidationError, match="VMA_LLM_API_KEY"):
        Settings(
            llm_base_url="https://203.0.113.10/v1",
            allow_external_llm=True,
        )


def test_external_endpoint_is_allowed_with_explicit_opt_in_and_key() -> None:
    settings = Settings(
        llm_base_url="https://203.0.113.10/v1",
        llm_api_key=SecretStr("test-key"),
        allow_external_llm=True,
    )

    assert settings.endpoint_scope == "external"


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("https://api.modelbest.cn/v1", "openai/MiniCPM-V-4.5-9B"),
        (
            "https://openrouter.ai/api/v1",
            "openrouter/nvidia/nemotron-3.5-lightning:free",
        ),
    ],
)
def test_documented_external_profiles_validate_with_explicit_opt_in(
    base_url: str,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent.config.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("203.0.113.10", 443))],
    )
    settings = Settings(
        llm_base_url=base_url,
        llm_model=model,
        llm_api_key=SecretStr("test-key"),
        allow_external_llm=True,
    )

    assert settings.endpoint_scope == "external"
    assert settings.llm_model == model


def test_openrouter_model_string_resolves_to_the_openrouter_provider() -> None:
    model, provider, *_ = get_llm_provider(
        model="openrouter/nvidia/nemotron-3.5-lightning:free",
        api_base="https://openrouter.ai/api/v1",
    )

    assert model == "nvidia/nemotron-3.5-lightning:free"
    assert provider == "openrouter"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
        "http://localhost:11434/v1",
        "http://0.0.0.0:11434/v1",
    ],
)
def test_loopback_endpoint_is_local(base_url: str) -> None:
    settings = Settings(llm_base_url=base_url)

    assert settings.endpoint_scope == "local"


def test_api_key_is_never_rendered() -> None:
    secret = "sk-this-must-never-be-rendered"
    settings = Settings(llm_api_key=SecretStr(secret))

    assert secret not in repr(settings)
    assert secret not in str(settings.model_dump())


def test_deploy_requires_an_internal_api_token() -> None:
    with pytest.raises(ValidationError, match="internal_api_token"):
        Settings(environment="deploy")


def test_memory_auth_falls_back_to_the_shared_internal_token() -> None:
    shared = SecretStr("shared-internal-token-of-at-least-32-chars")

    settings = Settings(internal_api_token=shared)

    assert settings.resolved_memory_api_token is shared


def test_dedicated_memory_token_takes_precedence() -> None:
    shared = SecretStr("shared-internal-token-of-at-least-32-chars")
    dedicated = SecretStr("dedicated-memory-token-of-at-least-32-chars")

    settings = Settings(internal_api_token=shared, memory_api_token=dedicated)

    assert settings.resolved_memory_api_token is dedicated
