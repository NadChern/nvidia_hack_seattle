from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import Settings


def test_hands_free_defaults_off_for_console_first_operation() -> None:
    settings = Settings(environment="ci")

    assert not settings.hands_free_enabled
    assert settings.accepted_wake_prefixes == (
        "hey memory",
        "hay memory",
        "he memory",
        "hey memories",
        "hey mammary",
        # Measured on the X3 Pro: Parakeet splits "memory" in two on that mic.
        "hey may me",
        "hey maybe",
        "hey mame",
    )


def test_wake_prefix_is_normalized() -> None:
    settings = Settings(environment="ci", wake_prefix="  Hey   MEMORY ")

    assert settings.wake_prefix == "hey memory"


def test_empty_wake_prefix_is_refused() -> None:
    with pytest.raises(ValidationError, match="wake_prefix"):
        Settings(environment="ci", wake_prefix="   ")


def test_wake_prefix_variants_are_normalized_and_deduplicated() -> None:
    settings = Settings(
        environment="ci",
        wake_prefix="Hey Memory",
        wake_prefix_variants=(" HAY   MEMORY ", "hey memory", "hay memory"),
    )

    assert settings.accepted_wake_prefixes == ("hey memory", "hay memory")


def test_wake_prefix_variants_accept_comma_separated_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VMA_WAKE_PREFIX_VARIANTS", "okay memory, computer")

    settings = Settings(environment="ci")

    assert settings.accepted_wake_prefixes == (
        "hey memory",
        "okay memory",
        "computer",
    )
