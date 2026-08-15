"""Configuration validation, especially the credential guards."""

import pytest
from pydantic import ValidationError

from media_gateway.config import MIN_SECRET_LENGTH, Settings

GOOD_SECRET = "x" * MIN_SECRET_LENGTH


def test_defaults_are_usable_in_dev() -> None:
    settings = Settings()

    assert settings.environment == "dev"
    assert settings.media_source == "livekit"
    assert settings.sample_fps == 8.0
    assert settings.video_encoding == "jpeg"
    assert settings.dimension_guard_mode == "strict"


def test_spike_credentials_are_rejected() -> None:
    with pytest.raises(ValueError, match="well-known development"):
        Settings(
            livekit_api_key="visual-memory-spike",
            livekit_api_secret="visual-memory-spike-secret-with-at-least-32-characters",
        )


def test_livekit_dev_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="well-known development"):
        Settings(livekit_api_key="devkey", livekit_api_secret=GOOD_SECRET)


def test_insecure_credentials_allowed_only_when_explicitly_opted_in() -> None:
    settings = Settings(
        environment="dev",
        allow_insecure_dev_credentials=True,
        livekit_api_key="devkey",
        livekit_api_secret="secret",
    )

    assert settings.livekit_api_key == "devkey"


def test_opt_in_does_not_apply_outside_dev() -> None:
    with pytest.raises(ValueError, match="well-known development"):
        Settings(
            environment="ci",
            allow_insecure_dev_credentials=True,
            livekit_api_key="devkey",
            livekit_api_secret=GOOD_SECRET,
        )


def test_short_secret_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings(livekit_api_key="unique-key", livekit_api_secret="too-short")


def test_deploy_requires_internal_token_and_allowlist() -> None:
    with pytest.raises(ValueError, match="internal_api_token"):
        Settings(environment="deploy")


def test_deploy_rejects_the_insecure_credential_opt_in() -> None:
    with pytest.raises(ValueError, match="cannot be set in deploy"):
        Settings(
            environment="deploy",
            internal_api_token=GOOD_SECRET,
            device_id_allowlist=("glasses-01",),
            allow_insecure_dev_credentials=True,
        )


def test_deploy_accepts_a_hardened_configuration() -> None:
    settings = Settings(
        environment="deploy",
        internal_api_token=GOOD_SECRET,
        device_id_allowlist=("glasses-01",),
        livekit_api_key="unique-key",
        livekit_api_secret=GOOD_SECRET,
    )

    assert settings.device_id_allowlist == ("glasses-01",)


def test_allowlist_accepts_a_comma_separated_string() -> None:
    settings = Settings(device_id_allowlist="glasses-01, glasses-02 ,")

    assert settings.device_id_allowlist == ("glasses-01", "glasses-02")


def test_raw_buffer_must_stay_zero() -> None:
    with pytest.raises(ValueError, match="retains no raw media"):
        Settings(raw_buffer_seconds=60)


def test_require_livekit_credentials_explains_the_alternative() -> None:
    settings = Settings()

    with pytest.raises(ValueError, match="VMA_MEDIA_SOURCE=scripted"):
        settings.require_livekit_credentials()


def test_require_livekit_credentials_returns_the_pair() -> None:
    settings = Settings(livekit_api_key="unique-key", livekit_api_secret=GOOD_SECRET)

    assert settings.require_livekit_credentials() == ("unique-key", GOOD_SECRET)


def test_secret_is_not_exposed_by_repr() -> None:
    settings = Settings(livekit_api_key="unique-key", livekit_api_secret=GOOD_SECRET)

    assert GOOD_SECRET not in repr(settings)
    assert GOOD_SECRET not in str(settings.livekit_api_secret)


def test_settings_are_frozen() -> None:
    settings = Settings()

    with pytest.raises(ValueError, match="frozen"):
        settings.sample_fps = 5.0  # type: ignore[misc]


def test_the_device_allowlist_reads_a_comma_separated_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented form has to be the form that works.

    pydantic-settings JSON-decodes complex types straight from the environment
    before any validator runs, so without `NoDecode` the documented
    `VMA_DEVICE_ID_ALLOWLIST=glasses-01` raises at startup and only JSON is
    accepted -- making a security control unusable exactly as documented.
    """
    monkeypatch.setenv("VMA_DEVICE_ID_ALLOWLIST", "glasses-01, glasses-02")

    settings = Settings(environment="ci", media_source="scripted")

    assert settings.device_id_allowlist == ("glasses-01", "glasses-02")


def test_a_json_device_allowlist_is_refused_rather_than_misread() -> None:
    """One device literally named '["glasses-01"]' would reject the real one."""
    with pytest.raises(ValidationError, match="comma-separated, not JSON"):
        Settings(
            environment="ci",
            media_source="scripted",
            device_id_allowlist='["glasses-01"]',  # type: ignore[arg-type]
        )


def test_devices_get_the_public_livekit_url_not_the_internal_one() -> None:
    """Under Compose these differ, and handing over the wrong one breaks joins.

    `ws://livekit:7880` resolves on the internal network and nowhere else, so
    a device given that address cannot connect at all.
    """
    settings = Settings(
        environment="ci",
        media_source="scripted",
        livekit_url="ws://livekit:7880",
        livekit_public_url="ws://192.168.1.42:7880",
    )

    assert settings.client_livekit_url == "ws://192.168.1.42:7880"


def test_the_public_livekit_url_falls_back_to_the_internal_one() -> None:
    """Correct whenever the gateway and the server share a host."""
    settings = Settings(
        environment="ci", media_source="scripted", livekit_url="ws://127.0.0.1:7880"
    )

    assert settings.client_livekit_url == "ws://127.0.0.1:7880"
