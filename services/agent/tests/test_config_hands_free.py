from __future__ import annotations

from agent.config import Settings


def test_hands_free_defaults_off_for_console_first_operation() -> None:
    settings = Settings(environment="ci")

    assert not settings.hands_free_enabled


def test_hands_free_transport_defaults_are_bounded() -> None:
    settings = Settings(environment="ci")

    assert settings.session_poll_interval_s > 0
    assert settings.listener_reconnect_s > 0
    assert settings.gateway_event_timeout_s > 0
    assert settings.reply_echo_suppression_s > 0
