"""`/v1/status`: which backend is actually running.

The endpoint exists because silence is ambiguous. `StubTextToSpeech` returns a
valid WAV containing a tenth of a second of pure silence, with a 200 -- so a
caller that cannot ask has no way to tell "no model is installed here" from
"synthesis is broken". Both are a button press and nothing audible.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from speech.main import app


def test_status_names_the_backends_in_force() -> None:
    with TestClient(app) as client:
        body = client.get("/v1/status").json()

    assert body["service"] == "speech"
    assert set(body["backends"]) == {"tts", "stt"}
    for capability in ("tts", "stt"):
        assert body["backends"][capability]["name"]
        assert isinstance(body["backends"][capability]["real"], bool)


def test_a_stub_backend_is_reported_as_not_real() -> None:
    """The distinction the console needs in order to say so out loud, rather
    than presenting silence as a working voice."""
    with TestClient(app) as client:
        body = client.get("/v1/status").json()

    tts = body["backends"]["tts"]
    if tts["name"].startswith("Stub"):
        assert tts["real"] is False
    else:
        assert tts["real"] is True


def test_status_reports_what_it_expects_from_the_relay() -> None:
    with TestClient(app) as client:
        body = client.get("/v1/status").json()

    assert body["config"]["gateway_audio_url"].startswith("ws")
    assert body["config"]["expected_audio_sample_rate"] > 0
    assert isinstance(body["config"]["warm_models_on_startup"], bool)
