"""The register-button poll consumes presses and starts center-anchor registration."""

from __future__ import annotations

import httpx
import pytest

from agent.config import Settings
from agent.events import ConsumedRegister
from agent.register_listener import RegisterTriggerListener, _placeholder_label

pytestmark = pytest.mark.anyio


class FakeEvents:
    def __init__(self, arms: dict[str, ConsumedRegister]) -> None:
        self._arms = arms
        self.consumed: list[str] = []

    async def consume_register_trigger(self, session_id: str) -> ConsumedRegister:
        self.consumed.append(session_id)
        return self._arms.get(session_id, ConsumedRegister(armed=False))


class FakeWorkflow:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, str]] = []

    def start(self, *, label: str, session_id: str, mode: str = "grounded") -> bool:
        self.started.append((label, session_id, mode))
        return True


def _sessions_client(session_ids: list[str]) -> httpx.AsyncClient:
    payload = {"sessions": [{"session_id": s, "publisher_present": True} for s in session_ids]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/sessions"
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://gw")


async def test_labelled_press_starts_center_anchor_with_that_label() -> None:
    events = FakeEvents({"sess_1": ConsumedRegister(armed=True, label="keys")})
    workflow = FakeWorkflow()
    listener = RegisterTriggerListener(Settings(environment="ci"), workflow, events=events)

    async with _sessions_client(["sess_1"]) as client:
        await listener._poll_once(client)

    assert workflow.started == [("keys", "sess_1", "center-anchor")]


async def test_unlabelled_press_gets_a_unique_placeholder() -> None:
    events = FakeEvents({"sess_1": ConsumedRegister(armed=True, label=None)})
    workflow = FakeWorkflow()
    listener = RegisterTriggerListener(Settings(environment="ci"), workflow, events=events)

    async with _sessions_client(["sess_1"]) as client:
        await listener._poll_once(client)

    (label, session_id, mode) = workflow.started[0]
    assert session_id == "sess_1"
    assert mode == "center-anchor"
    assert label.startswith("item ") and label != "item "


async def test_no_press_starts_nothing_but_still_consumes() -> None:
    events = FakeEvents({})  # every session reports not-armed
    workflow = FakeWorkflow()
    listener = RegisterTriggerListener(Settings(environment="ci"), workflow, events=events)

    async with _sessions_client(["sess_1", "sess_2"]) as client:
        await listener._poll_once(client)

    assert events.consumed == ["sess_1", "sess_2"]
    assert workflow.started == []


def test_placeholder_labels_are_unique() -> None:
    assert _placeholder_label() != _placeholder_label()
