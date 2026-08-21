"""Background registration always speaks a prompt and one terminal outcome."""

from __future__ import annotations

import asyncio

import pytest

from agent.guard import registration_message
from agent.metrics import AgentMetrics
from agent.tools.register import RegistrationOutcome
from agent.workflow import RegistrationWorkflow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeTool:
    def __init__(self, *, succeeded: bool = True, fail: bool = False) -> None:
        self.succeeded = succeeded
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []
        self.release = asyncio.Event()
        self.release.set()

    async def register(
        self, label: str, session_id: str, mode: str = "grounded"
    ) -> RegistrationOutcome:
        self.calls.append((label, session_id, mode))
        await self.release.wait()
        if self.fail:
            raise RuntimeError("vision unavailable")
        return RegistrationOutcome(
            "object_keys",
            label,
            self.succeeded,
            "enrollment_complete" if self.succeeded else "too_few_quality_frames",
            3 if self.succeeded else 0,
        )


class FakeReply:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def send(self, session_id: str, text: str) -> None:
        self.calls.append((session_id, text))


async def test_successful_workflow_speaks_fixed_prompt_then_confirmation() -> None:
    tool = FakeTool()
    reply = FakeReply()
    metrics = AgentMetrics()
    workflow = RegistrationWorkflow(tool, reply, metrics=metrics)  # type: ignore[arg-type]

    assert workflow.start(label="keys", session_id="sess_1") is True
    await workflow.drain()

    assert tool.calls == [("keys", "sess_1", "grounded")]
    assert reply.calls == [
        ("sess_1", registration_message("prompt", "keys")),
        ("sess_1", registration_message("succeeded", "keys")),
    ]
    assert metrics.registrations_started == 1
    assert metrics.registrations_succeeded == 1
    assert metrics.registrations_failed == 0


async def test_center_anchor_registers_without_speaking() -> None:
    tool = FakeTool()
    reply = FakeReply()
    metrics = AgentMetrics()
    workflow = RegistrationWorkflow(tool, reply, metrics=metrics)  # type: ignore[arg-type]

    assert workflow.start(label="keys", session_id="sess_1", mode="center-anchor") is True
    await workflow.drain()

    # The button path threads center-anchor to the tool but speaks nothing:
    # no STT to prompt, and TTS may be evicted for the capture.
    assert tool.calls == [("keys", "sess_1", "center-anchor")]
    assert reply.calls == []
    assert metrics.registrations_started == 1
    assert metrics.registrations_succeeded == 1


@pytest.mark.parametrize("tool", [FakeTool(succeeded=False), FakeTool(fail=True)])
async def test_failure_always_reaches_the_scripted_terminal_line(tool: FakeTool) -> None:
    reply = FakeReply()
    metrics = AgentMetrics()
    workflow = RegistrationWorkflow(tool, reply, metrics=metrics)  # type: ignore[arg-type]

    assert workflow.start(label="keys", session_id="sess_1") is True
    await workflow.drain()

    assert reply.calls[0] == ("sess_1", registration_message("prompt", "keys"))
    assert reply.calls[-1] == ("sess_1", registration_message("failed", "keys"))
    assert len(reply.calls) == 2
    assert metrics.registrations_failed == 1


async def test_second_registration_cannot_overlap_in_the_same_session() -> None:
    tool = FakeTool()
    tool.release.clear()
    workflow = RegistrationWorkflow(tool, FakeReply(), metrics=AgentMetrics())  # type: ignore[arg-type]

    assert workflow.start(label="keys", session_id="sess_1") is True
    await asyncio.sleep(0)
    assert workflow.start(label="wallet", session_id="sess_1") is False
    tool.release.set()
    await workflow.drain()
