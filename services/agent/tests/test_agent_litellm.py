from __future__ import annotations

import os
from typing import Any

import pytest
from conftest import confirmed_answer
from google.adk.models import LlmRequest
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from litellm import ModelResponse

from agent.agent import create_agent
from agent.config import Settings
from agent.runner import APP_NAME, USER_ID, AdkRunnerBackend
from agent.tools.memory import MemoryTool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeRegistration:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start(self, *, label: str, session_id: str) -> bool:
        self.calls.append((label, session_id))
        return True


class RegistrationLiteLlmClient(LiteLLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def acompletion(
        self, model: Any, messages: Any, tools: Any, **kwargs: Any
    ) -> ModelResponse:
        del model, messages, tools, kwargs
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "register-1",
                                    "type": "function",
                                    "function": {
                                        "name": "start_registration",
                                        "arguments": '{"label":"keys"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            )
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Starting now."},
                }
            ]
        )


class NoToolLiteLlmClient(LiteLLMClient):
    """Model response that answers directly without a Memory tool call."""

    async def acompletion(
        self, model: Any, messages: Any, tools: Any, **kwargs: Any
    ) -> ModelResponse:
        del model, messages, tools, kwargs
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Private chain-of-thought must not reach speech.",
                        "content": "Here is a concise general answer.",
                    },
                }
            ]
        )


class FakeLiteLlmClient(LiteLLMClient):
    """Two-step completion: one tool call, then one grounded answer."""

    def __init__(self) -> None:
        self.calls = 0

    async def acompletion(
        self,
        model: Any,
        messages: Any,
        tools: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        del model, messages, tools, kwargs
        self.calls += 1
        if self.calls % 2 == 1:
            return ModelResponse(
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{self.calls}",
                                    "type": "function",
                                    "function": {
                                        "name": "where_is",
                                        "arguments": '{"label":"keys"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            )
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "The keys are on the living room coffee table.",
                    },
                }
            ]
        )


async def test_llm_timeout_is_independent_from_local_dependency_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def capture_completion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "unknown"},
                }
            ]
        )

    monkeypatch.setattr("litellm.acompletion", capture_completion)
    settings = Settings(environment="ci", request_timeout_s=7, llm_timeout_s=120)
    agent = create_agent(
        settings,
        MemoryTool(settings, ask_memory=lambda _label, _session: confirmed_answer()),
    )

    responses = [item async for item in agent.model.generate_content_async(LlmRequest())]

    assert responses
    assert captured["timeout"] == 120
    assert captured["max_tokens"] == settings.llm_max_output_tokens
    assert captured["max_retries"] == 0
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert settings.request_timeout_s == 7


async def test_fake_litellm_calls_where_is_and_preserves_the_whole_result() -> None:
    expected = confirmed_answer()
    calls: list[tuple[str, str | None]] = []

    def ask(label: str, session_id: str | None):  # type: ignore[no-untyped-def]
        calls.append((label, session_id))
        return expected

    settings = Settings(environment="ci", max_turns_kept=2)
    client = FakeLiteLlmClient()
    model = LiteLlm(model="openai/test", llm_client=client)
    memory = MemoryTool(settings, ask_memory=ask)
    backend = AdkRunnerBackend(settings, create_agent(settings, memory, model=model))

    result = await backend.query("Where did I leave my keys?", "sess_01")

    assert result.text == "The keys are on the living room coffee table."
    assert result.tool_result == expected
    assert result.tool_result.last_confirmed_placement is not None
    assert calls == [("keys", "sess_01")]
    assert client.calls == 2


async def test_general_question_uses_model_answer_without_memory() -> None:
    calls: list[tuple[str, str | None]] = []

    def ask(label: str, session_id: str | None):  # type: ignore[no-untyped-def]
        calls.append((label, session_id))
        return confirmed_answer()

    settings = Settings(environment="ci")
    memory = MemoryTool(settings, ask_memory=ask)
    model = LiteLlm(model="openai/test", llm_client=NoToolLiteLlmClient())
    backend = AdkRunnerBackend(settings, create_agent(settings, memory, model=model))

    result = await backend.query("Explain photosynthesis briefly.", "sess_01")

    assert result.text == "Here is a concise general answer."
    assert result.tool_result is None
    assert calls == []


async def test_fake_litellm_routes_remember_intent_to_registration_tool() -> None:
    settings = Settings(environment="ci")
    model = LiteLlm(model="openai/test", llm_client=RegistrationLiteLlmClient())
    memory = MemoryTool(settings, ask_memory=lambda _label, _session: confirmed_answer())
    registration = FakeRegistration()
    backend = AdkRunnerBackend(
        settings,
        create_agent(settings, memory, registration, model=model),  # type: ignore[arg-type]
    )

    result = await backend.query("Remember my keys", "sess_01")

    assert result.registration_started is True
    assert result.tool_result is None
    assert registration.calls == [("keys", "sess_01")]


async def test_session_history_is_bounded_by_turn_not_raw_event_count() -> None:
    settings = Settings(environment="ci", max_turns_kept=2)
    client = FakeLiteLlmClient()
    model = LiteLlm(model="openai/test", llm_client=client)
    memory = MemoryTool(settings, ask_memory=lambda _label, _session: confirmed_answer())
    backend = AdkRunnerBackend(settings, create_agent(settings, memory, model=model))

    for _ in range(4):
        await backend.query("Where are my keys?", "sess_bounded")

    session = await backend.sessions.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="sess_bounded",
    )
    assert session is not None
    invocation_ids = {event.invocation_id for event in session.events if event.invocation_id}
    assert len(invocation_ids) == 2


@pytest.mark.skipif(
    not os.getenv("VMA_TEST_AGENT_LLM_URL"),
    reason="set VMA_TEST_AGENT_LLM_URL to run against a real model",
)
async def test_real_model_answers_a_known_question() -> None:
    url = os.environ["VMA_TEST_AGENT_LLM_URL"]
    allow_external = not ("127.0.0.1" in url or "localhost" in url)
    settings = Settings(
        environment="ci",
        llm_base_url=url,
        llm_model=os.getenv("VMA_TEST_AGENT_LLM_MODEL", "openai/qwen3:4b"),
        llm_api_key=os.getenv("VMA_TEST_AGENT_LLM_API_KEY"),
        allow_external_llm=allow_external,
    )
    memory = MemoryTool(settings, ask_memory=lambda _label, _session: confirmed_answer())
    backend = AdkRunnerBackend(settings, create_agent(settings, memory))

    result = await backend.query("Where did I leave my keys?", "sess_real_model")

    assert result.tool_result is not None
    assert result.tool_result.answer_status == "confirmed"
    assert result.text
