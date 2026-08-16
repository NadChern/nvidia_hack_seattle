"""Google ADK agent construction and the single grounded tool."""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext

from agent.config import Settings
from agent.tools.memory import MemoryTool
from agent.workflow import RegistrationWorkflow

REQUEST_SESSION_STATE = "request_session_id"

INSTRUCTION = """You are the wearer's concise, helpful personal assistant. Answer the question
they actually asked. For general knowledge, explanation, planning, and ordinary
conversation, answer directly from your own knowledge without calling a tool.
Treat phrases such as "hey memory" as optional ways of addressing you, not as
part of the requested task. Be honest about uncertainty and keep spoken answers
brief unless the user asks for detail.

You also have two personal visual-memory tools. Use them only when the request
needs them:

- When the wearer asks where a personal object is, where they left or placed
  something, or what their visual memory recorded about an object's location,
  call where_is exactly once with only the object label. The service supplies
  session identity; never ask for or invent a session identifier. Never claim a
  remembered personal-object location from your own knowledge.
- For "remember/register/scan/learn/save my X", call start_registration exactly
  once with only X as the label. Registration is a background scripted workflow;
  do not invent progress or confirmation text and do not call where_is for that
  request.

A visual-memory tool result is authoritative. Return a short spoken answer that
preserves its answer_status and all uncertainty or invalidation. If answer_status
is last_confirmed_only, say that the location is historical and preserve why it
is not current. If it is ambiguous_object, name every candidate and do not choose
one. Never add a room, surface, or location absent from the tool result.
"""


def create_agent(
    settings: Settings,
    memory: MemoryTool,
    registration: RegistrationWorkflow | None = None,
    *,
    model: LiteLlm | None = None,
) -> LlmAgent:
    """Build the ADK agent once; request identity arrives through tool state."""

    async def where_is(label: str, tool_context: ToolContext) -> dict[str, Any]:
        """Look up one object in trusted Memory and return its complete answer."""
        raw_session_id = tool_context.state.get(REQUEST_SESSION_STATE, "")
        session_id = str(raw_session_id) if raw_session_id else None
        result = await memory.where_is(label.strip(), session_id)
        return result.model_dump(mode="json")

    async def start_registration(label: str, tool_context: ToolContext) -> dict[str, Any]:
        """Start a narrated personal-object scan for the authenticated session."""
        raw_session_id = tool_context.state.get(REQUEST_SESSION_STATE, "")
        session_id = str(raw_session_id) if raw_session_id else ""
        started = (
            registration.start(label=label.strip(), session_id=session_id)
            if registration is not None
            else False
        )
        return {"started": started, "label": label.strip(), "background": True}

    if model is None:
        api_key = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key is not None
            else "local-no-key"
        )
        model_args: dict[str, Any] = {
            "model": settings.llm_model,
            "api_base": settings.llm_base_url,
            "api_key": api_key,
            "timeout": settings.llm_timeout_s,
            "temperature": 0,
            "max_tokens": settings.llm_max_output_tokens,
            "max_retries": settings.llm_max_retries,
        }
        if settings.endpoint_scope == "local":
            model_args["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": settings.llm_local_enable_thinking,
                }
            }
        model = LiteLlm(**model_args)

    return LlmAgent(
        name="visual_memory_agent",
        description="Grounded find and personal-object registration orchestration",
        model=model,
        instruction=INSTRUCTION,
        tools=(
            [FunctionTool(where_is), FunctionTool(start_registration)]
            if registration is not None
            else [FunctionTool(where_is)]
        ),
    )


__all__ = ["INSTRUCTION", "REQUEST_SESSION_STATE", "create_agent"]
