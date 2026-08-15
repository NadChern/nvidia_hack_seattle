"""Typed HTTP and internal boundary models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from visual_memory_memory_contract import AnswerStatus

GuardVerdict = Literal[
    "passed", "vetoed:1", "vetoed:2", "vetoed:3", "vetoed:4", "vetoed:5", "vetoed:6"
]
StatusBackend = Literal["stub", "local", "external"]


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=2_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=200)


class AgentQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reply: str
    answer_status: AnswerStatus | None
    object_id: str | None
    guard: GuardVerdict
    latency_ms: int = Field(ge=0)


class AgentMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: int = 0
    guard_passed: int = 0
    guard_vetoed: dict[str, int] = Field(default_factory=dict)
    hands_free_ignored: int = 0
    hands_free_triggered: int = 0
    hands_free_replies: int = 0
    hands_free_errors: int = 0


class AgentStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: StatusBackend
    model: str
    endpoint_host: str
    metrics: AgentMetricsResponse


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    service: str


__all__ = [
    "AgentQueryRequest",
    "AgentMetricsResponse",
    "AgentQueryResponse",
    "AgentStatusResponse",
    "GuardVerdict",
    "HealthResponse",
    "StatusBackend",
]
