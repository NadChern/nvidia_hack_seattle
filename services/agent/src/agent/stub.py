"""Deterministic Phase 0 conversational backend.

It recognizes only common ``where`` question shapes, calls the one Memory tool,
and returns Memory's ``spoken_answer`` unchanged. It intentionally does not
provide open chat or guess when the question shape is unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from visual_memory_memory_contract import QueryResponse

from agent.tools.memory import MemoryTool


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    text: str
    tool_result: QueryResponse | None
    registration_started: bool = False


class RegistrationStarter(Protocol):
    def start(self, *, label: str, session_id: str) -> bool: ...


class QueryBackend(Protocol):
    async def query(self, text: str, session_id: str | None) -> DraftAnswer: ...


_REGISTRATION_PATTERN = re.compile(
    r"\b(?:remember|scan|learn)\s+(?:my|our|the|this)?\s*(?P<label>.+?)\s*[?.!]*$",
    re.IGNORECASE,
)

_QUESTION_PATTERNS = (
    re.compile(
        r"\bwhere\s+did\s+(?:i|we)\s+(?:leave|put|place)\s+(?:my|our|the|a|an)?\s*(?P<label>.+?)\s*[?.!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhere\s+(?:is|are|was|were)\s+(?:my|our|the|a|an)?\s*(?P<label>.+?)\s*[?.!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhere\s+(?:have|had)\s+(?:i|we)\s+(?:left|put|placed)\s+(?:my|our|the|a|an)?\s*(?P<label>.+?)\s*[?.!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhere\s+(?:my|our|the)?\s*(?P<label>.+?)\s+(?:is|are)\s*[?.!]*$",
        re.IGNORECASE,
    ),
)


def object_label(text: str) -> str | None:
    """Extract a label only from a supported location-question shape."""
    compact = " ".join(text.split())
    for pattern in _QUESTION_PATTERNS:
        match = pattern.search(compact)
        if match:
            label = match.group("label").strip(" .?!")
            return label or None
    return None


def registration_label(text: str) -> str | None:
    match = _REGISTRATION_PATTERN.search(" ".join(text.split()))
    if match is None:
        return None
    label = match.group("label").strip(" .?!")
    return label or None


class StubLlm:
    """Offline backend used by Phase 0 and the endpoint test suite."""

    def __init__(self, memory: MemoryTool, registration: RegistrationStarter | None = None) -> None:
        self._memory = memory
        self._registration = registration

    async def query(self, text: str, session_id: str | None) -> DraftAnswer:
        register = registration_label(text)
        if register is not None and self._registration is not None and session_id is not None:
            return DraftAnswer(
                text="",
                tool_result=None,
                registration_started=self._registration.start(
                    label=register, session_id=session_id
                ),
            )
        label = object_label(text)
        if label is None:
            return DraftAnswer(text="", tool_result=None)
        result = await self._memory.where_is(label, session_id)
        return DraftAnswer(text=result.spoken_answer, tool_result=result)


__all__ = [
    "DraftAnswer",
    "QueryBackend",
    "RegistrationStarter",
    "StubLlm",
    "object_label",
    "registration_label",
]
