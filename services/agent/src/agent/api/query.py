"""Conversational query endpoint with deterministic truthfulness supervision."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from agent.deps import authorize_request, backend_of, metrics_of
from agent.guard import guard_reply
from agent.models import AgentQueryRequest, AgentQueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.post("/v1/agent/query", response_model=AgentQueryResponse)
async def query(body: AgentQueryRequest, request: Request) -> AgentQueryResponse:
    authorize_request(request)
    started = time.perf_counter()

    draft = await backend_of(request).query(body.text, body.session_id)
    guarded = guard_reply(draft.text, draft.tool_result)
    metrics_of(request).record_guard(guarded.verdict)
    latency_ms = max(0, round((time.perf_counter() - started) * 1000.0))

    # Never log body.text or reply: both are transcripts under docs/07.
    logger.info(
        "agent query completed",
        extra={
            "session_id": body.session_id,
            "answer_status": guarded.answer_status,
            "object_id": guarded.object_id,
            "guard": guarded.verdict,
            "latency_ms": latency_ms,
        },
    )
    return AgentQueryResponse(
        reply=guarded.reply,
        answer_status=guarded.answer_status,  # type: ignore[arg-type]
        object_id=guarded.object_id,
        guard=guarded.verdict,
        latency_ms=latency_ms,
    )
