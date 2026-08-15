from __future__ import annotations

import datetime as dt

import pytest
from visual_memory_memory_contract.protocol import AnsweredPlacement, Location, QueryResponse

T0 = dt.datetime(2026, 7, 29, 17, 42, 11, 240000, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
def isolate_internal_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LiteLLM's dotenv loading from authenticating unrelated tests."""
    monkeypatch.delenv("VMA_INTERNAL_API_TOKEN", raising=False)


def confirmed_answer() -> QueryResponse:
    return QueryResponse(
        object_id="object-keys-01",
        answer_status="confirmed",
        current_status="confirmed_at_location",
        current_location=Location(room="living_room", surface="coffee_table", relation="on"),
        last_confirmed_placement=AnsweredPlacement(
            occurred_at=T0,
            room="living_room",
            surface="coffee_table",
            relation="on",
            evidence_id="evidence-1",
        ),
        spoken_answer="The keys are on the living room coffee table.",
    )


def stale_answer() -> QueryResponse:
    return QueryResponse(
        object_id="object-keys-01",
        answer_status="last_confirmed_only",
        current_status="unknown",
        last_confirmed_placement=AnsweredPlacement(
            occurred_at=T0,
            room="living_room",
            surface="coffee_table",
            relation="on",
            evidence_id="evidence-1",
        ),
        spoken_answer=(
            "I last confirmed the keys on the living room coffee table, "
            "but they were picked up afterward."
        ),
    )


def unknown_answer() -> QueryResponse:
    return QueryResponse(
        answer_status="unknown",
        spoken_answer="I have no record of the keys.",
    )
