from __future__ import annotations

import pytest
from conftest import confirmed_answer, stale_answer, unknown_answer
from visual_memory_memory_contract.protocol import QueryResponse

from agent.guard import (
    NO_TOOL_REPLY,
    guard_registration_reply,
    guard_reply,
    registration_message,
)


def test_registration_fixed_vocabulary_passes_through_untouched() -> None:
    prompt = registration_message("prompt", "keys")

    guarded = guard_registration_reply(prompt, step="prompt", label="keys")

    assert guarded.reply.encode() == prompt.encode()
    assert guarded.verdict == "registration:prompt"
    assert guarded.answer_status is None


def test_registration_guard_replaces_non_scripted_text() -> None:
    guarded = guard_registration_reply(
        "I definitely memorized it forever.", step="succeeded", label="keys"
    )

    assert guarded.reply == registration_message("succeeded", "keys")
    assert guarded.verdict == "registration:succeeded"


def test_rule_1_vetoes_a_reply_without_a_tool_call() -> None:
    result = guard_reply("They are in the kitchen.", None)

    assert result.verdict == "vetoed:1"
    assert result.reply == NO_TOOL_REPLY
    assert result.answer_status is None


def test_rule_1_does_not_apply_when_memory_was_called() -> None:
    source = unknown_answer()

    result = guard_reply(source.spoken_answer, source)

    assert result.verdict == "passed"


def test_rule_2_vetoes_a_place_in_an_unknown_answer() -> None:
    source = unknown_answer()

    result = guard_reply("The keys may be in the kitchen.", source)

    assert result.verdict == "vetoed:2"


def test_rule_2_accepts_unknown_wording_without_a_place() -> None:
    source = unknown_answer()

    result = guard_reply("I do not know where the keys are.", source)

    assert result.verdict == "passed"


def test_rule_3_vetoes_last_confirmed_wording_that_drops_uncertainty() -> None:
    source = stale_answer()

    result = guard_reply("The keys are on the living room coffee table.", source)

    assert result.verdict == "vetoed:3"


@pytest.mark.parametrize(
    "draft",
    [
        "I last saw your keys on the living room coffee table.",
        "Your keys were last on the living room coffee table.",
        "The keys are on the living room coffee table, only.",
        (
            "I last confirmed the keys on the living room coffee table, "
            "but I have not confirmed a new location."
        ),
    ],
)
def test_rule_3_vetoes_wording_that_drops_current_uncertainty_or_invalidation(
    draft: str,
) -> None:
    source = stale_answer()

    result = guard_reply(draft, source)

    assert result.verdict == "vetoed:3"
    assert result.reply == source.spoken_answer


def test_rule_3_accepts_history_uncertainty_and_invalidation_together() -> None:
    source = stale_answer()

    result = guard_reply(source.spoken_answer, source)

    assert result.verdict == "passed"


def test_rule_3_requires_missing_evidence_reason_when_memory_downgraded_for_it() -> None:
    source = stale_answer().model_copy(
        update={
            "spoken_answer": (
                "I last recorded the keys on the living room coffee table, "
                "but I no longer have the picture that showed it, so I cannot confirm that."
            )
        }
    )

    unsafe = guard_reply(
        "I last recorded the keys on the living room coffee table, but I cannot confirm that.",
        source,
    )
    canonical = guard_reply(source.spoken_answer, source)

    assert unsafe.verdict == "vetoed:3"
    assert canonical.verdict == "passed"


def test_rule_4_vetoes_an_ambiguous_answer_that_picks_one_candidate() -> None:
    source = QueryResponse(
        answer_status="ambiguous_object",
        spoken_answer="I know about more than one thing called keys.",
        candidates=("object-keys-01", "object-keys-02"),
    )

    result = guard_reply("It is object-keys-01.", source)

    assert result.verdict == "vetoed:4"


def test_rule_4_accepts_all_candidates_without_selecting_one() -> None:
    source = QueryResponse(
        answer_status="ambiguous_object",
        spoken_answer="I know about more than one thing called keys.",
        candidates=("object-keys-01", "object-keys-02"),
    )
    draft = "I cannot tell which one: object-keys-01 or object-keys-02."

    result = guard_reply(draft, source)

    assert result.verdict == "passed"


def test_rule_5_vetoes_a_location_memory_did_not_return() -> None:
    source = confirmed_answer()

    result = guard_reply("The keys are in the kitchen.", source)

    assert result.verdict == "vetoed:5"


def test_rule_5_accepts_only_the_returned_location() -> None:
    source = confirmed_answer()

    result = guard_reply("The keys are on your living room coffee table.", source)

    assert result.verdict == "passed"


def test_rule_6_vetoes_an_empty_reply() -> None:
    source = confirmed_answer()

    result = guard_reply("", source)

    assert result.verdict == "vetoed:6"


def test_rule_6_accepts_a_bounded_reply() -> None:
    source = confirmed_answer()

    result = guard_reply(source.spoken_answer, source)

    assert result.verdict == "passed"


def test_rule_6_vetoes_an_overlong_reply() -> None:
    source = confirmed_answer()

    result = guard_reply("the " * 101, source, max_reply_chars=100)

    assert result.verdict == "vetoed:6"


def test_veto_falls_back_to_spoken_answer_verbatim() -> None:
    source = stale_answer()

    result = guard_reply("The keys are in the kitchen.", source)

    assert result.reply.encode() == source.spoken_answer.encode()
