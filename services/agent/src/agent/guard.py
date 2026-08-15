"""Deterministic supervision for model-rewritten memory answers.

The model is never the authority on location. This module is deliberately pure:
it receives a draft and the exact Memory query result, and either accepts the
draft or returns ``spoken_answer`` byte-for-byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from visual_memory_memory_contract import AnswerStatus, QueryResponse

from agent.models import GuardVerdict, RegistrationStep

NO_TOOL_REPLY = "I do not have a memory of that."
DEFAULT_MAX_REPLY_CHARS = 400

_REGISTRATION_MESSAGES = {
    "prompt": "I'll remember this {label}. Rotate it slowly so I can take a short clip.",
    "succeeded": "Done — I've scanned your {label} and will keep track of it.",
    "failed": "I couldn't get a clear look. Let's retry.",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_LOCATION_CLAIM = re.compile(
    r"\b(?:on|in|under|beside|behind|inside|near|at|next\s+to|in\s+front\s+of)\b\s+([^,.;!?]+)",
    re.IGNORECASE,
)
_NAMED_LOCATION_CLAIM = re.compile(
    r"\b(?:location|place|room|surface)\s+(?:is|was|would\s+be)\s+([^,.;!?]+)",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(
    r"\b(?:but|however|although|afterward|afterwards|since|and\s+i)\b|\bat\s+\d",
    re.IGNORECASE,
)

_HISTORICAL_MARKERS = (
    "last",
    "previously",
    "formerly",
    "historical",
    "history",
)
_CURRENT_UNCERTAINTY_MARKERS = (
    "not confirmed",
    "cannot confirm",
    "cant confirm",
    "do not know",
    "dont know",
    "no longer know",
    "current location is unknown",
    "currently unknown",
    "location is uncertain",
    "location remains uncertain",
)
_POST_HISTORY_MARKERS = (
    "afterward",
    "afterwards",
    "after that",
    "since then",
    "later",
)
_MISSING_EVIDENCE_MARKERS = (
    "picture",
    "photo",
    "image",
    "evidence",
)
_AMBIGUITY_MARKERS = (
    "cannot tell which",
    "can't tell which",
    "which one",
    "more than one",
    "multiple",
    "ambiguous",
    "cannot say which",
)

# Words a rewriter may introduce without introducing a new concrete noun.
# Concrete place words are intentionally absent. A harmless paraphrase may be
# vetoed; a false location may not be allowed through for fluency's sake.
_SAFE_REWRITE_WORDS = frozenset(
    """
    a about after afterward afterwards again all am an and are as at be been before being
    beside but by called can cannot cant confirm confirmed could currently did do does dont
    find for found from had has have however i idea in inside is it its know last left may
    maybe memory might moved multiple my near next no not of on one only or our perhaps place
    placed previously probably put recall record recorded remember right saw say seen since so
    sorry still surface tell than that the their them there these they theyre this those to
    uncertain under unknown unsure up was we were where which with would you your
    """.split()
)
_LOCATION_GRAMMAR = frozenset(
    "a an at beside behind front in inside my near next of on our right the there "
    "under your".split()
)


@dataclass(frozen=True, slots=True)
class GuardResult:
    reply: str
    answer_status: AnswerStatus | None
    object_id: str | None
    verdict: GuardVerdict


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.casefold().replace("'", "")))


def _normalized_phrase(text: str) -> str:
    return " ".join(_tokens(text))


def _allowed_location_tokens(result: QueryResponse) -> frozenset[str]:
    values: list[str] = []
    for location in (result.current_location, result.last_confirmed_placement):
        if location is None:
            continue
        for field in ("room", "surface"):
            value = getattr(location, field, None)
            if value:
                values.append(str(value))
        description = getattr(location, "description", None)
        if description:
            values.append(str(description))
    return frozenset(token for value in values for token in _tokens(value.replace("_", " ")))


def _claimed_location_tokens(reply: str) -> tuple[str, ...]:
    claimed: list[str] = []
    for pattern in (_LOCATION_CLAIM, _NAMED_LOCATION_CLAIM):
        for match in pattern.finditer(reply):
            phrase = _CLAUSE_END.split(match.group(1), maxsplit=1)[0]
            phrase_tokens = list(_tokens(phrase))
            # ``at 10:42`` is a time, not a place.
            if phrase_tokens and phrase_tokens[0].isdigit():
                continue
            claimed.extend(token for token in phrase_tokens if token not in _LOCATION_GRAMMAR)
    return tuple(claimed)


def _unexpected_tokens(reply: str, result: QueryResponse) -> frozenset[str]:
    trusted_text = " ".join((result.spoken_answer, *result.candidates))
    trusted = frozenset(_tokens(trusted_text)) | _allowed_location_tokens(result)
    return frozenset(_tokens(reply)) - trusted - _SAFE_REWRITE_WORDS


def _contains_phrase(normalized: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {normalized} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _preserves_stale_invalidation(draft: str, result: QueryResponse) -> bool:
    """Require history plus Memory's explicit invalidation/downgrade reason.

    A single word such as ``last`` or ``only`` cannot make a present-tense
    location claim safe. Memory currently emits one of two reasons for a
    historical-only answer: a later move/pickup (which itself invalidates the
    current location), or missing corroborating evidence plus an explicit
    inability to confirm. Unknown future wording fails closed to the canonical
    ``spoken_answer``.
    """

    normalized = _normalized_phrase(draft)
    source = _normalized_phrase(result.spoken_answer)
    if not _contains_phrase(normalized, _HISTORICAL_MARKERS):
        return False

    # A later pickup or move is itself the current-location invalidation, but
    # the temporal marker is required so merely mentioning an event does not
    # blur when it happened relative to the historical placement.
    if " picked up " in f" {source} ":
        return " picked up " in f" {normalized} " and _contains_phrase(
            normalized, _POST_HISTORY_MARKERS
        )
    if " moved " in f" {source} ":
        return " moved " in f" {normalized} " and _contains_phrase(
            normalized, _POST_HISTORY_MARKERS
        )
    if _contains_phrase(source, _MISSING_EVIDENCE_MARKERS):
        return _contains_phrase(normalized, _MISSING_EVIDENCE_MARKERS) and _contains_phrase(
            normalized, _CURRENT_UNCERTAINTY_MARKERS
        )

    # QueryResponse has no structured invalidation-reason field. Until one is
    # added to the canonical contract, unfamiliar Memory wording may not be
    # paraphrased safely.
    return normalized == source


def _veto(result: QueryResponse, rule: int) -> GuardResult:
    return GuardResult(
        reply=result.spoken_answer,
        answer_status=result.answer_status,
        object_id=result.object_id,
        verdict=cast(GuardVerdict, f"vetoed:{rule}"),
    )


def registration_message(step: RegistrationStep, label: str) -> str:
    safe_label = " ".join(label.strip().split()) or "object"
    return _REGISTRATION_MESSAGES[step].format(label=safe_label)


def guard_registration_reply(reply: str, *, step: RegistrationStep, label: str) -> GuardResult:
    """Pass only the scripted registration vocabulary, byte-for-byte."""
    expected = registration_message(step, label)
    return GuardResult(
        reply=reply if reply == expected else expected,
        answer_status=None,
        object_id=None,
        verdict=cast(GuardVerdict, f"registration:{step}"),
    )


def guard_reply(
    draft: str,
    tool_result: QueryResponse | None,
    *,
    max_reply_chars: int = DEFAULT_MAX_REPLY_CHARS,
) -> GuardResult:
    """Apply the six guard rules in their specified order."""
    # Rule 1: without a Memory tool result there is no source of truth.
    if tool_result is None:
        return GuardResult(
            reply=NO_TOOL_REPLY,
            answer_status=None,
            object_id=None,
            verdict="vetoed:1",
        )

    claimed_locations = _claimed_location_tokens(draft)
    unexpected = _unexpected_tokens(draft, tool_result)

    # Rule 2: an unknown answer with no historical placement names no place.
    if (
        tool_result.answer_status == "unknown"
        and tool_result.last_confirmed_placement is None
        and (claimed_locations or unexpected)
    ):
        return _veto(tool_result, 2)

    normalized = draft.casefold()

    # Rule 3: stale/history-only answers must say the location is historical
    # and preserve why Memory invalidated or downgraded the current claim.
    if tool_result.answer_status == "last_confirmed_only" and not _preserves_stale_invalidation(
        draft, tool_result
    ):
        return _veto(tool_result, 3)

    # Rule 4: ambiguity must remain ambiguity and all returned candidates must
    # be represented. Memory currently returns stable object IDs here.
    if tool_result.answer_status == "ambiguous_object":
        normalized_draft = _normalized_phrase(draft)
        names_all = all(
            _normalized_phrase(candidate) in normalized_draft
            for candidate in tool_result.candidates
        )
        preserves_ambiguity = any(marker in normalized for marker in _AMBIGUITY_MARKERS)
        if not names_all or not preserves_ambiguity:
            return _veto(tool_result, 4)

    # Rule 5: every location claim must use only location tokens Memory
    # returned, and no new concrete content word may appear.
    allowed_locations = _allowed_location_tokens(tool_result)
    if any(token not in allowed_locations for token in claimed_locations) or unexpected:
        return _veto(tool_result, 5)

    # Rule 6: bounded, non-empty speech only.
    if not draft.strip() or len(draft) > max_reply_chars:
        return _veto(tool_result, 6)

    return GuardResult(
        reply=draft,
        answer_status=tool_result.answer_status,
        object_id=tool_result.object_id,
        verdict="passed",
    )


__all__ = [
    "DEFAULT_MAX_REPLY_CHARS",
    "GuardResult",
    "NO_TOOL_REPLY",
    "guard_registration_reply",
    "guard_reply",
    "registration_message",
]
