"""The examples in docs/06's Candidate verification boundary section must be
what this package accepts.

`packages/memory-contract` runs the equivalent check for the observation
envelope. Both packages validate against the same document rather than
importing each other's models -- see `protocol.py`'s module docstring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visual_memory_vision_contract.protocol import SCHEMA_VERSION, CandidateEvent, VerifierResult

DOCS = Path(__file__).resolve().parents[3] / "docs"


@pytest.fixture
def contract() -> str:
    path = DOCS / "06-Data-Contract.md"
    if not path.exists():
        # The package is usable outside the monorepo; nothing to check.
        pytest.skip("docs/06-Data-Contract.md is not present")
    return path.read_text()


def json_blocks(text: str) -> list[str]:
    return re.findall(r"```json\n(.*?)```", text, re.S)


def section(text: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}(.*?)(?=\n## )", text, re.S)
    assert match, f"docs/06 no longer has a '{heading}' section"
    return match.group(1)


def test_the_candidate_event_example_validates(contract: str) -> None:
    blocks = json_blocks(section(contract, "Candidate verification boundary"))
    assert blocks, "the Candidate verification boundary section has no example candidate"

    candidate = CandidateEvent.model_validate(json.loads(blocks[0]))

    assert candidate.schema_version == SCHEMA_VERSION
    assert candidate.media_epoch_id is not None
    assert candidate.session_id.startswith("sess_")
    # The point of the amendment: hands are a nullable slot, not a requirement.
    assert candidate.hand_candidate is None


def test_the_verifier_result_example_validates(contract: str) -> None:
    blocks = json_blocks(section(contract, "Candidate verification boundary"))
    assert len(blocks) >= 2, "expected both a candidate-event and a verifier-result example"

    result = VerifierResult.model_validate(json.loads(blocks[1]))

    assert result.outcome == "confirmed"
    assert result.candidate_id.startswith("cand_")


def test_every_documented_outcome_is_a_valid_value(contract: str) -> None:
    """The doc lists the enum in prose; the model must accept exactly those."""
    section_text = section(contract, "Candidate verification boundary")
    listed = set(re.findall(r"^- `(confirmed|rejected|unverified)`:", section_text, re.M))

    assert listed == {"confirmed", "rejected", "unverified"}
