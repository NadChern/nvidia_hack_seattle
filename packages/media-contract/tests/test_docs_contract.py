"""The examples in the docs must be the thing the code accepts.

docs/06-Data-Contract.md and docs/12-Media-Relay-Contract.md both say that when
a document and this package disagree, one of them is a bug. That claim is only
worth making if something checks it -- a hand-written JSON example in a Markdown
file drifts silently, and a reviewer signing off on a lifecycle envelope is
signing off on what the document says, not on what the code does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visual_memory_media_contract.protocol import PROTOCOL_VERSION, LifecycleEnvelope

DOCS = Path(__file__).resolve().parents[3] / "docs"


def json_blocks(text: str) -> list[str]:
    return re.findall(r"```json\n(.*?)```", text, re.S)


def section(text: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}(.*?)(?=\n## )", text, re.S)
    assert match, f"docs/06 no longer has a '{heading}' section"
    return match.group(1)


@pytest.fixture
def data_contract() -> Path:
    path = DOCS / "06-Data-Contract.md"
    if not path.exists():
        # The package is usable outside the monorepo; there is nothing to check.
        pytest.skip("docs/06-Data-Contract.md is not present")
    return path


def test_the_proposed_lifecycle_envelope_validates(data_contract: Path) -> None:
    """The envelope the Memory owner is reviewing has to be a real one."""
    section = re.search(r"## Lifecycle signals(.*?)(?=\n## )", data_contract.read_text(), re.S)
    assert section, "docs/06 no longer has a Lifecycle signals section"
    blocks = re.findall(r"```json\n(.*?)```", section.group(1), re.S)
    assert blocks, "the Lifecycle signals section has no example envelope"

    envelope = LifecycleEnvelope.model_validate(json.loads(blocks[0]))

    assert envelope.signal.action == "track_lost"
    # The point of the whole amendment: scoped by epoch, with no object.
    assert envelope.scope.media_epoch_id is not None
    assert envelope.scope.object_id is None
    assert envelope.provenance.protocol_version == PROTOCOL_VERSION


def test_the_observation_envelope_carries_a_media_epoch(data_contract: Path) -> None:
    """The `media_epoch_id` field, present since schema 1.1.

    This package does not model observations -- they are the Memory Service's --
    so this asserts the field is present and documented rather than validating
    a model. Scoped to the "Observation envelope" section rather than
    indexing the document's first JSON block overall -- the "Candidate
    verification boundary" section above it also carries examples now (its
    own docs-contract test lives in packages/vision-contract), and a
    document-wide index would silently start validating the wrong block
    whenever a section above this one gains or loses an example.
    """
    blocks = json_blocks(section(data_contract.read_text(), "Observation envelope"))
    observation = json.loads(blocks[0])

    assert observation["schema_version"] == "1.2"
    assert "media_epoch_id" in observation
    assert observation["session_id"].startswith("sess_")


def test_the_relay_contract_agrees_on_the_protocol_version() -> None:
    path = DOCS / "12-Media-Relay-Contract.md"
    if not path.exists():
        pytest.skip("docs/12-Media-Relay-Contract.md is not present")

    assert PROTOCOL_VERSION in path.read_text()
