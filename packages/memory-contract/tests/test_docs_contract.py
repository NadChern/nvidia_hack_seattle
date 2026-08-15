"""The examples in docs/06 must be what this package accepts.

`docs/06-Data-Contract.md` claims to be normative and this package claims to be
its executable form. That pairing is only worth asserting if something checks
it: a hand-written JSON block in Markdown drifts silently, and every consumer
builds against the document rather than the code.

`packages/media-contract` runs the equivalent check for the relay. The lifecycle
envelope is deliberately defined in both packages -- coupling them would drag
numpy and websockets into every observation producer -- so both suites validate
against the *same* JSON block, and a divergence fails one of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visual_memory_memory_contract.protocol import (
    LIFECYCLE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    LifecycleEnvelope,
    ObjectState,
    Observation,
    QueryResponse,
)

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


def test_the_observation_example_validates(contract: str) -> None:
    """The envelope every producer copies from.

    Scoped to the "Observation envelope" section rather than indexing the
    document's first JSON block overall -- the "Candidate verification
    boundary" section above it also carries examples now (its own
    docs-contract test lives in packages/vision-contract), and a document-wide
    index would silently start validating the wrong block whenever a section
    above this one gains or loses an example.
    """
    observation = Observation.model_validate(
        json.loads(json_blocks(section(contract, "Observation envelope"))[0])
    )

    assert observation.schema_version == SCHEMA_VERSION
    assert observation.media_epoch_id is not None
    assert observation.session_id.startswith("sess_")
    assert observation.event.action == "placed"


def test_the_lifecycle_envelope_validates(contract: str) -> None:
    """The shape the Memory owner signed off on."""
    blocks = json_blocks(section(contract, "Lifecycle signals"))
    assert blocks, "the Lifecycle signals section has no example envelope"

    envelope = LifecycleEnvelope.model_validate(json.loads(blocks[0]))

    assert envelope.schema_version == LIFECYCLE_SCHEMA_VERSION
    assert envelope.signal.action == "track_lost"
    # The point of the whole amendment: scoped by epoch, carrying no object.
    assert envelope.scope.media_epoch_id is not None
    assert envelope.scope.object_id is None


def test_the_trusted_state_example_validates(contract: str) -> None:
    state = ObjectState.model_validate(
        json.loads(json_blocks(section(contract, "Trusted object state"))[0])
    )

    # The documented example is the invalidated case, which is the one worth
    # getting right: a location was confirmed, then the object moved.
    assert state.current_status == "unknown"
    assert state.current_location is None
    assert state.last_confirmed_placement is not None


def test_the_query_response_example_validates(contract: str) -> None:
    answer = QueryResponse.model_validate(
        json.loads(json_blocks(section(contract, "Query response contract"))[0])
    )

    assert answer.answer_status == "last_confirmed_only"
    assert answer.current_location is None
    # The documented spoken answer states the invalidation rather than stopping
    # at the last known location. That second clause is the product.
    assert "picked up" in answer.spoken_answer


def test_every_documented_answer_status_is_a_valid_value(contract: str) -> None:
    """The doc lists the enum in prose; the models must accept exactly those."""
    listed = set(
        re.findall(
            r"^- `(confirmed|last_confirmed_only|unknown|ambiguous_object)`$", contract, re.M
        )
    )

    assert listed == {"confirmed", "last_confirmed_only", "unknown", "ambiguous_object"}
