"""What Cosmos's reply *means*, against scripted model output.

No model runs here. The risk this covers is parsing: Cosmos returns boxes as
native `<ref>/<box>` grounding tokens (0-1000, xyxy) plus a free-form JSON tail
for actions -- a shape validated live in the Phase-0 probe -- and the pipeline's
whole identity crop depends on reading that box back correctly. The other thing
under test is the discipline the reasoner inherits from the verifier it
replaces: an unreachable model, or a reply with no boxes, invents no events.
"""

from __future__ import annotations

import pytest

from vision_worker.reason.cosmos import (
    CosmosReasoner,
    _parse_action_tail,
    _parse_boxes,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# A reply in exactly the shape the Phase-0 probe produced.
_REPLY = (
    "<ref>keys</ref><box>[300, 340, 690, 770]</box>\n"
    '[{"label": "keys", "action": "placed", "location": "on the kitchen table"}]'
)


def test_a_grounding_box_is_read_back_as_normalized_xyxy() -> None:
    boxes = _parse_boxes(_REPLY)
    assert len(boxes) == 1
    label, box = boxes[0]
    assert label == "keys"
    assert box.x_min == pytest.approx(0.300)
    assert box.y_min == pytest.approx(0.340)
    assert box.x_max == pytest.approx(0.690)
    assert box.y_max == pytest.approx(0.770)


def test_boxes_are_clamped_and_reordered() -> None:
    # Out-of-range and swapped corners should still yield a valid box.
    reply = "<ref>mug</ref><box>[900, 200, 100, 1200]</box>"
    [(label, box)] = _parse_boxes(reply)
    assert (box.x_min, box.x_max) == (pytest.approx(0.1), pytest.approx(0.9))
    assert (box.y_min, box.y_max) == (pytest.approx(0.2), pytest.approx(1.0))


def test_degenerate_boxes_are_dropped() -> None:
    assert _parse_boxes("<ref>keys</ref><box>[500, 500, 500, 500]</box>") == []


def test_the_action_tail_is_read_even_after_grounding_tags() -> None:
    actions = _parse_action_tail(_REPLY)
    assert actions["keys"]["action"] == "placed"
    assert actions["keys"]["location"] == "on the kitchen table"


def test_a_missing_or_malformed_tail_yields_no_actions() -> None:
    assert _parse_action_tail("<ref>keys</ref><box>[1,2,3,4]</box>") == {}
    assert _parse_action_tail("nonsense [not, json]") == {}


async def test_analyze_pairs_a_box_with_its_action(monkeypatch: pytest.MonkeyPatch) -> None:
    reasoner = CosmosReasoner()
    monkeypatch.setattr(reasoner, "_ask_blocking", lambda frames, labels: _REPLY)

    [event] = await reasoner.analyze((b"jpeg",), labels=("keys",))

    assert event.label == "keys"
    assert event.action == "placed"
    assert event.location_description == "on the kitchen table"
    assert event.is_memory_event
    assert event.box.x_min == pytest.approx(0.300)


async def test_boxes_for_unrequested_labels_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = (
        '<ref>laptop</ref><box>[100, 100, 200, 200]</box>\n[{"label":"laptop","action":"carried"}]'
    )
    reasoner = CosmosReasoner()
    monkeypatch.setattr(reasoner, "_ask_blocking", lambda frames, labels: reply)

    assert await reasoner.analyze((b"jpeg",), labels=("keys",)) == ()


async def test_a_box_without_a_json_entry_is_unknown_not_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoner = CosmosReasoner()
    monkeypatch.setattr(
        reasoner, "_ask_blocking", lambda frames, labels: "<ref>keys</ref><box>[1,2,3,4]</box>"
    )

    [event] = await reasoner.analyze((b"jpeg",), labels=("keys",))

    assert event.action == "unknown"
    assert not event.is_memory_event  # nothing to promote


async def test_an_unreachable_model_invents_no_events(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(frames: object, labels: object) -> str:
        raise OSError("connection refused")

    reasoner = CosmosReasoner()
    monkeypatch.setattr(reasoner, "_ask_blocking", boom)

    assert await reasoner.analyze((b"jpeg",), labels=("keys",)) == ()


async def test_empty_input_short_circuits() -> None:
    reasoner = CosmosReasoner()
    assert await reasoner.analyze((), labels=("keys",)) == ()
    assert await reasoner.analyze((b"jpeg",), labels=()) == ()
