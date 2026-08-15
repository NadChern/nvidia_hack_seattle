"""Session registry: bounded, TTL-swept, and allowlist-aware."""

import datetime as dt

import pytest

from media_gateway.config import Settings
from media_gateway.domain.ids import lifecycle_idempotency_key, new_session_id
from media_gateway.domain.session import SessionRegistry
from media_gateway.errors import CapacityError, ForbiddenError, NotFoundError

T0 = dt.datetime(2026, 7, 30, 18, 0, 0, tzinfo=dt.UTC)


def a_registry(**overrides: object) -> SessionRegistry:
    return SessionRegistry(Settings(media_source="scripted", **overrides))  # type: ignore[arg-type]


def test_created_session_has_a_prefixed_id_and_derived_room() -> None:
    registry = a_registry(room_prefix="vma")
    session = registry.create(device_id="glasses-01", at=T0)

    assert session.session_id.startswith("sess_")
    assert session.room == f"vma-{session.session_id}"
    assert session.active
    assert not session.publisher_present


def test_concurrency_limit_is_enforced() -> None:
    registry = a_registry(max_concurrent_sessions=1)
    registry.create(device_id="glasses-01", at=T0)

    with pytest.raises(CapacityError, match="no session slots"):
        registry.create(device_id="glasses-02", at=T0)


def test_expired_sessions_free_their_slot() -> None:
    """A crashed publisher must not hold the budget forever."""
    registry = a_registry(max_concurrent_sessions=1, session_ttl_s=60)
    registry.create(device_id="glasses-01", at=T0)

    later = T0 + dt.timedelta(seconds=61)
    session = registry.create(device_id="glasses-02", at=later)

    assert session.device_id == "glasses-02"
    assert len(registry) == 1


def test_sweep_reports_what_it_removed() -> None:
    registry = a_registry(session_ttl_s=60)
    registry.create(device_id="glasses-01", at=T0)

    expired = registry.sweep(now=T0 + dt.timedelta(seconds=61))

    assert len(expired) == 1
    assert expired[0].ended_at is not None
    assert len(registry) == 0


def test_touching_a_session_defers_expiry() -> None:
    registry = a_registry(session_ttl_s=60)
    session = registry.create(device_id="glasses-01", at=T0)

    session.touch(at=T0 + dt.timedelta(seconds=50))

    assert registry.sweep(now=T0 + dt.timedelta(seconds=100)) == []
    assert len(registry) == 1


def test_unlisted_device_is_rejected_without_echoing_the_allowlist() -> None:
    registry = a_registry(device_id_allowlist=("glasses-01",))

    with pytest.raises(ForbiddenError) as caught:
        registry.create(device_id="glasses-99", at=T0)

    assert "glasses-01" not in str(caught.value.context)


def test_empty_allowlist_permits_any_device() -> None:
    registry = a_registry(device_id_allowlist=())

    assert registry.create(device_id="anything", at=T0).device_id == "anything"


def test_unknown_session_lookup_is_explicit() -> None:
    registry = a_registry()

    with pytest.raises(NotFoundError, match="unknown session"):
        registry.get("sess_missing")

    assert registry.find("sess_missing") is None


def test_ending_a_session_removes_it() -> None:
    registry = a_registry()
    session = registry.create(device_id="glasses-01", at=T0)
    session.publisher_present = True

    ended = registry.end(session.session_id, at=T0)

    assert not ended.active
    assert not ended.publisher_present
    assert len(registry) == 0


def test_caller_supplied_session_id_is_honoured() -> None:
    """The Memory Service becomes the id authority once it exists."""
    registry = a_registry()

    session = registry.create(device_id="glasses-01", session_id="sess_from_memory", at=T0)

    assert session.session_id == "sess_from_memory"


def test_session_ids_are_unique_and_sortable() -> None:
    ids = [new_session_id() for _ in range(50)]

    assert len(set(ids)) == 50
    assert ids == sorted(ids)


def test_lifecycle_key_is_deterministic() -> None:
    """A restart mid-teardown must not apply the same transition twice."""
    key = lifecycle_idempotency_key(
        device_id="glasses-01",
        session_id="sess_1",
        scope_id="TR_VCaaa",
        action="track_lost",
    )

    assert key == "glasses-01/sess_1/TR_VCaaa/track_lost"
    assert key == lifecycle_idempotency_key(
        device_id="glasses-01",
        session_id="sess_1",
        scope_id="TR_VCaaa",
        action="track_lost",
    )


def test_a_session_nobody_joined_frees_its_slot_quickly() -> None:
    """Otherwise two failed joins lock the gateway out for the full TTL.

    A device that cannot reach LiveKit asks for a new session on every retry.
    At the default budget of two, that exhausts the slots and every later
    attempt gets `429 capacity_exhausted` -- observed repeatedly on the X3 Pro
    while its ICE path was broken, with no way to recover but a manual delete.
    """
    registry = a_registry(max_concurrent_sessions=2, session_ttl_s=3600, unclaimed_session_ttl_s=90)
    registry.create(device_id="glasses-01", at=T0)
    registry.create(device_id="glasses-01", at=T0)

    swept = registry.sweep(now=T0 + dt.timedelta(seconds=91))

    assert len(swept) == 2
    assert len(registry) == 0
    # The slot is genuinely free again, not merely reported as free.
    registry.create(device_id="glasses-01", at=T0 + dt.timedelta(seconds=92))


def test_a_wearer_who_drops_keeps_the_long_ttl() -> None:
    """`publisher_present` goes false on any blip; `ever_published` does not.

    Reclaiming on the short clock here would evict a live wearer mid-session
    the first time their link wobbled.
    """
    registry = a_registry(session_ttl_s=3600, unclaimed_session_ttl_s=90)
    session = registry.create(device_id="glasses-01", at=T0)
    session.ever_published = True
    session.publisher_present = False

    assert registry.sweep(now=T0 + dt.timedelta(seconds=91)) == []
    assert len(registry) == 1
