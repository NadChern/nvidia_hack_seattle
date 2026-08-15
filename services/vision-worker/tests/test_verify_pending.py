"""PendingVerifications: the queue that keeps a slow verifier off the frame loop.

The property under test throughout is that `submit` returns *now*, whatever the
verifier is doing. `verify/pending.py` explains why that is correctness rather
than performance: the gateway keeps one latest-frame slot per subscriber, so a
pipeline that blocks does not slow the stream down, it makes the gateway discard
frames the stability machine will never know were missing.
"""

from __future__ import annotations

import asyncio

import pytest

from vision_worker.verify.pending import PendingVerifications

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_submitted_work_runs() -> None:
    pending = PendingVerifications(depth=4, concurrency=1)
    ran: list[str] = []

    async def work() -> None:
        ran.append("done")

    pending.submit(work)
    await pending.aclose()

    assert ran == ["done"]
    assert pending.completed == 1


async def test_submit_returns_before_the_work_finishes() -> None:
    """The whole point. A verifier that takes twenty seconds must not hold the
    frame loop for twenty seconds."""
    pending = PendingVerifications(depth=4, concurrency=1)
    release = asyncio.Event()
    finished = False

    async def slow() -> None:
        nonlocal finished
        await release.wait()
        finished = True

    pending.submit(slow)
    # No await between submit and here: control never left this coroutine, so
    # the work cannot have run.
    assert finished is False
    assert pending.pending == 1

    release.set()
    await pending.aclose()
    assert finished is True


async def test_a_full_queue_drops_the_oldest_and_counts_it() -> None:
    """Dropping is bad -- it loses a real event -- but blocking corrupts the
    state machine, so the queue drops and says so loudly."""
    pending = PendingVerifications(depth=2, concurrency=1)
    release = asyncio.Event()
    ran: list[str] = []

    async def blocker() -> None:
        await release.wait()

    def work(name: str):  # noqa: ANN202 - a closure factory, not an API
        async def run() -> None:
            ran.append(name)

        return run

    # The first submission is picked up by the worker immediately and blocks
    # there, leaving the queue itself free to fill.
    pending.submit(blocker)
    await asyncio.sleep(0)  # let the worker collect it

    pending.submit(work("oldest"))
    pending.submit(work("middle"))
    pending.submit(work("newest"))  # queue is full: "oldest" is displaced

    assert pending.dropped == 1

    release.set()
    await pending.aclose()

    assert ran == ["middle", "newest"], "the oldest queued candidate was dropped"


async def test_drain_waits_for_work_that_is_already_running() -> None:
    """An empty queue is not an idle pipeline. A replay printing its summary
    the moment the queue emptied would report the in-flight candidate as never
    confirmed."""
    pending = PendingVerifications(depth=4, concurrency=1)
    finished = False

    async def slow() -> None:
        nonlocal finished
        await asyncio.sleep(0.05)
        finished = True

    pending.submit(slow)
    await pending.drain()

    assert finished is True, "drain returned while the verifier was still working"


async def test_a_failing_verification_does_not_kill_the_worker() -> None:
    """One bad window must not stop every later candidate from being verified
    -- that would turn a single verifier error into a silently dead service."""
    pending = PendingVerifications(depth=4, concurrency=1)
    ran: list[str] = []

    async def boom() -> None:
        raise RuntimeError("the model server hung up")

    async def afterwards() -> None:
        ran.append("still running")

    pending.submit(boom)
    pending.submit(afterwards)
    await pending.aclose()

    assert pending.failed == 1
    assert pending.completed == 1
    assert ran == ["still running"]


async def test_pending_counts_queued_and_in_flight_together() -> None:
    pending = PendingVerifications(depth=4, concurrency=1)
    release = asyncio.Event()

    async def blocker() -> None:
        await release.wait()

    async def queued() -> None:
        pass

    pending.submit(blocker)
    await asyncio.sleep(0)
    pending.submit(queued)

    assert pending.pending == 2, "one running plus one queued"

    release.set()
    await pending.aclose()
    assert pending.pending == 0


async def test_work_submitted_after_close_is_dropped_not_queued() -> None:
    """Accepting work after the workers are gone would make a later `drain`
    wait forever for something nothing will ever run."""
    pending = PendingVerifications(depth=4, concurrency=1)
    ran: list[str] = []

    async def work() -> None:
        ran.append("ran")

    await pending.aclose()
    pending.submit(work)
    await pending.drain()

    assert ran == []
    assert pending.dropped == 1


async def test_closing_an_idle_queue_is_harmless() -> None:
    """Nothing was ever submitted, so no worker was ever started -- shutdown
    must not wait on or cancel a task that does not exist."""
    pending = PendingVerifications(depth=4, concurrency=1)
    await pending.aclose()
    assert pending.pending == 0


@pytest.mark.parametrize(("depth", "concurrency"), [(0, 1), (1, 0), (-1, 1)])
async def test_a_queue_that_could_never_work_is_refused_at_construction(
    depth: int, concurrency: int
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        PendingVerifications(depth=depth, concurrency=concurrency)
