"""The S01 spike's sampler logic, tested with no LiveKit and no real time."""

import pytest

from media_gateway.domain.sampling import SUSTAINED_RUN, DimensionGuard, LatestSlot, Pacer


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


# --- LatestSlot ----------------------------------------------------------


def test_slot_returns_the_only_item() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.offer(1)

    assert slot.take() == 1
    assert slot.take() is None


def test_slot_keeps_the_newest_and_counts_the_displaced() -> None:
    """A slow consumer must lose stale frames, not the newest one."""
    slot: LatestSlot[int] = LatestSlot()
    for value in (1, 2, 3):
        slot.offer(value)

    assert slot.take() == 3
    assert slot.dropped == 2
    assert slot.offered == 3
    assert slot.taken == 1


def test_slot_never_blocks_the_producer() -> None:
    """The whole point: ingest speed is independent of consumer speed."""
    slot: LatestSlot[int] = LatestSlot()
    for value in range(10_000):
        slot.offer(value)

    assert slot.take() == 9_999
    assert slot.dropped == 9_999


def test_taking_then_offering_does_not_count_a_drop() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.offer(1)
    slot.take()
    slot.offer(2)

    assert slot.dropped == 0
    assert slot.take() == 2


def test_clear_discards_without_counting_a_take() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.offer(1)
    slot.clear()

    assert slot.take() is None
    assert slot.taken == 0
    assert not slot.occupied


def test_slot_can_hold_falsy_items() -> None:
    """Occupancy must be tracked separately from truthiness."""
    slot: LatestSlot[int] = LatestSlot()
    slot.offer(0)

    assert slot.occupied
    assert slot.take() == 0


# --- DimensionGuard ------------------------------------------------------


def a_guard(mode: str = "strict") -> DimensionGuard:
    return DimensionGuard(
        mode=mode,  # type: ignore[arg-type]
        expected_width=320,
        expected_height=180,
    )


def test_expected_frames_are_admitted() -> None:
    guard = a_guard()

    assert guard.admit(320, 180)
    assert guard.admitted == 1
    assert guard.rejected == 0


def test_transition_frames_are_rejected() -> None:
    """The spike saw transient 8x8 frames during simulcast adaptation."""
    guard = a_guard()

    assert not guard.admit(8, 8)
    assert guard.rejected == 1
    assert guard.admitted == 0


def test_histogram_records_rejected_sizes_too() -> None:
    """Tallied before the decision, so the histogram is a complete record."""
    guard = a_guard()
    for _ in range(3):
        guard.admit(320, 180)
    for _ in range(2):
        guard.admit(8, 8)

    assert guard.histogram == {"320x180": 3, "8x8": 2}


def test_strict_mode_reports_the_configured_size() -> None:
    guard = a_guard()

    assert guard.latched == (320, 180)


def test_first_frame_wins_latches_an_unexpected_resolution() -> None:
    """Real glasses arrive at a resolution nobody configured."""
    guard = a_guard("first_frame_wins")

    assert guard.admit(1280, 720)
    assert guard.latched == (1280, 720)
    assert guard.admit(1280, 720)
    assert not guard.admit(320, 180)
    assert guard.admitted == 2
    assert guard.rejected == 1


def test_reset_forgets_the_latched_size() -> None:
    """A rejoin may bring a different camera, so the latch cannot survive."""
    guard = a_guard("first_frame_wins")
    guard.admit(1280, 720)

    guard.reset()

    assert guard.latched is None
    assert guard.histogram == {}
    assert guard.admit(640, 480)
    assert guard.latched == (640, 480)


def test_strict_mode_reset_keeps_the_configured_size() -> None:
    guard = a_guard()
    guard.admit(8, 8)

    guard.reset()

    assert guard.latched == (320, 180)
    assert guard.rejected == 0


# --- Pacer ---------------------------------------------------------------


@pytest.mark.anyio
async def test_pacer_sleeps_one_interval_when_on_schedule() -> None:
    clock = FakeClock()
    pacer = Pacer(0.5, now=clock, sleep=clock.sleep)

    for _ in range(4):
        await pacer.wait()

    assert clock.slept == [0.5, 0.5, 0.5, 0.5]
    assert pacer.missed == 0


@pytest.mark.anyio
async def test_pacer_uses_absolute_deadlines_so_delay_does_not_accumulate() -> None:
    """Relative sleeps drift by seconds over a long session at 2 FPS."""
    clock = FakeClock()
    pacer = Pacer(1.0, now=clock, sleep=clock.sleep)

    await pacer.wait()
    clock.now += 0.25  # the consumer took a while
    await pacer.wait()

    # Second sleep is shortened to hold the original cadence.
    assert clock.slept == [1.0, 0.75]
    assert clock.now == pytest.approx(2.0)


@pytest.mark.anyio
async def test_pacer_skips_missed_deadlines_rather_than_bursting() -> None:
    """Catching up would defeat the point of a sampling rate."""
    clock = FakeClock()
    pacer = Pacer(1.0, now=clock, sleep=clock.sleep)

    await pacer.wait()  # now=1.0, next deadline 2.0
    clock.now += 3.5  # inference stalled; now=4.5
    await pacer.wait()

    # Deadlines at 2.0, 3.0 and 4.0 elapsed while the consumer was stalled.
    assert pacer.missed == 3
    assert clock.slept[-1] == 0  # yielded, did not burst three frames
    # The schedule is ahead of the clock again rather than still behind.
    await pacer.wait()
    assert clock.slept[-1] > 0


@pytest.mark.anyio
async def test_pacer_reset_restarts_the_schedule() -> None:
    clock = FakeClock()
    pacer = Pacer(1.0, now=clock, sleep=clock.sleep)

    await pacer.wait()
    clock.now += 10.0
    pacer.reset()
    await pacer.wait()

    assert clock.slept == [1.0, 1.0]
    assert pacer.missed == 0


def test_pacer_rejects_a_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Pacer(0)


# --- sustained: latching a size that actually holds ---------------------------


def _feed(guard: DimensionGuard, width: int, height: int, count: int) -> int:
    """Offer `count` frames of one size; return how many were admitted."""
    return sum(1 for _ in range(count) if guard.admit(width, height))


def test_sustained_ignores_a_size_that_does_not_persist() -> None:
    """One frame proves nothing. The spike's own observation was transient 8x8
    frames during adaptation; latching on the first would enshrine one."""
    guard = a_guard("sustained")

    assert not guard.admit(8, 8)
    assert guard.latched is None
    assert guard.rejected == 1


def test_sustained_latches_once_a_size_holds() -> None:
    guard = a_guard("sustained")

    admitted = _feed(guard, 1280, 720, SUSTAINED_RUN + 5)

    assert guard.latched == (1280, 720)
    # The run itself is rejected -- nothing has been proven while it is still
    # being proven -- and everything after it is admitted.
    assert admitted == 6


def test_a_ramping_encoder_ends_up_at_the_size_it_settles_on() -> None:
    """The bug this mode exists for, replayed from a real session.

    LiveKit ramps its encoder up to the negotiated resolution over tens of
    seconds. `first_frame_wins` latched 320x180 -- the bottom rung -- and then
    rejected 5002 of 5581 frames for the rest of the session, so the vision
    worker saw roughly two percent of the video, at postage-stamp size, and
    found nothing in it. Nothing reported a problem.
    """
    ramp = [(320, 180, 112), (480, 270, 119), (640, 360, 61), (960, 540, 298)]

    guard = a_guard("sustained")
    for width, height, count in ramp:
        _feed(guard, width, height, count)
    steady = _feed(guard, 1280, 720, 500)

    assert guard.latched == (1280, 720), "the size the stream settled on"
    assert steady >= 400, "and the steady state is admitted, not thrown away"


def test_the_old_mode_still_fails_the_way_it_did() -> None:
    """Kept as a comparison, and as the reason `first_frame_wins` must not be
    the default: same input, opposite outcome."""
    guard = a_guard("first_frame_wins")

    _feed(guard, 320, 180, 112)
    steady = _feed(guard, 1280, 720, 500)

    assert guard.latched == (320, 180)
    assert steady == 0, "every real frame rejected, for the whole session"


def test_sustained_re_latches_when_the_publisher_changes_resolution() -> None:
    """Refusing to would reject every frame for the rest of the epoch, which is
    the failure this mode exists to prevent -- just deferred."""
    guard = a_guard("sustained")
    _feed(guard, 640, 360, SUSTAINED_RUN)
    assert guard.latched == (640, 360)

    admitted = _feed(guard, 1280, 720, SUSTAINED_RUN + 3)

    assert guard.latched == (1280, 720)
    assert guard.relatched == 1
    assert admitted == 4


def test_a_brief_glitch_does_not_unseat_a_latched_size() -> None:
    """A few odd frames mid-stream are the transients the guard exists to
    reject, not a resolution change."""
    guard = a_guard("sustained")
    _feed(guard, 1280, 720, SUSTAINED_RUN)

    assert not guard.admit(8, 8)
    assert not guard.admit(8, 8)
    assert guard.admit(1280, 720)

    assert guard.latched == (1280, 720)
    assert guard.relatched == 0


def test_sustained_reset_forgets_the_run_as_well_as_the_latch() -> None:
    """A partial run must not carry into the next epoch, or a rejoin could
    latch on a size proven by frames belonging to the previous one."""
    guard = a_guard("sustained")
    _feed(guard, 1280, 720, SUSTAINED_RUN - 1)

    guard.reset()

    assert guard.latched is None
    assert guard.relatched == 0
    # One more frame of the old size must not be enough to complete the run.
    assert not guard.admit(1280, 720)
    assert guard.latched is None
