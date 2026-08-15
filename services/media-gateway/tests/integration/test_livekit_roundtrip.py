"""The S01 spike's ten assertions, as a regression suite.

Each test below names the spike assertion it carries over
(docs/spikes/livekit-media-gateway/RESULTS.md). The spike proved these once
against a throwaway script; the point of re-running them here is that they now
run against the code that actually ships, so a refactor that breaks the epoch
rule or lets media leave the machine fails a test rather than a demo.

Run against a local server:

    docker compose -f compose.dev.yaml up -d livekit
    export VMA_TEST_LIVEKIT_URL=ws://127.0.0.1:7880
    uv run pytest tests/integration -m livekit
"""

from __future__ import annotations

import pytest
from visual_memory_media_contract.protocol import (
    AudioChunk,
    EpochStarted,
    SessionStarted,
    VideoFrame,
)

from .roundtrip import CYCLES, VIDEO_HEIGHT, VIDEO_WIDTH, Roundtrip

pytestmark = pytest.mark.livekit


def _epochs(messages: list[object]) -> list[EpochStarted]:
    return [message for message in messages if isinstance(message, EpochStarted)]


# --- 1. invalid_token_rejected -------------------------------------------


def test_an_invalid_token_cannot_join_a_room(roundtrip: Roundtrip) -> None:
    """A tampered signature must be refused by the server, not by us."""
    assert roundtrip.invalid_livekit_token_rejected


def test_minting_a_token_requires_the_internal_credential(roundtrip: Roundtrip) -> None:
    """The gateway half of the same rule: token minting is not open."""
    assert roundtrip.unauthenticated_session_status == 401


# --- 2. three_publish_cycles_completed -----------------------------------


def test_every_publish_cycle_completed(roundtrip: Roundtrip) -> None:
    assert len(roundtrip.cycles) == CYCLES
    assert all(cycle.video_published > 0 for cycle in roundtrip.cycles)
    assert all(cycle.audio_published > 0 for cycle in roundtrip.cycles)


# --- 3 & 4. new_{video,audio}_track_sid_per_cycle ------------------------


def test_each_rejoin_starts_a_new_video_epoch(roundtrip: Roundtrip) -> None:
    """The finding the whole contract rests on.

    Identity is unchanged across all three cycles; the track SID is not. If
    this ever stops holding, `epoch_id` is the wrong boundary and every
    consumer's tracker-reset rule is wrong with it.
    """
    epochs = _epochs(roundtrip.video)

    assert len(epochs) == CYCLES
    assert len({epoch.epoch_id for epoch in epochs}) == CYCLES
    assert len({epoch.participant_identity for epoch in epochs}) == 1
    assert all(epoch.epoch_id == epoch.track_sid for epoch in epochs)


def test_each_rejoin_starts_a_new_audio_epoch(roundtrip: Roundtrip) -> None:
    epochs = _epochs(roundtrip.audio)

    assert len(epochs) == CYCLES
    assert len({epoch.epoch_id for epoch in epochs}) == CYCLES


def test_video_and_audio_epochs_are_distinct_but_share_a_session(
    roundtrip: Roundtrip,
) -> None:
    """Separate tracks, so separate epochs -- correlated only by session."""
    video = _epochs(roundtrip.video)
    audio = _epochs(roundtrip.audio)

    assert {epoch.epoch_id for epoch in video}.isdisjoint({epoch.epoch_id for epoch in audio})
    assert len({epoch.session_id for epoch in video + audio}) == 1


# --- 5 & 6. worker_received_every_{video,audio}_track --------------------


def test_every_video_epoch_delivered_frames(roundtrip: Roundtrip) -> None:
    """Not just that the epochs opened -- that media flowed on each."""
    counts: dict[str, int] = {epoch.epoch_id: 0 for epoch in _epochs(roundtrip.video)}
    for message in roundtrip.video:
        if isinstance(message, VideoFrame):
            counts[message.epoch_id] += 1

    assert len(counts) == CYCLES
    assert all(count > 0 for count in counts.values()), counts


def test_every_audio_epoch_delivered_chunks(roundtrip: Roundtrip) -> None:
    counts: dict[str, int] = {epoch.epoch_id: 0 for epoch in _epochs(roundtrip.audio)}
    for message in roundtrip.audio:
        if isinstance(message, AudioChunk):
            counts[message.epoch_id] += 1

    assert len(counts) == CYCLES
    assert all(count > 0 for count in counts.values()), counts


def test_a_subscriber_is_greeted_before_any_media(roundtrip: Roundtrip) -> None:
    assert roundtrip.video[0].type == "stream_hello"
    assert isinstance(roundtrip.video[1], SessionStarted)


# --- 7. bounded_sampler_processed_frames ---------------------------------


def test_the_sampler_sheds_load_rather_than_stalling_ingest(roundtrip: Roundtrip) -> None:
    """Publishing at 10 FPS into a 2 FPS sampler must drop, not queue.

    A slow consumer applying backpressure to LiveKit ingest is the failure this
    design exists to prevent, and it looks identical to a healthy run until you
    check that frames were actually discarded.
    """
    video = roundtrip.status_final["metrics"]["video"]

    assert video["relayed"] > 0
    assert video["relayed"] < video["received"]
    assert video["dropped_before_sampling"] > 0


def test_consumers_can_see_their_own_sampling_gaps(roundtrip: Roundtrip) -> None:
    """`dropped_since_previous` is how a consumer measures this without polling."""
    frames = [m for m in roundtrip.video if isinstance(m, VideoFrame)]

    assert frames
    assert sum(frame.dropped_since_previous for frame in frames) > 0


# --- 8. decoded_video_dimensions_preserved -------------------------------


def test_relayed_frames_are_the_published_size_and_decode_to_it(
    roundtrip: Roundtrip,
) -> None:
    """The dimension guard's whole purpose: nothing odd-sized reaches a detector."""
    frames = [m for m in roundtrip.video if isinstance(m, VideoFrame)]

    assert frames
    for frame in frames:
        assert (frame.width, frame.height) == (VIDEO_WIDTH, VIDEO_HEIGHT)
        assert frame.rgb.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)


def test_sequence_restarts_at_zero_for_each_epoch(roundtrip: Roundtrip) -> None:
    by_epoch: dict[str, list[int]] = {}
    for message in roundtrip.video:
        if isinstance(message, VideoFrame):
            by_epoch.setdefault(message.epoch_id, []).append(message.sequence)

    assert len(by_epoch) == CYCLES
    for sequences in by_epoch.values():
        assert sequences[0] == 0
        assert sequences == sorted(sequences)


def test_audio_pts_is_continuous_within_an_epoch(roundtrip: Roundtrip) -> None:
    """A gap must be arithmetic, not inferred from a message count."""
    chunks = [m for m in roundtrip.audio if isinstance(m, AudioChunk)]

    assert chunks
    for earlier, later in zip(chunks[:-1], chunks[1:], strict=True):
        if earlier.epoch_id == later.epoch_id:
            assert later.pts_samples == earlier.pts_samples + earlier.samples


def test_no_audio_was_dropped(roundtrip: Roundtrip) -> None:
    """Audio drops corrupt transcription invisibly; the policy is to close loudly."""
    audio = roundtrip.status_final["metrics"]["audio"]

    assert audio["subscribers_closed_for_backpressure"] == 0


# --- 9. return_audio_received_every_cycle --------------------------------


def test_the_device_hears_the_assistant_on_every_cycle(roundtrip: Roundtrip) -> None:
    """Closes the loop with no Speech Service and no glasses."""
    assert all(cycle.return_audio_frames > 0 for cycle in roundtrip.cycles), [
        cycle.return_audio_frames for cycle in roundtrip.cycles
    ]


# --- 10. server_has_no_nonlocal_established_connection -------------------


def test_the_gateway_talks_to_nothing_off_this_machine(roundtrip: Roundtrip) -> None:
    """The privacy assertion, and the reason Cloudflare Tunnel was rejected.

    Raw first-person camera video through a third-party edge is the precise
    failure docs/07-Privacy-and-Security.md promises not to have. Anything that
    routes media or signaling off-box turns this red, which is the point.
    """
    assert roundtrip.nonlocal_connections == []


# --- Status surface -------------------------------------------------------


def test_status_reports_a_live_publisher_then_a_clean_shutdown(
    roundtrip: Roundtrip,
) -> None:
    live = roundtrip.status_live
    final = roundtrip.status_final

    assert live["ready"] is True
    assert [session["publisher_present"] for session in live["sessions"]] == [True]
    assert final["sessions"] == []
    assert final["metrics"]["sessions"]["created"] == 1
    assert final["metrics"]["sessions"]["ended"] == 1
    # Six: video and audio, once per cycle.
    assert final["metrics"]["epochs"]["started"] == CYCLES * 2


def test_the_dimension_histogram_records_the_real_camera_size(
    roundtrip: Roundtrip,
) -> None:
    """The field that distinguishes 'no camera' from 'wrong size'."""
    epochs = [epoch for epoch in roundtrip.status_live["epochs"] if "guard" in epoch]

    assert epochs
    assert any(f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}" in epoch["guard"]["dimensions"] for epoch in epochs)


def test_no_secret_reaches_the_status_surface(roundtrip: Roundtrip) -> None:
    rendered = str(roundtrip.status_final)

    assert "eyJ" not in rendered
    assert "secret" not in rendered.lower()
