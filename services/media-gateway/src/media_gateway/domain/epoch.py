"""Media epochs.

An epoch is one continuous run of one media track. The S01 spike established
that a rejoin produces a new LiveKit track SID even when the participant
identity is unchanged, so the track SID -- not the identity -- is the epoch
boundary. This is what makes the rule in docs/02-Model-Landscape.md
("tracker IDs are scoped to a media epoch and must not survive reconnects")
mechanically enforceable rather than a convention.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from visual_memory_media_contract.protocol import EpochEndReason, StreamKind

from media_gateway.config import Settings
from media_gateway.domain.sampling import DimensionGuard


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class MediaEpoch:
    """One track's lifetime, and the counters reported for it."""

    epoch_id: str
    session_id: str
    stream_kind: StreamKind
    track_sid: str
    participant_identity: str
    started_at: dt.datetime
    ended_at: dt.datetime | None = None
    end_reason: EpochEndReason | None = None

    #: Restarts at 0 for every epoch, so a consumer that resets on
    #: `epoch_started` sees a clean sequence.
    next_sequence: int = 0
    received: int = 0
    relayed: int = 0
    guard: DimensionGuard | None = None

    #: Cumulative samples emitted, for audio pts arithmetic.
    pts_samples: int = 0

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def take_sequence(self) -> int:
        """Return the next sequence number for this epoch."""
        sequence = self.next_sequence
        self.next_sequence += 1
        return sequence

    def end(self, reason: EpochEndReason, *, at: dt.datetime | None = None) -> None:
        if self.ended_at is not None:
            return
        self.ended_at = at or _utcnow()
        self.end_reason = reason


class EpochRegistry:
    """Tracks the active epoch per (session, stream kind).

    Opening an epoch for a stream that already has one ends the previous epoch
    first, so a missed unsubscribe cannot leave two epochs claiming the same
    stream.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._epochs: dict[str, MediaEpoch] = {}
        self._active: dict[tuple[str, str], str] = {}

    def _guard_for(self, stream_kind: StreamKind) -> DimensionGuard | None:
        if stream_kind != "video":
            return None
        return DimensionGuard(
            mode=self._settings.dimension_guard_mode,
            expected_width=self._settings.expected_video_width,
            expected_height=self._settings.expected_video_height,
        )

    def begin(
        self,
        *,
        session_id: str,
        stream_kind: StreamKind,
        track_sid: str,
        participant_identity: str,
        at: dt.datetime | None = None,
    ) -> tuple[MediaEpoch, MediaEpoch | None]:
        """Start an epoch, returning it and any epoch it displaced."""
        displaced = self.end_active(
            session_id=session_id,
            stream_kind=stream_kind,
            reason="track_unsubscribed",
            at=at,
        )
        epoch = MediaEpoch(
            epoch_id=track_sid,
            session_id=session_id,
            stream_kind=stream_kind,
            track_sid=track_sid,
            participant_identity=participant_identity,
            started_at=at or _utcnow(),
            guard=self._guard_for(stream_kind),
        )
        self._epochs[epoch.epoch_id] = epoch
        self._active[(session_id, stream_kind)] = epoch.epoch_id
        return epoch, displaced

    def active_for(self, session_id: str, stream_kind: StreamKind) -> MediaEpoch | None:
        epoch_id = self._active.get((session_id, stream_kind))
        return self._epochs.get(epoch_id) if epoch_id else None

    def get(self, epoch_id: str) -> MediaEpoch | None:
        return self._epochs.get(epoch_id)

    def end_active(
        self,
        *,
        session_id: str,
        stream_kind: StreamKind,
        reason: EpochEndReason,
        at: dt.datetime | None = None,
    ) -> MediaEpoch | None:
        """End the active epoch for a stream, if there is one."""
        epoch = self.active_for(session_id, stream_kind)
        if epoch is None:
            return None
        epoch.end(reason, at=at)
        self._active.pop((session_id, stream_kind), None)
        return epoch

    def end_session(
        self,
        session_id: str,
        *,
        reason: EpochEndReason,
        at: dt.datetime | None = None,
    ) -> list[MediaEpoch]:
        """End every active epoch belonging to a session."""
        ended: list[MediaEpoch] = []
        for (owner, stream_kind), epoch_id in list(self._active.items()):
            if owner != session_id:
                continue
            epoch = self._epochs.get(epoch_id)
            if epoch is not None:
                epoch.end(reason, at=at)
                ended.append(epoch)
            self._active.pop((owner, stream_kind), None)
        return ended

    def active(self) -> list[MediaEpoch]:
        return [
            epoch
            for epoch_id in self._active.values()
            if (epoch := self._epochs.get(epoch_id)) is not None
        ]

    def forget_session(self, session_id: str) -> None:
        """Drop a finished session's epochs so the registry stays bounded."""
        for epoch_id, epoch in list(self._epochs.items()):
            if epoch.session_id == session_id:
                self._epochs.pop(epoch_id, None)
        for key in [key for key in self._active if key[0] == session_id]:
            self._active.pop(key, None)


__all__ = ["EpochRegistry", "MediaEpoch"]
