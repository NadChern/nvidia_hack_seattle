"""Encodes a confirmed candidate's evidence window into a short mp4 clip.

Only ever called after a `confirmed` `VerifierResult` -- `emit/memory.py`
uploads the result alongside a still frame, per the plan: "a still frame is
uploaded alongside, so a clip failure degrades to today's behaviour rather
than losing the evidence." PyAV does CPU software encoding (`libx264`), so
this needs no GPU and stays a core dependency for every profile, including
`ci` and `dev-macos`.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import av
from visual_memory_media_contract.images import decode_video_payload

from vision_worker.evidence.ring import BufferedFrame

#: `AnsweredPlacement.evidence_media_type` in memory-contract -- lets a
#: client choose `<video>` over `<img>` without sniffing.
CLIP_MEDIA_TYPE = "video/mp4"


class ClipEncodeError(RuntimeError):
    """A window could not be encoded, or a still frame could not be selected."""


def encode_clip(frames: Sequence[BufferedFrame], *, fps: float) -> bytes:
    """Mux `frames` into an mp4, encoded with `libx264`.

    All frames must share one width and height -- true within one epoch,
    since the gateway's dimension guard latches the first frame's size and
    rejects a mismatch (see `VMA_DIMENSION_GUARD_MODE`).
    """
    if not frames:
        raise ClipEncodeError("cannot encode an empty evidence window")

    width = frames[0].width
    height = frames[0].height
    mismatched = next((f for f in frames if f.width != width or f.height != height), None)
    if mismatched is not None:
        raise ClipEncodeError(
            f"evidence window has mixed frame sizes: {width}x{height} vs "
            f"{mismatched.width}x{mismatched.height}"
        )

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        # PyAV's bundled stubs cover the decode path cleanly (see
        # media_gateway.publisher.sources, which only decodes) but leave the
        # encode path -- add_stream's overloads, and Packet's generic type --
        # partially unknown. Targeted ignores, matching this repository's own
        # precedent at media_gateway/publisher/publish.py:121, rather than a
        # blanket suppression.
        stream = container.add_stream(  # pyright: ignore[reportUnknownMemberType]
            "libx264", rate=max(1, round(fps))
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for buffered in frames:
            rgb = decode_video_payload(
                buffered.payload,
                encoding="jpeg",
                width=buffered.width,
                height=buffered.height,
                pixel_format="rgb",
            )
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                video_frame
            ):
                container.mux(packet)  # pyright: ignore[reportUnknownMemberType]

        for packet in stream.encode():  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            container.mux(packet)  # pyright: ignore[reportUnknownMemberType]
    finally:
        container.close()

    return buffer.getvalue()


def select_still_frame(frames: Sequence[BufferedFrame]) -> BufferedFrame:
    """The still-frame fallback: the last buffered frame in the window, on
    the theory that it is closest to the moment of confirmation."""
    if not frames:
        raise ClipEncodeError("cannot select a still frame from an empty window")
    return frames[-1]


__all__ = ["CLIP_MEDIA_TYPE", "ClipEncodeError", "encode_clip", "select_still_frame"]
