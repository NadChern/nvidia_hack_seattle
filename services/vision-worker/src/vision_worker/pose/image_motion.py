"""Background ego-motion from consecutive frames, with no model and no extra
dependency -- the no-glasses, no-GPU default `PoseSource`.

The inversion `domain/stability.py` exploits: a held object stays roughly
fixed in the frame while the background sweeps past; a resting object's
screen motion tracks the background's, since both come from ego-motion
alone. This module estimates that background motion via phase correlation --
an FFT-based global-translation estimate between two frames, the same
principle behind `skimage.registration.phase_cross_correlation`. Pure numpy:
no OpenCV, no torch, so it runs identically on a laptop webcam, in CI, and on
the browser publisher.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from visual_memory_vision_contract.protocol import Point2D

#: Luminance weights (ITU-R BT.601), matching PyAV/Pillow's own RGB-to-gray
#: convention closely enough for a motion estimate that only needs relative
#: contrast, not colorimetric accuracy.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def _to_grayscale(frame_rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    return frame_rgb[..., :3].astype(np.float64) @ _LUMA_WEIGHTS


def _phase_correlation_shift(
    previous: NDArray[np.float64], current: NDArray[np.float64]
) -> Point2D:
    """The dominant global translation between `previous` and `current`,
    normalized to frame fractions (so it composes directly with a
    normalized-coordinate `Detection.centroid`)."""
    height, width = previous.shape
    # A Hanning window suppresses the FFT's implicit edge-wraparound
    # artifacts, which would otherwise masquerade as spurious high-frequency
    # "motion" at the frame boundary.
    window = np.outer(np.hanning(height), np.hanning(width))

    f0 = np.fft.fft2(previous * window)
    f1 = np.fft.fft2(current * window)
    # conj(f0) * f1, not the other order: the peak of this cross-power
    # spectrum sits at the displacement *from* `previous` *to* `current`,
    # matching the sign convention of `np.roll`'s own `shift` argument.
    cross_power = np.conj(f0) * f1
    magnitude = np.abs(cross_power)
    normalized = np.divide(
        cross_power, magnitude, out=np.zeros_like(cross_power), where=magnitude > 1e-10
    )
    correlation = np.abs(np.fft.ifft2(normalized))

    peak_row, peak_col = np.unravel_index(np.argmax(correlation), correlation.shape)
    # The FFT's frequency-domain peak wraps around at the Nyquist frequency,
    # so a shift larger than half the frame appears as a negative shift the
    # other way -- unwrap it before returning.
    dy = peak_row if peak_row <= height // 2 else peak_row - height
    dx = peak_col if peak_col <= width // 2 else peak_col - width

    return Point2D(x=dx / width, y=dy / height)


class ImageMotionPose:
    """Compares each frame against the previous one. `reset()` on every
    `epoch_started` -- comparing across a cut would estimate a meaningless
    shift between two unrelated frames.
    """

    def __init__(self) -> None:
        self._previous: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self._previous = None

    def observe(self, frame_rgb: NDArray[np.uint8]) -> Point2D | None:
        gray = _to_grayscale(frame_rgb)
        if self._previous is None or self._previous.shape != gray.shape:
            # First frame since a reset, or a dimension change mid-epoch
            # (should not happen under the gateway's dimension guard, but
            # this is cheaper and more honest than assuming it never will).
            self._previous = gray
            return None

        shift = _phase_correlation_shift(self._previous, gray)
        self._previous = gray
        return shift


__all__ = ["ImageMotionPose"]
