"""ImageMotionPose: phase-correlation background motion, checked against
known synthetic shifts rather than merely "it runs"."""

from __future__ import annotations

import numpy as np

from vision_worker.pose.base import PoseSource
from vision_worker.pose.image_motion import ImageMotionPose

_RNG = np.random.default_rng(seed=42)


def a_textured_frame(size: int = 64) -> np.ndarray:
    """Random noise, not a uniform image -- phase correlation needs texture
    to have any signal to lock onto."""
    return _RNG.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_the_first_frame_after_a_reset_returns_none() -> None:
    pose = ImageMotionPose()

    result = pose.observe(a_textured_frame())

    assert result is None


def test_two_identical_frames_estimate_zero_motion() -> None:
    pose = ImageMotionPose()
    frame = a_textured_frame()
    pose.observe(frame)

    shift = pose.observe(frame)

    assert shift is not None
    assert abs(shift.x) < 1e-6
    assert abs(shift.y) < 1e-6


def test_a_known_circular_shift_is_recovered_within_a_pixel() -> None:
    """np.roll is an exact circular shift -- what phase correlation is
    fundamentally built to detect -- so the recovered shift should match the
    true one closely, not just have the right sign."""
    size = 64
    frame = a_textured_frame(size)
    dx_pixels, dy_pixels = 5, -3
    shifted = np.roll(frame, shift=(dy_pixels, dx_pixels), axis=(0, 1))

    pose = ImageMotionPose()
    pose.observe(frame)
    shift = pose.observe(shifted)

    assert shift is not None
    assert abs(shift.x * size - dx_pixels) <= 1.0
    assert abs(shift.y * size - dy_pixels) <= 1.0


def test_reset_forgets_the_previous_frame() -> None:
    pose = ImageMotionPose()
    pose.observe(a_textured_frame())

    pose.reset()
    result = pose.observe(a_textured_frame())

    assert result is None


def test_satisfies_the_pose_source_protocol() -> None:
    pose: PoseSource = ImageMotionPose()
    assert pose is not None
