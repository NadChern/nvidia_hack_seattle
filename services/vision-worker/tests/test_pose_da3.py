"""The DA3 adapter's pure conversions, without DA3.

Loading the model needs a checkpoint, a GPU, and a package this service
cannot declare (see `pose/da3.py`'s module docstring). The conversions
between DA3's output conventions and the domain layer's are pure arithmetic
though, and they are exactly where a silent, plausible-looking error lives:
a transposed rotation or a z-depth mistaken for a ray range produces world
points that are wrong and smooth rather than wrong and obvious.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision_worker.pose.da3 import (
    MIN_VIEWS,
    Da3WindowPose,
    _frame_geometry,
    _matrix_to_quaternion,
    _subsample,
)


def test_a_window_longer_than_the_cap_keeps_both_endpoints() -> None:
    """Peak VRAM scales with view count, so a long window is thinned -- but
    a drift measurement subtracts the endpoints, so those must survive."""
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(20)]

    selected = _subsample(frames, 5)

    assert len(selected) <= 5
    assert selected[0][0, 0, 0] == 0
    assert selected[-1][0, 0, 0] == 19


def test_a_window_within_the_cap_is_untouched() -> None:
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]

    assert _subsample(frames, 8) is frames


def test_max_views_below_two_is_refused() -> None:
    """One view puts the camera at the origin by definition, which says
    nothing about whether anything moved."""
    with pytest.raises(ValueError, match="at least"):
        Da3WindowPose(max_views=MIN_VIEWS - 1)


def test_identity_rotation_round_trips() -> None:
    q = _matrix_to_quaternion(np.eye(3))

    assert (q.w, q.x, q.y, q.z) == pytest.approx((1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("angle", [math.pi / 2, math.pi, -math.pi / 3])
def test_a_rotation_survives_the_quaternion_conversion(angle: float) -> None:
    """Rotating a vector by the matrix and by the derived quaternion must
    agree. 180 degrees is included on purpose: it is the case the naive
    trace formula cannot handle, which is why Shepperd's method is used."""
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    vector = (0.3, -0.7, 0.2)

    by_matrix = matrix @ np.array(vector)
    by_quaternion = _matrix_to_quaternion(matrix).rotate(vector)

    assert by_quaternion == pytest.approx(tuple(by_matrix), abs=1e-9)


def test_extrinsics_become_a_camera_to_world_capture_pose() -> None:
    """DA3 returns world-to-camera `[R|t]`; a capture pose is the inverse.
    Getting this backwards yields poses that look reasonable and place every
    object in the wrong half of the room."""
    angle = math.pi / 4
    c, s = math.cos(angle), math.sin(angle)
    rotation_w2c = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    camera_position = np.array([1.5, -2.0, 3.0])
    translation_w2c = -rotation_w2c @ camera_position

    extrinsics = np.zeros((3, 4))
    extrinsics[:3, :3] = rotation_w2c
    extrinsics[:3, 3] = translation_w2c
    intrinsics = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])

    geometry = _frame_geometry(np.ones((8, 8)), intrinsics, extrinsics)

    position = geometry.capture_pose.position
    assert (position.x, position.y, position.z) == pytest.approx(tuple(camera_position))


def test_z_depth_becomes_a_range_along_the_view_ray() -> None:
    """`compute_world_point` scales a *unit* ray, so a z-depth has to be
    lengthened by each pixel's ray length. At the principal point the two
    are equal; off-axis the range must be strictly larger, and treating them
    as equal is wrong in a way that grows toward the corners."""
    intrinsics = np.array([[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]])
    geometry = _frame_geometry(
        np.ones((8, 8)), intrinsics, np.hstack([np.eye(3), np.zeros((3, 1))])
    )

    centre = geometry.ranges[3, 3]  # ~the principal point
    corner = geometry.ranges[0, 0]

    assert centre == pytest.approx(1.0, abs=0.02)
    assert corner > centre


def test_range_in_box_takes_a_median_over_the_box() -> None:
    """A centroid landing on an object's edge samples the surface behind it;
    one such frame is enough to fake a jump in world position."""
    ranges = np.full((10, 10), 2.0)
    ranges[0, 0] = 99.0  # an edge pixel reading straight through to the wall
    geometry = _frame_geometry(
        ranges / _ray_lengths(), _intrinsics(), np.hstack([np.eye(3), np.zeros((3, 1))])
    )

    assert geometry.range_in_box(0.0, 0.0, 1.0, 1.0) == pytest.approx(2.0, rel=0.05)


def test_range_in_box_returns_none_where_nothing_is_valid() -> None:
    geometry = _frame_geometry(
        np.full((8, 8), np.nan), _intrinsics(), np.hstack([np.eye(3), np.zeros((3, 1))])
    )

    assert geometry.range_in_box(0.2, 0.2, 0.8, 0.8) is None


def _intrinsics() -> np.ndarray:
    return np.array([[50.0, 0.0, 5.0], [0.0, 50.0, 5.0], [0.0, 0.0, 1.0]])


def _ray_lengths() -> np.ndarray:
    from vision_worker.pose.da3 import _ray_length_map

    return _ray_length_map(_intrinsics(), width=10, height=10)


# --- Scale alignment against a metric reference -----------------------------


def test_scale_is_applied_to_depth_and_pose_together() -> None:
    """Scaling only one of them leaves a self-consistent-looking geometry in
    which every object sits at the wrong distance from a camera that moved
    the wrong amount -- wrong in a way that still looks plausible."""
    from vision_worker.pose.da3 import _build_geometry

    depth = np.full((1, 8, 8), 2.0)
    intrinsics = np.array([_intrinsics()])
    extrinsics = np.array([np.hstack([np.eye(3), np.array([[0.0], [0.0], [-4.0]])])])

    native = _build_geometry(depth, intrinsics, extrinsics, scale=None)
    scaled = _build_geometry(depth, intrinsics, extrinsics, scale=0.5)
    assert native is not None and scaled is not None

    assert scaled.scene_scale == pytest.approx(native.scene_scale * 0.5)
    native_z = native.frames[0].capture_pose.position.z
    scaled_z = scaled.frames[0].capture_pose.position.z
    assert scaled_z == pytest.approx(native_z * 0.5)


def test_only_an_anchored_window_claims_to_be_metric() -> None:
    from vision_worker.pose.da3 import _build_geometry

    depth = np.full((1, 8, 8), 2.0)
    intrinsics = np.array([_intrinsics()])
    extrinsics = np.array([np.hstack([np.eye(3), np.zeros((3, 1))])])

    assert _build_geometry(depth, intrinsics, extrinsics, scale=None).is_metric is False  # type: ignore[union-attr]
    assert _build_geometry(depth, intrinsics, extrinsics, scale=0.7).is_metric is True  # type: ignore[union-attr]


def test_the_scale_factor_is_a_median_ratio_robust_to_bad_pixels() -> None:
    """The two models disagree hardest exactly where one of them is wrong --
    glass, sky, an object's silhouette. A mean would let those set the scale
    for the whole window."""
    from vision_worker.pose.da3 import _median_ratio

    native = np.full((40, 40), 2.0)
    metric = np.full((40, 40), 1.0)  # a true factor of 0.5
    metric[0, :10] = 400.0  # a patch where the metric model is badly wrong

    assert _median_ratio(metric, native) == pytest.approx(0.5)


def test_a_scale_fit_with_too_few_valid_pixels_is_refused() -> None:
    """Better to keep honest arbitrary units than to invent metres from a
    handful of pixels."""
    from vision_worker.pose.da3 import _median_ratio

    native = np.full((40, 40), 2.0)
    metric = np.full((40, 40), np.nan)
    metric[0, 0] = 1.0

    assert _median_ratio(metric, native) is None


def test_the_metric_map_is_resampled_onto_the_reconstruction_grid() -> None:
    """The two models work at different resolutions; a ratio needs them
    pixel-aligned or it compares unrelated parts of the scene."""
    from vision_worker.pose.da3 import _median_ratio

    native = np.full((40, 40), 2.0)
    metric = np.full((720, 1280), 1.0)  # a much larger grid

    assert _median_ratio(metric, native) == pytest.approx(0.5)
