"""Back-projection math -- pure, no GPU, no glasses. Every case here is
something `Pipeline` will eventually feed real ARDK pose through (task
#46); until then this is what proves the math itself is correct.
"""

from __future__ import annotations

import math

import pytest
from visual_memory_vision_contract.protocol import Point2D, Point3D

from vision_worker.domain.geometry import (
    RAYNEO_X3_PRO_INTRINSICS,
    CameraIntrinsics,
    CapturePose,
    Quaternion,
    compute_ray_direction,
    compute_world_point,
)

CENTER = Point2D(x=0.5, y=0.5)
ORIGIN_POSE = CapturePose(position=Point3D(x=0.0, y=0.0, z=0.0), rotation=Quaternion())


def test_a_centered_detection_projects_straight_ahead() -> None:
    direction = compute_ray_direction(
        CENTER,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=ORIGIN_POSE,
    )
    assert direction.x == pytest.approx(0.0, abs=1e-9)
    assert direction.y == pytest.approx(0.0, abs=1e-9)
    assert direction.z == pytest.approx(1.0, abs=1e-9)


def test_world_point_at_the_image_center_is_depth_along_forward() -> None:
    point = compute_world_point(
        CENTER, 2.0, image_width=640, image_height=480, capture_pose=ORIGIN_POSE
    )
    assert point.x == pytest.approx(0.0, abs=1e-9)
    assert point.y == pytest.approx(0.0, abs=1e-9)
    assert point.z == pytest.approx(2.0, abs=1e-9)


def test_world_point_translates_with_the_capture_pose_origin() -> None:
    pose = CapturePose(position=Point3D(x=1.0, y=2.0, z=3.0), rotation=Quaternion())
    point = compute_world_point(CENTER, 2.0, image_width=640, image_height=480, capture_pose=pose)
    assert point.x == pytest.approx(1.0, abs=1e-9)
    assert point.y == pytest.approx(2.0, abs=1e-9)
    assert point.z == pytest.approx(5.0, abs=1e-9)


def test_a_90_degree_yaw_rotates_the_ray_into_x() -> None:
    """q = (cos45, 0, sin45, 0) is a +90-degree rotation about Y; the
    standard quaternion-to-matrix formula maps forward (0, 0, 1) to (1, 0,
    0) under this convention -- checked independently of `Quaternion.rotate`
    against that formula, not just self-consistency."""
    half = math.pi / 4.0
    yaw90 = Quaternion(w=math.cos(half), x=0.0, y=math.sin(half), z=0.0)
    pose = CapturePose(position=Point3D(x=0.0, y=0.0, z=0.0), rotation=yaw90)

    direction = compute_ray_direction(
        CENTER,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=pose,
    )

    assert direction.x == pytest.approx(1.0, abs=1e-9)
    assert direction.y == pytest.approx(0.0, abs=1e-9)
    assert direction.z == pytest.approx(0.0, abs=1e-9)


def test_an_off_center_detection_leans_away_from_the_optical_axis() -> None:
    right_of_center = Point2D(x=0.75, y=0.5)
    direction = compute_ray_direction(
        right_of_center,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=ORIGIN_POSE,
    )
    assert direction.x > 0.0
    assert direction.z > 0.0


def test_a_detection_above_center_leans_upward() -> None:
    """Image y is top-down; a centroid above the vertical center (smaller
    y) must produce a positive (upward) camera-space y component."""
    above_center = Point2D(x=0.5, y=0.25)
    direction = compute_ray_direction(
        above_center,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=ORIGIN_POSE,
    )
    assert direction.y > 0.0


def test_the_mount_offset_is_applied_before_the_capture_pose() -> None:
    """A 90-degree mount offset about Y, with an identity capture pose,
    must produce the same direction as an identity mount offset with that
    same 90-degree rotation as the capture pose -- both apply one 90-degree
    yaw to the same ray, just at a different stage of the pipeline."""
    half = math.pi / 4.0
    yaw90 = Quaternion(w=math.cos(half), x=0.0, y=math.sin(half), z=0.0)

    via_mount = compute_ray_direction(
        CENTER,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=ORIGIN_POSE,
        mount_offset=yaw90,
    )
    via_pose = compute_ray_direction(
        CENTER,
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=CapturePose(position=Point3D(x=0.0, y=0.0, z=0.0), rotation=yaw90),
    )
    assert via_mount.x == pytest.approx(via_pose.x, abs=1e-9)
    assert via_mount.y == pytest.approx(via_pose.y, abs=1e-9)
    assert via_mount.z == pytest.approx(via_pose.z, abs=1e-9)


def test_zero_or_negative_depth_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_world_point(
            CENTER, 0.0, image_width=640, image_height=480, capture_pose=ORIGIN_POSE
        )
    with pytest.raises(ValueError):
        compute_world_point(
            CENTER, -1.0, image_width=640, image_height=480, capture_pose=ORIGIN_POSE
        )


def test_non_finite_depth_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_world_point(
            CENTER, float("nan"), image_width=640, image_height=480, capture_pose=ORIGIN_POSE
        )
    with pytest.raises(ValueError):
        compute_world_point(
            CENTER, float("inf"), image_width=640, image_height=480, capture_pose=ORIGIN_POSE
        )


def test_quaternion_compose_matches_sequential_rotation() -> None:
    """`a.compose(b).rotate(v) == a.rotate(b.rotate(v))` -- the defining
    property of quaternion composition, checked on a non-trivial pair."""
    half = math.pi / 4.0
    yaw90 = Quaternion(w=math.cos(half), x=0.0, y=math.sin(half), z=0.0)
    pitch90 = Quaternion(w=math.cos(half), x=math.sin(half), y=0.0, z=0.0)
    v = (0.3, 0.4, 0.8660254)  # arbitrary, not axis-aligned

    composed = yaw90.compose(pitch90).rotate(v)
    sequential = yaw90.rotate(pitch90.rotate(v))

    assert composed[0] == pytest.approx(sequential[0], abs=1e-9)
    assert composed[1] == pytest.approx(sequential[1], abs=1e-9)
    assert composed[2] == pytest.approx(sequential[2], abs=1e-9)


def test_camera_intrinsics_is_reused_across_calls() -> None:
    custom = CameraIntrinsics(focal_px=100.0)
    near = compute_ray_direction(
        Point2D(x=0.6, y=0.5),
        image_width=640,
        image_height=480,
        intrinsics=custom,
        capture_pose=ORIGIN_POSE,
    )
    far = compute_ray_direction(
        Point2D(x=0.6, y=0.5),
        image_width=640,
        image_height=480,
        intrinsics=RAYNEO_X3_PRO_INTRINSICS,
        capture_pose=ORIGIN_POSE,
    )
    # A shorter focal length (wider apparent FOV) bends the same off-center
    # pixel further from the optical axis.
    assert near.x > far.x
