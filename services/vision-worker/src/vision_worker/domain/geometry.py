"""Back-projection: a detection's screen position, ranged by metric depth,
placed in the world through the camera's pose at capture time.

Ported from the prior-art Unity client's `DetectionAnchor.ComputeWorldPoint`
and `ComputeRayDirection` (`litert-test/Assets/Scripts/DetectionAnchor.cs`),
stripped to the primary path only: this service has no plane-raycast
fallback (no on-device SLAM to raycast against) and no bbox-width depth
heuristic (a `Detection` with no `depth_m` simply gets no `world_point` --
the image-space stability path already handles that case, per
`domain/stability.py`).

Pure stdlib math, like `domain/stability.py` -- no numpy, no torch; asserted
by `tests/test_domain_isolation.py`. `CapturePose` is a plain dataclass, not
a `vision-contract` type, because no wire message carries one yet: it exists
to receive whatever `pose/device.py` (task #46, gated on the RayNeo X3 Pro
uplink) produces once ARDK 6DoF pose reaches this service over the relay's
data channel. Until then, nothing in this service can construct a
`CapturePose` from a live stream, so `Pipeline` never calls
`compute_world_point` and `TrackSample.world_point` stays `None` -- this
module exists ready-tested for the day that changes, not wired to a stub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from visual_memory_vision_contract.protocol import Point2D, Point3D


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Unit quaternion, Unity's `(w, x, y, z)` convention -- ARDK pose (task
    #46) will arrive in this form, so `CapturePose.rotation` matches it
    without a conversion step."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def rotate(self, v: tuple[float, float, float]) -> tuple[float, float, float]:
        """Rotate a vector by this quaternion.

        The optimized `t = 2*cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)`
        form -- equivalent to `q * v * q^-1` for a unit quaternion, without
        constructing the pure-vector quaternion or computing an inverse.
        """
        vx, vy, vz = v
        tx = 2.0 * (self.y * vz - self.z * vy)
        ty = 2.0 * (self.z * vx - self.x * vz)
        tz = 2.0 * (self.x * vy - self.y * vx)
        return (
            vx + self.w * tx + (self.y * tz - self.z * ty),
            vy + self.w * ty + (self.z * tx - self.x * tz),
            vz + self.w * tz + (self.x * ty - self.y * tx),
        )

    def compose(self, other: Quaternion) -> Quaternion:
        """`self * other` -- rotating by `other` first, then by `self`."""
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )


IDENTITY_ROTATION = Quaternion()


@dataclass(frozen=True, slots=True)
class CapturePose:
    """Where the camera was, in world space, at the moment a frame was
    captured. `position` and `rotation` are both required -- there is no
    meaningful default for "the camera's pose," unlike `mount_offset` below,
    which has an honest identity default."""

    position: Point3D
    rotation: Quaternion


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """A single focal length and a centered principal point.

    The RGB sensor has near-square pixels (`fx≈fy`) and a principal point
    within about a pixel of image center, so the prior project models the
    projection with one focal length rather than a full `fx, fy, cx, cy` --
    see `DetectionAnchor.cs`'s `FocalPx`. `principal_x`/`principal_y` are
    recomputed from the actual frame dimensions at call time (`image_width /
    2`, `image_height / 2`), not stored here, since they depend on the
    frame being projected, not the sensor.
    """

    focal_px: float


#: Measured on the RayNeo X3 Pro's RGB sensor at 640x480 by the prior-art
#: project (`DetectionAnchor.cs`'s `FocalPx`); reused here as the default
#: because this is the same physical camera. A frame at any other resolution
#: still uses this focal length in pixels -- it is orientation- and
#: resolution-agnostic per the prior project's own note, since scaling the
#: frame scales focal length and image dimensions together.
RAYNEO_X3_PRO_INTRINSICS = CameraIntrinsics(focal_px=376.4)


def compute_ray_direction(
    centroid: Point2D,
    *,
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsics,
    capture_pose: CapturePose,
    mount_offset: Quaternion = IDENTITY_ROTATION,
) -> Point3D:
    """The unit view ray through a normalized image point, in world space.

    Camera space is x-right, y-up, z-forward; the image's `y` is top-down,
    so it is flipped when building the camera-space direction.
    `mount_offset` corrects the RGB camera's physical mounting offset from
    the eye/display before the capture pose's own rotation is applied --
    the prior project's `RayRotationOffset`, calibrated on-device
    (`vision_calib.json`) rather than assumed to be zero. Defaults to
    identity: correct for any camera whose optical axis is the pose's own
    forward direction, which is the honest default until this service has
    its own on-device calibration.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    px = centroid.x * image_width
    py = centroid.y * image_height

    x_cam = (px - cx) / intrinsics.focal_px
    y_cam = (cy - py) / intrinsics.focal_px
    direction_camera = _normalize((x_cam, y_cam, 1.0))

    direction_mounted = mount_offset.rotate(direction_camera)
    wx, wy, wz = capture_pose.rotation.rotate(direction_mounted)
    return Point3D(x=wx, y=wy, z=wz)


def compute_world_point(
    centroid: Point2D,
    depth_m: float,
    *,
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsics = RAYNEO_X3_PRO_INTRINSICS,
    capture_pose: CapturePose,
    mount_offset: Quaternion = IDENTITY_ROTATION,
) -> Point3D:
    """A detection's world position: the capture pose's origin, plus its
    view ray scaled by `depth_m` -- the metric range MoGe-2 (or another
    depth adapter) measured along that same ray, not a z-depth.

    `depth_m` must be a positive, finite range; `depth/base.py`'s adapters
    only ever set `Detection.depth_m` to a value that already satisfies
    this, so a caller passing a raw, unvalidated depth should check first.
    """
    if not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError(f"depth_m must be a positive, finite range, got {depth_m!r}")

    direction = compute_ray_direction(
        centroid,
        image_width=image_width,
        image_height=image_height,
        intrinsics=intrinsics,
        capture_pose=capture_pose,
        mount_offset=mount_offset,
    )
    origin = capture_pose.position
    return Point3D(
        x=origin.x + direction.x * depth_m,
        y=origin.y + direction.y * depth_m,
        z=origin.z + direction.z * depth_m,
    )


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return (x / length, y / length, z / length)


__all__ = [
    "RAYNEO_X3_PRO_INTRINSICS",
    "CameraIntrinsics",
    "CapturePose",
    "IDENTITY_ROTATION",
    "Quaternion",
    "compute_ray_direction",
    "compute_world_point",
]
