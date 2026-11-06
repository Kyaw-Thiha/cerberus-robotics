"""Ground-truth pose field builders, decoupled from Isaac Lab.

These functions operate on plain tuples (position, quaternion, velocities) and
have no dependency on Isaac Lab's API — the caller is responsible for reading
the actual simulation state and passing it in as plain values.

Isaac Lab attribute names (confirmed 2026-09-20 against the pinned Isaac Lab
version, on a RunPod GPU pod running Isaac-Velocity-Flat-Unitree-Go2-v0): the
robot articulation's `.data.root_pos_w` (world-frame position, shape (N, 3))
and `.data.root_quat_w` (world-frame orientation, shape (N, 4); (w, x, y, z)
order on Isaac Lab 2.x, (x, y, z, w) on 3.x -- see common/cerberus_compat/quat.py)
both exist and are populated with live data every physics step, as
the original design assumed. Newer aliases `root_link_pos_w` /
`root_link_quat_w` also exist and return identical values, but the original
`root_pos_w` / `root_quat_w` names are still valid and are what callers should
use to feed `isaac_quat_to_ros`, `build_odometry_fields`, and
`build_ground_truth_transform` below.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.cerberus_compat.quat import to_xyzw


def isaac_quat_to_ros(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Cerberus-internal (w, x, y, z) components -> ROS (x, y, z, w). Order is explicit by keyword."""
    return tuple(to_xyzw((w, x, y, z), "wxyz"))


def native_isaac_quat_to_ros(isaac_quat, convention: str | None = None) -> tuple[float, float, float, float]:
    """Quaternion in Isaac's NATIVE order (wxyz on Lab 2.x, xyzw on 3.x) -> ROS (x, y, z, w)."""
    return tuple(float(c) for c in to_xyzw(isaac_quat, convention))


@dataclass(frozen=True)
class OdometryFields:
    frame_id: str
    child_frame_id: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


def build_odometry_fields(position, isaac_quat, linear_velocity, angular_velocity) -> OdometryFields:
    """`isaac_quat` is Cerberus-internal (w, x, y, z); convert native Isaac data with `from_isaac` first."""
    w, x, y, z = isaac_quat
    return OdometryFields("world", "base_link", position, isaac_quat_to_ros(w=w, x=x, y=y, z=z), linear_velocity, angular_velocity)


@dataclass(frozen=True)
class TransformFields:
    frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def build_ground_truth_transform(position, isaac_quat) -> TransformFields:
    """`isaac_quat` is Cerberus-internal (w, x, y, z); convert native Isaac data with `from_isaac` first."""
    w, x, y, z = isaac_quat
    return TransformFields("world", "base_link_gt", position, isaac_quat_to_ros(w=w, x=x, y=y, z=z))
