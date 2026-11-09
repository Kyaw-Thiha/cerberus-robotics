"""Quaternion-order helpers for the Isaac Lab 2.x (WXYZ) / 3.x (XYZW) split.

Cerberus-internal convention is w-first (``wxyz``). Isaac Lab >= 3.0 returns
and expects ``xyzw``. ``from_isaac`` / ``to_isaac`` convert between the two;
all functions accept numpy arrays, torch tensors (last dim 4) and plain
tuples/lists (returned as tuples). Pure python: numpy/torch are never
imported here, indexing is duck-typed.

Copy semantics: when the source and target order differ the result is a new
array (fancy indexing); when they are equal the input object is returned
unchanged (no copy). Treat results as read-only.
"""

from __future__ import annotations

import os
from importlib import metadata

ENV_OVERRIDE = "CERBERUS_QUAT_CONVENTION"
_VALID = ("wxyz", "xyzw")
_TO_XYZW = [1, 2, 3, 0]  # wxyz -> xyzw
_TO_WXYZ = [3, 0, 1, 2]  # xyzw -> wxyz


def _check(convention: str) -> str:
    c = str(convention).lower()
    if c not in _VALID:
        raise ValueError(
            f"quaternion convention must be one of {_VALID}, got {convention!r}"
        )
    return c


def isaac_quat_convention() -> str:
    """Native quaternion order of the installed Isaac Lab: 'wxyz' (<3.0 or absent) or 'xyzw'."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return _check(override)
    try:
        version = metadata.version("isaaclab")
    except metadata.PackageNotFoundError:
        return "wxyz"
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return "wxyz"
    return "xyzw" if major >= 3 else "wxyz"


def _reorder(q, idx):
    if isinstance(q, (tuple, list)):
        if len(q) != 4:
            raise ValueError(f"quaternion must have 4 components, got {len(q)}")
        return tuple(q[i] for i in idx)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion last dim must be 4, got shape {tuple(q.shape)}")
    return q[..., idx]


def _convert(q, src: str, dst: str):
    if src == dst:
        return q
    return _reorder(q, _TO_XYZW if dst == "xyzw" else _TO_WXYZ)


def to_wxyz(q, convention: str | None = None):
    """Convert ``q`` given in ``convention`` (default: Isaac's native) to wxyz."""
    return _convert(
        q, _check(convention) if convention else isaac_quat_convention(), "wxyz"
    )


def to_xyzw(q, convention: str | None = None):
    """Convert ``q`` given in ``convention`` (default: Isaac's native) to xyzw."""
    return _convert(
        q, _check(convention) if convention else isaac_quat_convention(), "xyzw"
    )


def from_isaac(q):
    """Native Isaac order -> Cerberus internal (w-first)."""
    return to_wxyz(q, isaac_quat_convention())


def to_isaac(q):
    """Cerberus internal (w-first) -> native Isaac order."""
    return _convert(q, "wxyz", isaac_quat_convention())


def projected_gravity_from_quat(q_xyzw):
    """Gravity direction (unit vector, world) projected into the body frame, from an ``xyzw`` quat.

    Returns ``(gx, gy, gz)`` in the body frame; on level ground this is ``(0, 0, -1)``. One
    quaternion per call (4 components, tuple/list/1-D array or tensor), because that is what the ROS
    inference node feeds per control step.

    The order is explicit (``xyzw``) on purpose: Isaac Lab >= 3.0, ``geometry_msgs/Quaternion`` and
    ``sensor_msgs/Imu.orientation`` all use scalar-last, so no reordering happens between ROS and the
    policy. The module-level ``from_isaac``/``to_isaac`` helpers exist for the pre-3.0 ``wxyz`` stack.
    """
    if len(q_xyzw) != 4:
        raise ValueError(f"quaternion must have 4 components, got {len(q_xyzw)}")
    x, y, z, w = (float(v) for v in q_xyzw)
    # R^T @ [0, 0, -1] with R the body->world rotation: minus the third row of R.
    return (
        2.0 * (y * w - x * z),
        -2.0 * (y * z + x * w),
        2.0 * (x * x + y * y) - 1.0,
    )
