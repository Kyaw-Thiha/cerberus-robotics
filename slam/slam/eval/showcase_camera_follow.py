"""Pure chase-cam pose math for the showcase camera:
position tracks the robot, orientation stays level (yaw-only, no roll/pitch)
rather than rigidly inheriting the robot's full orientation. Kept separate
from any Isaac Lab per-step wiring (needs a pod, same constraint as
ground_truth/pose_export.py) so the math is unit-testable without Isaac Sim.
"""

from __future__ import annotations

import math


def compute_chase_cam_pose(
    robot_position: tuple[float, float, float],
    robot_yaw: float,
    offset_behind: float = 1.5,
    offset_up: float = 0.8,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Returns (position, orientation_xyzw). Camera trails offset_behind
    meters behind the robot along its current yaw heading and offset_up
    meters above it -- only yaw is inherited, never roll/pitch."""
    x, y, z = robot_position
    camera_position = (
        x - offset_behind * math.cos(robot_yaw),
        y - offset_behind * math.sin(robot_yaw),
        z + offset_up,
    )
    half_yaw = robot_yaw / 2.0
    orientation_xyzw = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
    return camera_position, orientation_xyzw
