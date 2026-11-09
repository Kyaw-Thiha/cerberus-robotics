"""Planar body-twist kinematics shared by the scripted-path generator and the closed-loop kinematic test.

Pure Python (stdlib only). A ``LocoCommand`` (vx, vy, wz) is a twist in the BASE frame; this integrates it into a
world-frame (x, y, yaw) pose. It is an idealised command integrator, not a model of the legged robot: it assumes
perfect tracking, which is exactly why the kinematic navigation result is an *upper bound* for the physics runs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

Pose = tuple[float, float, float]


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def step_pose(pose: Pose, command: Mapping[str, float], dt: float) -> Pose:
    """Integrate one base-frame twist ``command`` (vx, vy, wz) over ``dt`` seconds (exact for constant twist).

    Uses the closed-form constant-twist arc, so a turning command does not drift with the step size.
    """
    x, y, yaw = pose
    vx, vy, wz = float(command["vx"]), float(command["vy"]), float(command["wz"])
    dyaw = wz * dt
    if abs(dyaw) < 1e-9:
        dx_b, dy_b = vx * dt, vy * dt
    else:
        s, c = math.sin(dyaw), math.cos(dyaw)
        # integral of R(wz t) [vx, vy] dt over [0, dt]
        dx_b = (vx * s - vy * (1.0 - c)) / wz
        dy_b = (vx * (1.0 - c) + vy * s) / wz
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (x + cy * dx_b - sy * dy_b, y + sy * dx_b + cy * dy_b, wrap_angle(yaw + dyaw))


def integrate(pose: Pose, commands, dt: float) -> Pose:
    """Pose after applying every command in ``commands`` for ``dt`` each."""
    for command in commands:
        pose = step_pose(pose, command, dt)
    return pose
