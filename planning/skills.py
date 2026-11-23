"""Scripted manipulation skills: differential-IK end-effector waypoints for a pick and a place.

Scope and honesty: this module is the *kinematic* half of the mission work. It models the ARX5 as a 6-DoF serial
arm with the spec's joint limits and nominal link lengths (a 3-DoF position task on joints 1-3 plus a fixed grasp
orientation), solves the waypoints with damped least squares, and detects the three failures a scripted skill can
actually detect: the cube was not lifted, the cube was dropped, and the skill timed out. It does not model contact
forces, friction or the gripper's compliance; when it is used to drive the physics robot, the motion is labelled
"scripted IK" in every clip and report.

Pure Python (numpy only), so it is testable on the laptop and reusable from the Isaac side.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# Nominal geometry of the X5A, in the arm base frame (metres). These are the kinematic sim's parameters; the
# physics robot uses the real USD and the same waypoints.
L1_M = 0.26          # shoulder to elbow
L2_M = 0.26          # elbow to wrist
TOOL_M = 0.10        # wrist to tool centre point (total reach 0.62 m, the X5A's published reach)
BASE_HEIGHT_M = 0.07  # arm base above the Go2 body origin (spec GO2_ARX5_MOUNT_XYZ)
GRIPPER_OPEN_M = 0.044
CUBE_SIZE_M = 0.04
GRASP_TOLERANCE_M = 0.02      # how close the TCP must be to the cube centre to grasp
PLACE_TOLERANCE_M = 0.05      # mission success: cube centre within 5 cm of the target XY
DEFAULT_TIMEOUT_S = 12.0


def clamp_joint(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ArmModel:
    """A 3-DoF position arm (pan, lift, elbow) with a fixed wrist; limits in radians."""

    limits: tuple[tuple[float, float], ...] = ((-2.6, 2.6), (-1.4, 1.4), (-2.4, 2.4))

    def fk(self, q: Sequence[float]) -> np.ndarray:
        q1, q2, q3 = (float(v) for v in q[:3])
        # pan about z, then a planar 2-link chain in the pan plane, plus the tool
        r = L1_M * math.cos(q2) + L2_M * math.cos(q2 + q3) + TOOL_M * math.cos(q2 + q3)
        z = BASE_HEIGHT_M + L1_M * math.sin(q2) + L2_M * math.sin(q2 + q3) + TOOL_M * math.sin(q2 + q3)
        return np.array([r * math.cos(q1), r * math.sin(q1), z])

    def jacobian(self, q: Sequence[float], eps: float = 1e-5) -> np.ndarray:
        base = self.fk(q)
        J = np.zeros((3, 3))
        for i in range(3):
            dq = list(q[:3])
            dq[i] += eps
            J[:, i] = (self.fk(dq) - base) / eps
        return J

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        return np.array([clamp_joint(v, *self.limits[i]) for i, v in enumerate(q[:3])])


def differential_ik(model: ArmModel, target: np.ndarray, q0: Sequence[float] = (0.0, 0.6, -1.0), *,
                    max_iters: int = 200, damping: float = 0.05, step_limit: float = 0.12,
                    tol: float = 0.005) -> dict[str, Any]:
    """Damped least squares to a position target; returns ``{q, error_m, converged, iterations}``."""
    q = model.clamp(q0)
    target = np.asarray(target, dtype=float)
    error = model.fk(q) - target
    for iteration in range(1, max_iters + 1):
        if float(np.linalg.norm(error)) <= tol:
            return {"q": q, "error_m": float(np.linalg.norm(error)), "converged": True, "iterations": iteration - 1}
        J = model.jacobian(q)
        # dq = J^T (J J^T + lambda^2 I)^-1 (-e)
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), -error)
        norm = float(np.linalg.norm(dq))
        if norm > step_limit:
            dq *= step_limit / norm
        q = model.clamp(q + dq)
        error = model.fk(q) - target
    return {"q": q, "error_m": float(np.linalg.norm(error)),
            "converged": float(np.linalg.norm(error)) <= tol, "iterations": max_iters}


# ---- waypoints -------------------------------------------------------------------------------------------------

def pick_waypoints(cube_xy: Sequence[float], table_top_z: float, *, approach_m: float = 0.06) -> dict[str, Any]:
    """pre-grasp (above the cube) -> descend (cube centre) -> close -> lift."""
    x, y = float(cube_xy[0]), float(cube_xy[1])
    grasp_z = table_top_z + CUBE_SIZE_M / 2.0
    return {
        "cube_xy": (x, y),
        "pre_grasp": np.array([x, y, grasp_z + approach_m]),
        "descend": np.array([x, y, grasp_z]),
        "close": GRIPPER_OPEN_M,
        "lift": np.array([x, y, grasp_z + approach_m]),
        "grasp_z": grasp_z,
    }


def place_waypoints(target_xy: Sequence[float], surface_z: float, *, approach_m: float = 0.06) -> dict[str, Any]:
    """approach (above the target) -> lower -> open -> retreat."""
    x, y = float(target_xy[0]), float(target_xy[1])
    release_z = surface_z + CUBE_SIZE_M / 2.0
    return {
        "target_xy": (x, y),
        "approach": np.array([x, y, release_z + approach_m]),
        "lower": np.array([x, y, release_z]),
        "open": GRIPPER_OPEN_M,
        "retract": np.array([x, y, release_z + approach_m]),
        "release_z": release_z,
    }


# ---- skill world interface -------------------------------------------------------------------------------------

class SkillWorld(Protocol):
    """What a skill may touch. The kinematic sim and the Isaac side both implement it; tests use a mock."""

    def move_tcp(self, target: np.ndarray, timeout_s: float) -> dict[str, Any]: ...
    def set_gripper(self, opening_m: float) -> None: ...
    def cube_state(self) -> dict[str, Any]: ...      # {"pos": (x,y,z), "attached": bool, "on_surface": str|None}
    def attach_cube(self) -> bool: ...
    def release_cube(self) -> None: ...
    def elapsed_s(self) -> float: ...


@dataclass
class SkillResult:
    ok: bool
    phase: str
    detail: str
    retries: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "phase": self.phase, "detail": self.detail, "retries": self.retries,
                "log": self.log}


class PickSkill:
    """Scripted pick: pre-grasp, descend, close, lift -- with the three detectable failures."""

    def __init__(self, model: ArmModel | None = None, *, timeout_s: float = DEFAULT_TIMEOUT_S,
                 lift_m: float = 0.10):
        self.model = model or ArmModel()
        self.timeout_s = timeout_s
        self.lift_m = lift_m

    def run(self, world: SkillWorld, cube_xy: Sequence[float], table_top_z: float) -> SkillResult:
        started = world.elapsed_s()
        plan = pick_waypoints(cube_xy, table_top_z)
        log: list[dict[str, Any]] = []

        world.set_gripper(plan["close"])
        for name in ("pre_grasp", "descend"):
            result = world.move_tcp(plan[name], self.timeout_s)
            log.append({"phase": name, **{k: result.get(k) for k in ("reached", "error_m", "steps")}})
            if world.elapsed_s() - started > self.timeout_s:
                return SkillResult(False, "timeout", f"exceeded {self.timeout_s:.1f}s before {name}", log=log)
            if not result.get("reached", False):
                return SkillResult(False, "unreachable", f"{name} not reached: {result.get('error_m')}", log=log)

        attached = world.attach_cube()
        if not attached:
            return SkillResult(False, "grasp_failed", "gripper closed with nothing within tolerance", log=log)

        lifted = np.array(plan["lift"]) + np.array([0.0, 0.0, self.lift_m])
        result = world.move_tcp(lifted, self.timeout_s)
        log.append({"phase": "lift", "reached": result.get("reached"), "error_m": result.get("error_m")})
        state = world.cube_state()
        if not state.get("attached") or state["pos"][2] < plan["grasp_z"] + 0.5 * self.lift_m:
            return SkillResult(False, "cube_not_lifted", "cube did not leave the table with the gripper", log=log)
        return SkillResult(True, "picked", "cube lifted and attached", log=log)


class PlaceSkill:
    """Scripted place: approach, lower, open, retreat; success measured on the cube, not on the arm."""

    def __init__(self, model: ArmModel | None = None, *, timeout_s: float = DEFAULT_TIMEOUT_S,
                 tolerance_m: float = PLACE_TOLERANCE_M):
        self.model = model or ArmModel()
        self.timeout_s = timeout_s
        self.tolerance_m = tolerance_m

    def run(self, world: SkillWorld, target_xy: Sequence[float], surface_z: float,
            surface_id: str = "place_surface") -> SkillResult:
        started = world.elapsed_s()
        plan = place_waypoints(target_xy, surface_z)
        log: list[dict[str, Any]] = []
        for name in ("approach", "lower"):
            result = world.move_tcp(plan[name], self.timeout_s)
            log.append({"phase": name, "reached": result.get("reached"), "error_m": result.get("error_m")})
            if world.elapsed_s() - started > self.timeout_s:
                return SkillResult(False, "timeout", f"exceeded {self.timeout_s:.1f}s before {name}", log=log)
            if not result.get("reached", False):
                return SkillResult(False, "unreachable", f"{name} not reached: {result.get('error_m')}", log=log)

        world.release_cube()
        state = world.cube_state()
        if state.get("attached"):
            return SkillResult(False, "release_failed", "cube still attached after opening", log=log)
        if state.get("on_surface") != surface_id:
            return SkillResult(False, "cube_dropped", f"cube ended on {state.get('on_surface')!r}", log=log)

        dx = state["pos"][0] - float(target_xy[0])
        dy = state["pos"][1] - float(target_xy[1])
        miss = math.hypot(dx, dy)
        log.append({"phase": "place_error", "miss_m": round(miss, 4)})
        if miss > self.tolerance_m:
            return SkillResult(False, "off_target", f"cube landed {miss:.3f} m from the target", log=log)

        result = world.move_tcp(plan["retract"], self.timeout_s)
        log.append({"phase": "retract", "reached": result.get("reached")})
        return SkillResult(True, "placed", f"cube within {miss:.3f} m of the target", log=log)
