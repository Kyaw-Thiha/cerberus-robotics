"""Scripted pick-and-place waypoint state machine for the W2a stack task. Pure python (no torch, no isaaclab).

Sequence for "put cube_2 on cube_1": above pick (gripper open) -> descend -> close (pause) -> lift -> above place ->
descend -> open (pause) -> retreat -> done.

Action convention (the one Isaac Lab Mimic's `target_eef_pose_to_action` uses for the stock Franka stack task, and
the one of the env's single `DifferentialInverseKinematicsActionCfg(use_relative_mode=True, scale=IK_ACTION_SCALE)`):

    action[0:3]  position error target - current (metres), NOT divided by the IK scale
    action[3:6]  rotation error as an axis-angle vector (radians), world/base frame
    action[6]    gripper: +1 open, -1 close (`BinaryJointPositionAction`: a negative command closes)

The controller then moves the commanded pose by `IK_ACTION_SCALE * action` per env step. To keep the arm
inside what the IK can track, the position/rotation parts are norm-clipped so that the *commanded* step stays
below `max_command_step_m` / `max_command_rot_rad`, i.e. `|action| <= max_command / ik_scale`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

Vec3 = tuple[float, float, float]

GRIPPER_ACTION_OPEN = 1.0
GRIPPER_ACTION_CLOSE = -1.0
ACTION_DIM = 7


class Phase(str, Enum):
    ABOVE_PICK = "above_pick"
    DESCEND_PICK = "descend_pick"
    CLOSE = "close"
    LIFT = "lift"
    ABOVE_PLACE = "above_place"
    DESCEND_PLACE = "descend_place"
    OPEN = "open"
    RETREAT = "retreat"
    DONE = "done"
    FAILED = "failed"


PHASE_ORDER: tuple[Phase, ...] = (
    Phase.ABOVE_PICK,
    Phase.DESCEND_PICK,
    Phase.CLOSE,
    Phase.LIFT,
    Phase.ABOVE_PLACE,
    Phase.DESCEND_PLACE,
    Phase.OPEN,
    Phase.RETREAT,
)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _scale(v: Vec3, k: float) -> Vec3:
    return (v[0] * k, v[1] * k, v[2] * k)


def _up(p: Vec3, dz: float) -> Vec3:
    return (p[0], p[1], p[2] + dz)


def clip_norm(v: Vec3, limit: float) -> Vec3:
    """Scale `v` down to norm `limit` if longer (direction preserved); shorter vectors pass unchanged."""
    n = _norm(v)
    if n <= limit or n == 0.0:
        return (float(v[0]), float(v[1]), float(v[2]))
    k = limit / n
    return (v[0] * k, v[1] * k, v[2] * k)


@dataclass(frozen=True)
class PickPlaceConfig:
    cube_size: float = 0.04
    hover_height: float = 0.10  # TCP height above the grasp / release pose during approach and transport (m)
    grasp_z_offset: float = (
        0.0  # TCP height above the cube centre at grasp (m); tune on the pod
    )
    place_margin: float = 0.004  # extra height above the cube_1 top at release (m)
    ik_scale: float = 0.5  # the DiffIK action scale (IK_ACTION_SCALE)
    max_command_step_m: float = 0.015  # max commanded TCP translation per env step (m)
    max_command_rot_rad: float = 0.04  # max commanded rotation per env step (rad)
    gain_pos: float = 1.0
    gain_rot: float = 1.0
    hover_pos_tol: float = 0.015
    fine_pos_tol: float = 0.006
    rot_tol: float = 0.08
    close_pause_steps: int = 40  # pod-tested: fingers still ~0.025 m open after 15 steps and slip before lift
    open_pause_steps: int = 12
    max_phase_steps: int = 300

    def __post_init__(self) -> None:
        positive = (
            "cube_size",
            "hover_height",
            "ik_scale",
            "max_command_step_m",
            "max_command_rot_rad",
        )
        for name in positive + ("hover_pos_tol", "fine_pos_tol", "rot_tol"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if (
            self.close_pause_steps < 1
            or self.open_pause_steps < 1
            or self.max_phase_steps < 1
        ):
            raise ValueError("pause/timeout step counts must be >= 1")

    @property
    def max_pos_action(self) -> float:
        """Largest allowed |position action| (m); commanded step = ik_scale * action <= max_command_step_m."""
        return self.max_command_step_m / self.ik_scale

    @property
    def max_rot_action(self) -> float:
        return self.max_command_rot_rad / self.ik_scale


@dataclass(frozen=True)
class ScriptedAction:
    pos: Vec3
    rot: Vec3
    gripper: float
    phase: Phase

    def as_list(self) -> list[float]:
        return [*self.pos, *self.rot, self.gripper]


@dataclass(frozen=True)
class Waypoint:
    phase: Phase
    target: Vec3
    gripper: float
    pos_tol: float
    hold_steps: int = (
        0  # > 0: advance after this many steps in the phase, regardless of position
    )


def relative_ik_action(
    err_pos: Vec3, err_rot: Vec3, cfg: PickPlaceConfig
) -> tuple[Vec3, Vec3]:
    """Clipped relative-IK position/rotation action for the given errors (target minus current)."""
    pos = clip_norm(_scale(err_pos, cfg.gain_pos), cfg.max_pos_action)
    rot = clip_norm(_scale(err_rot, cfg.gain_rot), cfg.max_rot_action)
    return pos, rot


def commanded_delta(action_part: Vec3, ik_scale: float) -> Vec3:
    """What the DiffIK action term adds to the commanded pose for this action part (`scale * action`)."""
    return _scale(action_part, ik_scale)


class ScriptedPickPlace:
    """Waypoint state machine. Call `reset()` once per episode, then `step()` once per env step."""

    def __init__(self, cfg: PickPlaceConfig | None = None) -> None:
        self.cfg = cfg or PickPlaceConfig()
        self._plan: list[Waypoint] = []
        self._index = 0
        self._phase_steps = 0
        self.total_steps = 0
        self.phase: Phase = Phase.DONE
        self.failure_reason: str | None = None

    # -- planning ----------------------------------------------------------------------------------------------
    def reset(self, pick_pos: Vec3, place_pos: Vec3) -> None:
        """Plan an episode: grasp the cube centred at `pick_pos`, release it on top of the cube at `place_pos`.

        Both are cube-centre positions in the frame the TCP position is reported in (the env frame).
        """
        c = self.cfg
        grasp = _up(pick_pos, c.grasp_z_offset)
        release = _up(place_pos, c.cube_size + c.grasp_z_offset + c.place_margin)
        hover, fine = c.hover_pos_tol, c.fine_pos_tol
        self._plan = [
            Waypoint(
                Phase.ABOVE_PICK, _up(grasp, c.hover_height), GRIPPER_ACTION_OPEN, hover
            ),
            Waypoint(Phase.DESCEND_PICK, grasp, GRIPPER_ACTION_OPEN, fine),
            Waypoint(
                Phase.CLOSE,
                grasp,
                GRIPPER_ACTION_CLOSE,
                fine,
                hold_steps=c.close_pause_steps,
            ),
            Waypoint(
                Phase.LIFT, _up(grasp, c.hover_height), GRIPPER_ACTION_CLOSE, hover
            ),
            Waypoint(
                Phase.ABOVE_PLACE,
                _up(release, c.hover_height),
                GRIPPER_ACTION_CLOSE,
                hover,
            ),
            Waypoint(Phase.DESCEND_PLACE, release, GRIPPER_ACTION_CLOSE, fine),
            Waypoint(
                Phase.OPEN,
                release,
                GRIPPER_ACTION_OPEN,
                fine,
                hold_steps=c.open_pause_steps,
            ),
            Waypoint(
                Phase.RETREAT, _up(release, c.hover_height), GRIPPER_ACTION_OPEN, hover
            ),
        ]
        self._index = 0
        self._phase_steps = 0
        self.total_steps = 0
        self.phase = self._plan[0].phase
        self.failure_reason = None

    @property
    def plan(self) -> tuple[Waypoint, ...]:
        return tuple(self._plan)

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def failed(self) -> bool:
        return self.phase is Phase.FAILED

    @property
    def finished(self) -> bool:
        return self.done or self.failed

    @property
    def target(self) -> Vec3 | None:
        return (
            self._plan[self._index].target if self._plan and not self.finished else None
        )

    # -- stepping ----------------------------------------------------------------------------------------------
    def step(self, eef_pos: Vec3, rot_err: Vec3 = (0.0, 0.0, 0.0)) -> ScriptedAction:
        """Next action for the observed TCP position and the axis-angle rotation error (desired minus current)."""
        if not self._plan:
            raise RuntimeError("ScriptedPickPlace.step() called before reset()")
        eef = (float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]))
        rot = (float(rot_err[0]), float(rot_err[1]), float(rot_err[2]))
        if self.finished:
            return ScriptedAction(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GRIPPER_ACTION_OPEN, self.phase
            )

        wp = self._plan[self._index]
        err = _sub(wp.target, eef)
        if wp.hold_steps > 0:
            arrived = self._phase_steps >= wp.hold_steps
        else:
            arrived = _norm(err) <= wp.pos_tol and _norm(rot) <= self.cfg.rot_tol

        if arrived:
            self._index += 1
            self._phase_steps = 0
            if self._index >= len(self._plan):
                self.phase = Phase.DONE
                return ScriptedAction(
                    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GRIPPER_ACTION_OPEN, self.phase
                )
            wp = self._plan[self._index]
            err = _sub(wp.target, eef)
            self.phase = wp.phase
        elif self._phase_steps >= self.cfg.max_phase_steps:
            self.phase = Phase.FAILED
            self.failure_reason = (
                f"timeout in {wp.phase.value} after {self._phase_steps} steps: "
                f"pos_err={_norm(err):.4f} m, rot_err={_norm(rot):.4f} rad"
            )
            return ScriptedAction(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GRIPPER_ACTION_OPEN, self.phase
            )

        self._phase_steps += 1
        self.total_steps += 1
        pos_action, rot_action = relative_ik_action(err, rot, self.cfg)
        return ScriptedAction(pos_action, rot_action, wp.gripper, wp.phase)
