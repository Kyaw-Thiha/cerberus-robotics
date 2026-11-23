"""Mission orchestrator: navigate -> pick -> navigate -> place, with retries and a hard time budget.

The chain is written against two interfaces so the same code runs in the kinematic sim (today) and on the physics
robot (when the scene run reaches it):

* ``navigate(start_xy, goal_xy, yaw)`` -> a dict with ``success``/``time_s``/``path_length_m`` (the kinematic sim
  drives ``planning.kinematic_sim.run_episode``);
* a ``SkillWorld`` (see ``planning.skills``) for the arm.

Success is defined on the world, not on the plan: the cube rests on the place surface within 5 cm of the target XY,
the robot is upright, and the whole mission finishes inside ``time_budget_s``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from planning.props import MissionProps
from planning.skills import (
    PLACE_TOLERANCE_M,
    PickSkill,
    PlaceSkill,
    SkillResult,
    SkillWorld,
)


class Navigator(Protocol):
    def __call__(self, start_xy: Sequence[float], goal_xy: Sequence[float], yaw: float) -> dict[str, Any]: ...


@dataclass
class MissionConfig:
    time_budget_s: float = 240.0
    max_retries_per_skill: int = 1
    upright_required: bool = True
    place_tolerance_m: float = PLACE_TOLERANCE_M


@dataclass
class MissionResult:
    success: bool
    reason: str
    time_s: float
    steps: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"success": self.success, "reason": self.reason, "time_s": round(self.time_s, 2),
                "retries": self.retries, "steps": self.steps}


class Mission:
    def __init__(self, world: SkillWorld, props: MissionProps, navigate: Navigator,
                 config: MissionConfig | None = None, pick: PickSkill | None = None,
                 place: PlaceSkill | None = None):
        self.world = world
        self.props = props
        self.navigate = navigate
        self.config = config or MissionConfig()
        self.pick = pick or PickSkill(timeout_s=12.0)
        self.place = place or PlaceSkill(timeout_s=12.0, tolerance_m=self.config.place_tolerance_m)
        self.steps: list[dict[str, Any]] = []
        self.retries = 0

    # -- helpers
    def _record(self, name: str, **payload: Any) -> None:
        self.steps.append({"step": name, "t": round(self.world.elapsed_s(), 2), **payload})

    def _retry(self, fn: Callable[[], SkillResult], name: str) -> SkillResult:
        result = fn()
        self._record(name, ok=result.ok, phase=result.phase, detail=result.detail)
        attempts = 0
        while not result.ok and attempts < self.config.max_retries_per_skill:
            attempts += 1
            self.retries += 1
            self._record(f"{name}_retry", attempt=attempts, why=result.phase)
            result = fn()
            self._record(name, ok=result.ok, phase=result.phase, detail=result.detail)
        return result

    def _navigate(self, goal_xy: Sequence[float], yaw: float, name: str) -> bool:
        start = self.world.pose_xy() if hasattr(self.world, "pose_xy") else self.props.approach_xy
        if math.dist(start, goal_xy) < 0.05:
            self._record(name, success=True, time_s=0.0, path_length_m=0.0, outcome="already_there")
            if hasattr(self.world, "set_base_xy"):
                self.world.set_base_xy(goal_xy, yaw)
            return True
        result = self.navigate(start, goal_xy, yaw)
        self._record(name, **{k: result.get(k) for k in ("success", "time_s", "path_length_m", "outcome")})
        ok = bool(result.get("success"))
        if ok and hasattr(self.world, "add_time"):
            self.world.add_time(float(result.get("time_s") or 0.0))  # navigation time counts against the budget
        if ok and hasattr(self.world, "set_base_xy"):
            self.world.set_base_xy(goal_xy, yaw)  # the arm base travels with the robot
        return ok

    # -- the chain
    def run(self) -> MissionResult:
        start_t = self.world.elapsed_s()
        if not self._navigate(self.props.approach_xy, self.props.approach_yaw, "navigate_to_table"):
            return MissionResult(False, "navigation_to_table_failed", self.world.elapsed_s() - start_t,
                                 self.steps, self.retries)

        pick = self._retry(lambda: self.pick.run(self.world, self.props.cube_xy, self.props.pick_table.top_z),
                           "pick")
        if not pick.ok:
            return MissionResult(False, f"pick_{pick.phase}", self.world.elapsed_s() - start_t,
                                 self.steps, self.retries)

        place_goal = self.props.place_approach_xy
        if not self._navigate(place_goal, self.props.place_approach_yaw, "navigate_to_place"):
            return MissionResult(False, "navigation_to_place_failed", self.world.elapsed_s() - start_t,
                                 self.steps, self.retries)

        target_xy = self.props.place_surface.centre_xy
        place = self._retry(lambda: self.place.run(self.world, target_xy, self.props.place_surface.top_z,
                                                   self.props.place_surface.name), "place")
        if not place.ok:
            return MissionResult(False, f"place_{place.phase}", self.world.elapsed_s() - start_t,
                                 self.steps, self.retries)

        if self.config.upright_required and not self.world.upright():
            return MissionResult(False, "robot_fell", self.world.elapsed_s() - start_t, self.steps, self.retries)

        elapsed = self.world.elapsed_s() - start_t
        if elapsed > self.config.time_budget_s:
            return MissionResult(False, "time_budget_exceeded", elapsed, self.steps, self.retries)
        return MissionResult(True, "cube_placed", elapsed, self.steps, self.retries)
