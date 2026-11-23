"""Kinematic mission: the full navigate -> pick -> navigate -> place chain on the real scene-1 grid, no physics.

It is the *upper bound* for the physics mission (docs language: kinematic simulation, scripted IK skills) and it is
labelled that way wherever its output appears. The navigation legs reuse ``planning.kinematic_sim`` (A* + pure
pursuit on a unicycle), the cube is a state machine, and the arm solves the same waypoints the physics robot would
be given.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planning.kinematic_sim import KinematicConfig, run_episode
from planning.props import block_grid, build_props, free_approach
from planning.sampling import StartGoalPair
from planning.skills import (
    CUBE_SIZE_M,
    GRASP_TOLERANCE_M,
    ArmModel,
    differential_ik,
)
from planning.task_orchestrator import Mission, MissionConfig


@dataclass
class KinematicSkillWorld:
    """A SkillWorld whose arm is the IK model and whose gripper moves a cube state machine."""

    arm_base_xy: tuple[float, float] = (0.0, 0.0)
    arm_base_yaw: float = 0.0
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cube: dict[str, Any] = field(default_factory=dict)
    t_s: float = 0.0
    grip_m: float = 0.044
    holds_cube: bool = False
    stage_timer_s: float = 0.0
    drop_on_lift: bool = False          # failure injection for tests
    surfaces: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {centre_xy, top_z, footprint_m}
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = ArmModel()
        self.q = np.array([0.0, 0.6, -1.0])
        if not self.cube:
            self.cube = {"pos": (0.5, 0.0, 0.37), "attached": False, "on_surface": "pick_table"}

    # -- SkillWorld
    def pose_xy(self) -> tuple[float, float]:
        return (self.pose[0], self.pose[1])

    def upright(self) -> bool:
        return True

    def elapsed_s(self) -> float:
        return self.t_s

    def add_time(self, seconds: float) -> None:
        self.t_s += max(float(seconds), 0.0)

    def set_base_xy(self, xy: Sequence[float], yaw: float | None = None) -> None:
        """The robot arrived: the arm's base (and heading) move with it, so targets stay world targets."""
        self.arm_base_xy = (float(xy[0]), float(xy[1]))
        if yaw is not None:
            self.arm_base_yaw = float(yaw)
            self.pose = (float(xy[0]), float(xy[1]), float(yaw))

    def _to_arm_frame(self, target: np.ndarray) -> np.ndarray:
        delta = np.asarray(target, dtype=float) - np.array([self.arm_base_xy[0], self.arm_base_xy[1], 0.0])
        c, s = math.cos(-self.arm_base_yaw), math.sin(-self.arm_base_yaw)
        return np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]])

    def _tcp_world(self) -> np.ndarray:
        local = self.model.fk(self.q)
        c, s = math.cos(self.arm_base_yaw), math.sin(self.arm_base_yaw)
        return np.array([c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]]) \
            + np.array([self.arm_base_xy[0], self.arm_base_xy[1], 0.0])

    def move_tcp(self, target: np.ndarray, timeout_s: float) -> dict[str, Any]:
        """Solve IK to a world-frame ``target`` (the arm base is at the robot's position) and charge motion time."""
        start_q = self.q.copy()
        result = differential_ik(self.model, self._to_arm_frame(target), self.q)
        self.q = result["q"]
        # time: 0.4 s per radian of joint motion, plus a fixed settling time, is a plausible scripted profile
        travel = float(np.abs(self.q - start_q).sum())
        dt = 0.4 * travel + 0.15
        self.t_s += dt
        reached = result["error_m"] <= 0.02
        self.sync_cube()  # a held cube travels with the gripper
        self.log.append({"tcp_target": [round(float(v), 3) for v in target],
                         "error_m": round(float(result["error_m"]), 4), "reached": reached})
        return {"reached": reached, "error_m": float(result["error_m"]), "steps": int(dt / 0.02)}

    def set_gripper(self, opening_m: float) -> None:
        self.grip_m = float(opening_m)
        self.t_s += 0.3

    def cube_state(self) -> dict[str, Any]:
        return dict(self.cube)

    def attach_cube(self) -> bool:
        tcp = self._tcp_world()
        cube = np.array(self.cube["pos"], dtype=float)
        if float(np.linalg.norm(tcp - cube)) > GRASP_TOLERANCE_M:
            return False
        if self.drop_on_lift:
            return False
        self.cube["attached"] = True
        self.cube["on_surface"] = None
        self.holds_cube = True
        self.t_s += 0.2
        return True

    def register_surface(self, name: str, centre_xy: Sequence[float], top_z: float, footprint_m: float) -> None:
        self.surfaces[name] = {"centre_xy": (float(centre_xy[0]), float(centre_xy[1])), "top_z": float(top_z),
                               "footprint_m": float(footprint_m)}

    def release_cube(self) -> None:
        """Open the gripper: the cube falls onto whichever registered surface is under it (or to the floor)."""
        tcp = self._tcp_world()
        self.cube["attached"] = False
        self.holds_cube = False
        landed = None
        for name, surface in self.surfaces.items():
            half = surface["footprint_m"] / 2.0
            if (abs(tcp[0] - surface["centre_xy"][0]) <= half + CUBE_SIZE_M
                    and abs(tcp[1] - surface["centre_xy"][1]) <= half + CUBE_SIZE_M
                    and tcp[2] >= surface["top_z"]):
                landed = name
                self.cube["pos"] = (float(tcp[0]), float(tcp[1]), surface["top_z"] + CUBE_SIZE_M / 2.0)
                break
        if landed is None:
            self.cube["pos"] = (float(tcp[0]), float(tcp[1]), 0.0 + CUBE_SIZE_M / 2.0)
        self.cube["on_surface"] = landed
        self.t_s += 0.2

    # the skills move the cube with the gripper between attach and release
    def sync_cube(self) -> None:
        if not self.cube.get("attached"):
            return
        tcp = self._tcp_world()
        self.cube["pos"] = (float(tcp[0]), float(tcp[1]), float(tcp[2]))


def _navigator(grid, *, start_pose: tuple[float, float, float], speed_note: str = "unicycle upper bound"):
    """Adapter: navigate(start_xy, goal_xy, yaw) -> kinematic_sim episode result."""

    def navigate(start_xy: Sequence[float], goal_xy: Sequence[float], yaw: float) -> dict[str, Any]:
        pair = StartGoalPair(start=(float(start_xy[0]), float(start_xy[1])),
                             goal=(float(goal_xy[0]), float(goal_xy[1])),
                             shortest_path_m=math.dist(start_xy, goal_xy))
        result = run_episode(grid, pair, config=KinematicConfig(), initial_yaw=yaw)
        result["note"] = speed_note
        return result

    return navigate


def run_kinematic_mission(grid, *, seed: int = 0, pick_centre=(0.5, 0.0), place_centre=None,
                          config: MissionConfig | None = None) -> dict[str, Any]:
    """One mission on ``grid``; returns the orchestrator record plus the props and the world log."""
    if place_centre is None:
        # a few metres away, inside the grid, chosen deterministically per seed
        ex = grid.extent()
        place_centre = (min(max(ex[0] + 0.35 * (ex[2] - ex[0]), ex[0] + 1.0), ex[2] - 1.0),
                        min(max(ex[1] + 0.5 * (ex[3] - ex[1]), ex[1] + 1.0), ex[3] - 1.0))
    props = build_props(pick_centre, place_centre)
    blocked = block_grid(grid, props)
    # stand where the arm can actually reach the cube and the base is not inside the table's footprint
    pick_approach = free_approach(blocked, props.pick_table)
    if pick_approach is not None:
        props.approach_xy = (pick_approach[0], pick_approach[1])
        props.approach_yaw = pick_approach[2]
    place_approach = free_approach(blocked, props.place_surface)
    if place_approach is not None:
        props.place_approach_xy = (place_approach[0], place_approach[1])
        props.place_approach_yaw = place_approach[2]

    # the robot starts ��1.2 m back from the table approach, facing it: a real navigation leg, not a no-op
    back = 1.2
    start = (props.approach_xy[0] - back * math.cos(props.approach_yaw),
             props.approach_xy[1] - back * math.sin(props.approach_yaw))
    if not grid.is_free(start[0], start[1], inflated=True):
        start = props.approach_xy
    world = KinematicSkillWorld(arm_base_xy=props.approach_xy, pose=(start[0], start[1], props.approach_yaw),
                                arm_base_yaw=props.approach_yaw, cube={
        "pos": (props.cube_xy[0], props.cube_xy[1], props.pick_table.top_z + CUBE_SIZE_M / 2.0),
        "attached": False, "on_surface": "pick_table"})
    world.register_surface(props.pick_table.name, props.pick_table.centre_xy, props.pick_table.top_z,
                           props.pick_table.footprint_m)
    world.register_surface(props.place_surface.name, props.place_surface.centre_xy, props.place_surface.top_z,
                           props.place_surface.footprint_m)
    mission = Mission(world, props, _navigator(blocked, start_pose=(*props.approach_xy, props.approach_yaw)),
                      config=config or MissionConfig())
    result = mission.run()
    return {"seed": seed, "result": result.as_dict(), "props": props.as_dict(),
            "world_log": world.log, "world_elapsed_s": round(world.t_s, 2)}


def mission_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    ok = sum(1 for r in records if r["result"]["success"])
    times = [r["result"]["time_s"] for r in records]
    retries = sum(r["result"]["retries"] for r in records)
    reasons: dict[str, int] = {}
    for record in records:
        if not record["result"]["success"]:
            reasons[record["result"]["reason"]] = reasons.get(record["result"]["reason"], 0) + 1
    return {"trials": n, "success": ok, "success_rate": (ok / n) if n else 0.0,
            "mean_time_s": round(sum(times) / n, 2) if n else 0.0, "retries": retries,
            "failure_modes": reasons, "kind": "kinematic simulation, scripted IK skills"}
