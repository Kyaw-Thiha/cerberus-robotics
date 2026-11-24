"""Chained exploration tour: pick reachable goals on a grid, plan A* between them, follow with pure pursuit.

This is the multi-goal counterpart of :func:`planning.kinematic_sim.run_episode` and the shared driver behind
`scripts/explore_tour.py` (kinematic upper bound and, with the Isaac backend, the physics tour). One backend, one
continuous run: the robot is never teleported between goals, so the travelled trail and the net displacement mean
what they say.

The record is the "record once, render many" source for the showcase clips: per control step it stores time, root
pose (XYZW), commands, the active goal, the planned path and contact/fall flags, plus the occupancy grid. Joint
state, the simulated LiDAR depth image and any other backend-specific fields enter through ``extra``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from common.cerberus_policies.command_limiter import CommandRateLimiter
from planning.global_planner.core.astar import astar, traversable_mask
from planning.local_planner.core.pure_pursuit import PurePursuit, PurePursuitConfig
from planning.occupancy import OccupancyGrid


class TourBackend(Protocol):
    num_envs: int

    def step(self, commands: Sequence[dict[str, float]]) -> None: ...
    def poses(self) -> list[tuple[float, float, float]]: ...
    def falls(self) -> list[bool]: ...


@dataclass
class TourConfig:
    dt: float = 0.02
    goal_tolerance_m: float = 0.3
    max_leg_time_s: float = 60.0
    max_total_time_s: float = 180.0


@dataclass
class TourResult:
    goals: list[tuple[float, float]]
    reached: list[bool]
    final_distances_m: list[float]
    net_displacement_m: float
    path_length_m: float
    duration_s: float
    fallen: bool
    trace: list[tuple[float, float, float]] = field(default_factory=list)
    commands: list[tuple[float, float, float]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    leg_paths: list[list[tuple[float, float]]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.goals) and all(self.reached) and not self.fallen


def _reachable_goals(grid: OccupancyGrid, start: tuple[float, float], seed: int,
                     pool: int = 48, min_leg_m: float = 1.5,
                     max_leg_m: float = 15.0) -> list[tuple[tuple[float, float], float]]:
    """Reachable free-cell centres with their A* length from ``start`` (deterministic in ``seed``)."""
    import random

    free = traversable_mask(grid)
    ys, xs = np.nonzero(free)
    if len(ys) < 2:
        return []
    rng = random.Random(seed)
    candidates: list[tuple[tuple[float, float], float]] = []
    orders = list(range(len(ys)))
    rng.shuffle(orders)
    for index in orders[: pool * 3]:
        goal = grid.centre_of(int(ys[index]), int(xs[index]))
        if not grid.is_free(goal[0], goal[1]):
            continue
        plan = astar(grid, start, goal)
        if plan.success and min_leg_m <= plan.length_m <= max_leg_m:
            candidates.append((goal, float(plan.length_m)))
        if len(candidates) >= pool:
            break
    return candidates


def pick_tour_goals(grid: OccupancyGrid, start: tuple[float, float], *, seed: int = 0, n: int = 4,
                    min_total_m: float = 6.0, max_leg_m: float = 15.0) -> list[tuple[float, float]]:
    """``n`` reachable goals chosen farthest-first, so the tour covers the home instead of circling one corner.

    Returns fewer than ``n`` only when the grid offers too few reachable destinations; the caller reports that.
    """
    chosen: list[tuple[float, float]] = []
    current = start
    for _ in range(max(1, n)):
        candidates = _reachable_goals(grid, current, seed + len(chosen), max_leg_m=max_leg_m)
        if not candidates:
            break
        best, _length = max(candidates, key=lambda item: item[1])
        chosen.append(best)
        current = best
    # extend if the planned tour is shorter than the acceptance floor
    guard = 0
    while chosen and guard < 4:
        guard += 1
        total = 0.0
        pose = start
        for goal in chosen:
            plan = astar(grid, pose, goal)
            total += plan.length_m if plan.success else 0.0
            pose = goal
        if total >= min_total_m:
            break
        extra = _reachable_goals(grid, pose, seed + 100 + guard, max_leg_m=max_leg_m)
        if not extra:
            break
        chosen.append(max(extra, key=lambda item: item[1])[0])
    return _order_goals(start, chosen)


def _order_goals(start: tuple[float, float], goals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Nearest-neighbour order from the start, with the farthest goal last.

    Farthest-first picks a good *set*, but visiting it in pick order ping-pongs across the home (measured: 52 m of
    path for four goals in a 16 m house). Nearest-neighbour walks it as a loop; ending on the farthest goal keeps
    the tour's net displacement honest instead of finishing back at the start.
    """
    remaining = list(goals)
    ordered: list[tuple[float, float]] = []
    pose = start
    while remaining:
        nxt = min(remaining, key=lambda goal: math.hypot(goal[0] - pose[0], goal[1] - pose[1]))
        ordered.append(nxt)
        remaining.remove(nxt)
        pose = nxt
    if ordered:
        farthest = max(ordered, key=lambda goal: math.hypot(goal[0] - start[0], goal[1] - start[1]))
        last = ordered.index(farthest)
        ordered[last], ordered[-1] = ordered[-1], ordered[last]
    return ordered


@dataclass
class _Leg:
    index: int
    goal: tuple[float, float]
    path: list[tuple[float, float]]
    controller: PurePursuit
    t: float = 0.0


def run_tour(
    backend: TourBackend,
    grid: OccupancyGrid,
    goals: Sequence[tuple[float, float]],
    *,
    config: TourConfig | None = None,
    pursuit: PurePursuitConfig | None = None,
    limiter: CommandRateLimiter | None = None,
    extra: Callable[[], dict[str, Any]] | None = None,
) -> TourResult:
    """Drive env 0 through ``goals`` without teleporting; one record row per control step."""
    config = config or TourConfig()
    pursuit_cfg = pursuit or PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m)
    trace: list[tuple[float, float, float]] = []
    commands: list[tuple[float, float, float]] = []
    rows: list[dict[str, Any]] = []
    reached: list[bool] = []
    distances: list[float] = []
    leg_paths: list[list[tuple[float, float]]] = []
    notes: list[str] = []
    fallen = False
    t_total = 0.0

    pose = backend.poses()[0]
    trace.append(pose)
    for index, goal in enumerate(goals):
        plan = astar(grid, (pose[0], pose[1]), goal)
        if not plan.success:
            leg_paths.append([])
            reached.append(False)
            distances.append(float(math.hypot(goal[0] - pose[0], goal[1] - pose[1])))
            notes.append(f"goal {index + 1} unreachable: {plan.reason}")
            continue
        leg_paths.append(list(plan.path))
        leg = _Leg(index=index, goal=goal, path=plan.path, controller=PurePursuit(pursuit_cfg, limiter))
        leg.controller.reset()
        while leg.t < config.max_leg_time_s and t_total < config.max_total_time_s:
            command = leg.controller.command((pose[0], pose[1], pose[2]), leg.path, dt=config.dt)
            backend.step([command])
            pose = backend.poses()[0]
            fell = bool(backend.falls()[0])
            leg.t += config.dt
            t_total += config.dt
            step_command = (float(command["vx"]), float(command["vy"]), float(command["wz"]))
            trace.append(pose)
            commands.append(step_command)
            row = {
                "t": round(t_total, 4), "x": pose[0], "y": pose[1], "yaw": pose[2],
                "vx": step_command[0], "vy": step_command[1], "wz": step_command[2],
                "goal_index": index, "goal_x": goal[0], "goal_y": goal[1], "fallen": fell,
                "contact": not grid.is_free(pose[0], pose[1], inflated=False),
            }
            if extra is not None:
                row.update(extra())
            rows.append(row)
            if fell:
                fallen = True
                notes.append(f"fell during goal {index + 1}; legs after it were not run")
                break
            if leg.controller.at_goal(pose, goal):
                break
        final = float(math.hypot(goal[0] - pose[0], goal[1] - pose[1]))
        distances.append(final)
        reached.append(final <= config.goal_tolerance_m and not fallen)
        if fallen:
            for rest in goals[index + 1:]:
                reached.append(False)
                distances.append(float(math.hypot(rest[0] - pose[0], rest[1] - pose[1])))
            break

    steps = np.asarray([(x, y) for x, y, _ in trace], dtype=float)
    path = float(np.linalg.norm(np.diff(steps, axis=0), axis=1).sum()) if len(steps) > 1 else 0.0
    net = float(np.linalg.norm(steps[-1] - steps[0])) if len(steps) > 1 else 0.0
    return TourResult(goals=list(goals), reached=reached, final_distances_m=distances, net_displacement_m=net,
                      path_length_m=path, duration_s=t_total, fallen=fallen, trace=trace, commands=commands,
                      rows=rows, leg_paths=leg_paths, notes=notes)


def tour_npz_payload(grid: OccupancyGrid, result: TourResult, *, extra_arrays: dict[str, np.ndarray] | None = None,
                     scene_id: str = "") -> dict[str, np.ndarray]:
    """A single dict ready for ``np.savez_compressed``: poses (XYZW), joints/commands, goals, grid, depth."""
    rows = result.rows
    payload: dict[str, np.ndarray] = {
        "scene_id": np.array([scene_id or grid.scene_id]),
        "t": np.asarray([row["t"] for row in rows], dtype=np.float32),
        "pose_xyzw": np.asarray([
            [*(row.get("root_pos") or [row["x"], row["y"], 0.0]),
             *(row.get("root_quat_xyzw") or [0.0, 0.0, 0.0, 1.0])]
            for row in rows], dtype=np.float32),
        "cmd": np.asarray([[row["vx"], row["vy"], row["wz"]] for row in rows], dtype=np.float32),
        "goal_index": np.asarray([row["goal_index"] for row in rows], dtype=np.int16),
        "goal_xy": np.asarray([[row["goal_x"], row["goal_y"]] for row in rows], dtype=np.float32),
        "fallen": np.asarray([bool(row["fallen"]) for row in rows], dtype=bool),
        "contact": np.asarray([bool(row["contact"]) for row in rows], dtype=bool),
        "goals_xy": np.asarray(result.goals, dtype=np.float32).reshape(-1, 2),
        "reached": np.asarray(result.reached, dtype=bool),
        "grid": np.asarray(grid.grid, dtype=np.int8),
        "grid_origin_xy": np.asarray(grid.origin_xy, dtype=np.float64),
        "grid_resolution_m": np.array([grid.resolution_m]),
        "grid_scene_id": np.array([grid.scene_id]),
        "metrics": np.asarray([result.net_displacement_m, result.path_length_m, result.duration_s,
                               float(result.fallen), float(result.success)], dtype=np.float64),
    }
    if any("joint_pos" in row for row in rows):
        payload["joint_pos"] = np.asarray([row.get("joint_pos", []) for row in rows], dtype=np.float32)
        payload["root_quat_xyzw"] = np.asarray([row.get("root_quat_xyzw", []) for row in rows], dtype=np.float32)
        payload["root_pos"] = np.asarray([row.get("root_pos", []) for row in rows], dtype=np.float32)
    flat = [(leg, x, y) for leg, path in enumerate(result.leg_paths) for x, y in path]
    payload["path_xy"] = np.asarray([[x, y] for _leg, x, y in flat], dtype=np.float32).reshape(-1, 2)
    payload["path_leg"] = np.asarray([leg for leg, _x, _y in flat], dtype=np.int16)
    payload["path_offsets"] = np.asarray(
        np.cumsum([0] + [len(path) for path in result.leg_paths]), dtype=np.int32)
    if extra_arrays:
        payload.update(extra_arrays)
    return payload
