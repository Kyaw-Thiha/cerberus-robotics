"""The navigation loop, shared by the physics driver and the local mock.

Pure Python (stdlib + numpy) and backend-agnostic: a backend only has to provide ``num_envs``, ``reset(poses)``,
``step(commands)``, ``poses()`` and ``falls()``. ``scripts/nav_eval.py`` supplies the Isaac backend on the pod and a
unicycle :class:`MockBackend` locally, so the episode bookkeeping, the metrics and the measurement-bug guards are
exercised before any GPU time.

Per episode: reset to the start pose, plan with A*, follow with pure pursuit, step, and classify with
``nav_metrics.evaluate_episode``. Two guards against "success" that is a measurement bug:

* the walked pose trace is re-checked against the grid (``trace_collision_free``) -- a success whose trace crosses
  blocked geometry is flagged ``suspect_success``;
* the goal must be reached before any fall (that is inside ``evaluate_episode``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from common.cerberus_eval.kinematics import step_pose
from common.cerberus_eval.nav_metrics import StepState, evaluate_episode
from common.cerberus_policies.command_limiter import CommandRateLimiter
from planning.global_planner.core.astar import astar, path_is_collision_free
from planning.local_planner.core.pure_pursuit import PurePursuit, PurePursuitConfig
from planning.occupancy import OccupancyGrid
from planning.sampling import StartGoalPair


class NavBackend(Protocol):
    num_envs: int

    def reset(self, poses: Sequence[tuple[float, float, float] | None]) -> None:
        """Teleport the envs whose entry is not None; leave the others running."""
    def step(self, commands: Sequence[dict[str, float]]) -> None: ...
    def poses(self) -> list[tuple[float, float, float]]: ...
    def falls(self) -> list[bool]: ...


@dataclass
class NavConfig:
    dt: float = 0.02
    max_time_s: float = 120.0
    goal_tolerance_m: float = 0.5
    fall_settle_steps: int = 1


@dataclass
class _Episode:
    pair_index: int
    pair: StartGoalPair
    path: list[tuple[float, float]]
    controller: PurePursuit
    rows: list[StepState] = field(default_factory=list)
    t: float = 0.0


class MockBackend:
    """A unicycle per env: the local stand-in for the physics env (no Isaac, no policy)."""

    def __init__(self, num_envs: int, dt: float = 0.02, speed_scale: float = 1.0) -> None:
        self.num_envs = num_envs
        self.dt = dt
        self.speed_scale = speed_scale
        self._poses = [(0.0, 0.0, 0.0)] * num_envs

    def reset(self, poses: Sequence[tuple[float, float, float] | None]) -> None:
        self._poses = [
            (float(p[0]), float(p[1]), float(p[2])) if p is not None else pose
            for pose, p in zip(self._poses, poses)
        ]

    def step(self, commands: Sequence[dict[str, float]]) -> None:
        self._poses = [
            step_pose(pose, {**cmd, "vx": cmd["vx"] * self.speed_scale, "wz": cmd["wz"] * self.speed_scale}, self.dt)
            for pose, cmd in zip(self._poses, commands)
        ]

    def poses(self) -> list[tuple[float, float, float]]:
        return list(self._poses)

    def falls(self) -> list[bool]:
        return [False] * self.num_envs


def _initial_yaw(path: Sequence[tuple[float, float]], fallback: float = 0.0) -> float:
    if len(path) < 2:
        return fallback
    (x0, y0), (x1, y1) = path[0], path[1]
    return math.atan2(y1 - y0, x1 - x0)


def run_navigation(
    backend: NavBackend,
    grid: OccupancyGrid,
    pairs: Sequence[StartGoalPair],
    *,
    seed: int = 0,
    config: NavConfig | None = None,
    pursuit: PurePursuitConfig | None = None,
    limiter_factory: Callable[[], CommandRateLimiter] | None = None,
    on_step: Callable[[int, list[dict[str, float]]], None] | None = None,
    max_episodes: int | None = None,
) -> list[dict[str, Any]]:
    """Run ``pairs`` (up to ``max_episodes``) across ``backend.num_envs`` envs; return one record per episode."""
    config = config or NavConfig()
    pursuit_cfg = pursuit or PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m)
    limit = len(pairs) if max_episodes is None else min(len(pairs), max_episodes)
    queue = list(range(limit))
    records: list[dict[str, Any]] = []
    active: dict[int, _Episode] = {}
    rejected: list[dict[str, Any]] = []

    def start(env_index: int, pair_index: int) -> bool:
        pair = pairs[pair_index]
        plan = astar(grid, pair.start, pair.goal)
        if not plan.success:
            rejected.append({"pair_index": pair_index, "reason": plan.reason, "start": pair.start, "goal": pair.goal})
            return False
        controller = PurePursuit(pursuit_cfg, limiter_factory() if limiter_factory else None)
        controller.reset()
        active[env_index] = _Episode(pair_index=pair_index, pair=pair, path=plan.path, controller=controller)
        return True

    def fill() -> None:
        """Start the next pair on every free env; only those envs are teleported."""
        poses: list[tuple[float, float, float] | None] = [None] * backend.num_envs
        changed = False
        for env_index in range(backend.num_envs):
            if env_index in active:
                continue
            while queue:
                if start(env_index, queue.pop(0)):
                    episode = active[env_index]
                    poses[env_index] = (episode.pair.start[0], episode.pair.start[1], _initial_yaw(episode.path))
                    changed = True
                    break
        if changed:
            backend.reset(poses)

    fill()
    step_index = 0
    while active:
        poses = backend.poses()
        commands: list[dict[str, float]] = [
            active[i].controller.command(poses[i], active[i].path, dt=config.dt) if i in active
            else {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            for i in range(backend.num_envs)
        ]
        backend.step(commands)
        if on_step is not None:
            on_step(step_index, commands)
        step_index += 1
        poses = backend.poses()
        falls = backend.falls()

        for env_index in list(active):
            episode = active[env_index]
            episode.t += config.dt
            pose = poses[env_index]
            episode.rows.append(StepState(
                t=episode.t, x=pose[0], y=pose[1], yaw=pose[2], fallen=bool(falls[env_index]),
                contact=not grid.is_free(pose[0], pose[1], inflated=False),
            ))
            done = bool(falls[env_index]) or episode.controller.at_goal(pose, episode.pair.goal) \
                or episode.t >= config.max_time_s
            if not done:
                continue
            record = evaluate_episode(episode.rows, episode.pair.goal, tolerance_m=config.goal_tolerance_m,
                                      shortest_path_m=episode.pair.shortest_path_m)
            trace = [(r.x, r.y) for r in episode.rows]
            record.update({
                "scene": grid.scene_id,
                "seed": seed,
                "pair_index": episode.pair_index,
                "start": episode.pair.start,
                "goal": episode.pair.goal,
                "planned_length_m": episode.pair.shortest_path_m,
                "trace_collision_free": path_is_collision_free(grid, trace, inflated=False),
                "fall_settle_steps": config.fall_settle_steps,
            })
            record["suspect_success"] = bool(record["outcome"] == "success" and not record["trace_collision_free"])
            records.append(record)
            del active[env_index]

        if len(active) < backend.num_envs:
            fill()

    records.sort(key=lambda r: r["pair_index"])
    for item in rejected:
        records.append({
            "scene": grid.scene_id, "seed": seed, "pair_index": item["pair_index"],
            "outcome": "planner_failure", "success": False, "spl": 0.0, "time_s": 0.0,
            "path_length_m": 0.0, "contacts": 0, "notes": [item["reason"]],
            "start": item["start"], "goal": item["goal"],
        })
    records.sort(key=lambda r: r["pair_index"])
    return records
