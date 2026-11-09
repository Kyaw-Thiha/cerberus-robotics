"""Closed-loop kinematic test: A* path + pure pursuit on a unicycle, with the B1 limiter in the loop.

No physics, no Isaac, no policy. Its job is to answer "can the planner + controller pair reach the goal on this
grid?" BEFORE any GPU time, and to be the UPPER BOUND for the physics runs: a perfect tracker that never slips,
never falls and never lags. The gap between this and the physics result is a finding about the policy, not about
the planner.

A collision is counted when the simulated base CENTRE enters a cell the planner treats as blocked (blocked or
inflated). A path can be collision-free and the *tracked* pose still cut a corner -- that is exactly the controller
failure this test exists to surface, so it is an outcome (``collision``), not a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common.cerberus_eval.kinematics import step_pose
from common.cerberus_eval.nav_metrics import StepState, evaluate_episode
from common.cerberus_policies.command_limiter import CommandRateLimiter
from planning.global_planner.core.astar import astar
from planning.local_planner.core.pure_pursuit import PurePursuit, PurePursuitConfig
from planning.occupancy import OccupancyGrid
from planning.sampling import StartGoalPair

DEFAULT_MAX_TIME_S = 120.0
DEFAULT_MAX_COLLISION_STEPS = 10


@dataclass
class KinematicConfig:
    dt: float = 0.02
    max_time_s: float = DEFAULT_MAX_TIME_S
    max_collision_steps: int = DEFAULT_MAX_COLLISION_STEPS
    goal_tolerance_m: float = 0.5


def run_episode(
    grid: OccupancyGrid,
    pair: StartGoalPair,
    *,
    config: KinematicConfig | None = None,
    pursuit: PurePursuitConfig | None = None,
    limiter: CommandRateLimiter | None = None,
    initial_yaw: float | None = None,
) -> dict:
    """Drive the unicycle from ``pair.start`` to ``pair.goal`` along the A* path; return a metrics record."""
    config = config or KinematicConfig()
    pursuit_cfg = pursuit or PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m)
    plan = astar(grid, pair.start, pair.goal)
    if not plan.success:
        return {
            "outcome": "planner_failure", "success": False, "scene": grid.scene_id,
            "start": pair.start, "goal": pair.goal, "shortest_path_m": pair.shortest_path_m,
            "path_length_m": 0.0, "spl": 0.0, "time_s": 0.0, "steps": 0, "final_distance_m": None,
            "contacts": 0, "blocked_cell_steps": 0, "notes": [plan.reason],
        }

    controller = PurePursuit(pursuit_cfg, limiter)
    controller.reset()
    if initial_yaw is None:
        (sx, sy), (nx, ny) = plan.path[0], plan.path[min(1, len(plan.path) - 1)]
        initial_yaw = math.atan2(ny - sy, nx - sx)
    pose = (float(pair.start[0]), float(pair.start[1]), float(initial_yaw))

    rows: list[StepState] = []
    blocked_steps = 0
    margin_steps = 0
    max_steps = int(config.max_time_s / config.dt)
    for step in range(max_steps):
        # A collision is entering BLOCKED geometry. Entering the inflation margin is not a collision -- that band is
        # the planner's safety margin, and counting it as one would blame the controller for the planner's conservatism.
        in_blocked = not grid.is_free(pose[0], pose[1], inflated=False)
        in_margin = not in_blocked and not grid.is_free(pose[0], pose[1], inflated=True)
        blocked_steps += int(in_blocked)
        margin_steps += int(in_margin)
        rows.append(StepState(t=step * config.dt, x=pose[0], y=pose[1], yaw=pose[2],
                              fallen=False, nonfinite=False, contact=in_blocked))
        if controller.at_goal(pose, pair.goal):
            break
        if blocked_steps > config.max_collision_steps:
            break
        command = controller.command(pose, plan.path, dt=config.dt)
        pose = step_pose(pose, command, config.dt)

    record = evaluate_episode(rows, pair.goal, tolerance_m=config.goal_tolerance_m,
                              shortest_path_m=pair.shortest_path_m)
    if blocked_steps > 0:
        record["outcome"] = "collision"
        record["success"] = False
        record["spl"] = 0.0
    record.update({
        "scene": grid.scene_id,
        "start": pair.start,
        "goal": pair.goal,
        "blocked_cell_steps": blocked_steps,
        "margin_steps": margin_steps,
        "planned_length_m": plan.length_m,
        "planner_expanded": plan.expanded,
    })
    return record
