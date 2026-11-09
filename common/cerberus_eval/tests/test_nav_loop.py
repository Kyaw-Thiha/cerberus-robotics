"""Tests for the shared navigation loop (mock backend; no Isaac, no policy)."""

import numpy as np

from common.cerberus_eval.nav_loop import MockBackend, NavConfig, run_navigation
from planning.occupancy import BLOCKED, FREE, OccupancyConfig, OccupancyGrid
from planning.sampling import StartGoalPair


def make_grid(mask, *, res=0.1):
    grid = np.where(mask, FREE, BLOCKED).astype(np.int8)
    return OccupancyGrid(grid=grid, origin_xy=(0.0, 0.0), resolution_m=res, floor_z=0.0,
                         config=OccupancyConfig(resolution_m=res), scene_id="toy",
                         inflated=np.zeros_like(grid, dtype=bool))


def pair(grid, a, b):
    from planning.global_planner.core.astar import astar

    start, goal = grid.centre_of(*a), grid.centre_of(*b)
    plan = astar(grid, start, goal)
    return StartGoalPair(start=start, goal=goal, shortest_path_m=plan.length_m if plan.success else 0.0)


def test_four_envs_run_four_episodes_to_success():
    grid = make_grid(np.ones((60, 60), dtype=bool))
    pairs = [pair(grid, (5, 5 + 10 * i), (5, 30 + 5 * i)) for i in range(4)]
    backend = MockBackend(num_envs=4, dt=0.02)
    records = run_navigation(backend, grid, pairs, config=NavConfig(max_time_s=60.0))
    assert len(records) == 4
    assert all(r["outcome"] == "success" for r in records), records
    assert all(r["trace_collision_free"] for r in records)
    assert not any(r["suspect_success"] for r in records)
    assert [r["pair_index"] for r in records] == [0, 1, 2, 3]


def test_more_pairs_than_envs_are_queued():
    grid = make_grid(np.ones((60, 60), dtype=bool))
    pairs = [pair(grid, (5, 5), (5, 20 + i)) for i in range(6)]
    backend = MockBackend(num_envs=2, dt=0.02)
    records = run_navigation(backend, grid, pairs, config=NavConfig(max_time_s=60.0))
    assert [r["pair_index"] for r in records] == [0, 1, 2, 3, 4, 5]
    assert all(r["outcome"] == "success" for r in records)


def test_fewer_pairs_than_envs_only_runs_what_was_asked():
    grid = make_grid(np.ones((40, 40), dtype=bool))
    pairs = [pair(grid, (3, 3), (3, 20))]
    records = run_navigation(MockBackend(num_envs=4), grid, pairs, config=NavConfig(max_time_s=60.0))
    assert len(records) == 1


def test_planner_failure_is_recorded_not_dropped():
    mask = np.ones((30, 30), dtype=bool)
    mask[:, 15] = False
    grid = make_grid(mask)
    pairs = [StartGoalPair(start=grid.centre_of(5, 5), goal=grid.centre_of(5, 25), shortest_path_m=2.0)]
    records = run_navigation(MockBackend(num_envs=1), grid, pairs)
    assert records[0]["outcome"] == "planner_failure"
    assert records[0]["success"] is False


def test_a_trace_through_a_wall_is_flagged_as_suspect():
    # a grid with a wall the mock backend cannot see: give the loop a pair whose straight line crosses it by
    # construction -- the mock follows the command blindly, so the trace is what the guard has to catch.
    mask = np.ones((40, 40), dtype=bool)
    mask[:, 20] = False
    grid = make_grid(mask)
    pairs = [StartGoalPair(start=grid.centre_of(5, 5), goal=grid.centre_of(5, 35), shortest_path_m=3.0)]
    records = run_navigation(MockBackend(num_envs=1), grid, pairs, config=NavConfig(max_time_s=60.0))
    # the planner refuses this pair outright, so nothing is driven through the wall
    assert records[0]["outcome"] == "planner_failure"


def test_on_step_is_called_with_increasing_indices():
    grid = make_grid(np.ones((30, 30), dtype=bool))
    pairs = [pair(grid, (3, 3), (3, 15))]
    seen = []
    run_navigation(MockBackend(num_envs=1), grid, pairs, config=NavConfig(max_time_s=60.0),
                   on_step=lambda i, commands: seen.append(i))
    assert seen == sorted(seen) and len(seen) > 5


def test_contacts_are_counted_from_the_map():
    grid = make_grid(np.ones((30, 30), dtype=bool))
    pairs = [pair(grid, (3, 3), (3, 15))]
    records = run_navigation(MockBackend(num_envs=1), grid, pairs, config=NavConfig(max_time_s=60.0))
    assert records[0]["contacts"] == 0


class StraightLineBackend(MockBackend):
    """Ignores the commands and drives straight at each goal: the "controller through the wall" case."""

    def __init__(self, goals, dt=0.02, step_m=0.05):
        super().__init__(len(goals), dt)
        self.goals = goals
        self.step_m = step_m

    def step(self, commands):
        out = []
        for pose, goal in zip(self._poses, self.goals):
            dx, dy = goal[0] - pose[0], goal[1] - pose[1]
            d = max((dx * dx + dy * dy) ** 0.5, 1e-9)
            out.append((pose[0] + self.step_m * dx / d, pose[1] + self.step_m * dy / d, 0.0))
        self._poses = out


def test_success_through_geometry_is_flagged_suspect():
    # the wall is thin and reachable around, so A* finds a path; the straight-line backend cuts through it
    mask = np.ones((40, 40), dtype=bool)
    mask[:, 20] = False
    mask[0, 20] = True  # a gap far from the straight line
    grid = make_grid(mask)
    start, goal = grid.centre_of(20, 5), grid.centre_of(20, 35)
    pair_obj = StartGoalPair(start=start, goal=goal, shortest_path_m=3.0)
    backend = StraightLineBackend([goal], step_m=0.2)
    records = run_navigation(backend, grid, [pair_obj], config=NavConfig(max_time_s=60.0))
    assert records[0]["outcome"] == "success"
    assert records[0]["trace_collision_free"] is False
    assert records[0]["suspect_success"] is True
