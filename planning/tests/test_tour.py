"""The chained exploration tour: goal picking, continuous driving, honest misses, and the record payload."""

import math

import numpy as np

from common.cerberus_eval.nav_loop import MockBackend
from planning.global_planner.core.astar import astar, path_is_collision_free
from planning.occupancy import BLOCKED, FREE, OccupancyConfig, OccupancyGrid, inflate
from planning.tour import TourConfig, pick_tour_goals, run_tour, tour_npz_payload


def make_grid(mask, *, res=0.1, radius_m=0.2):
    grid = np.where(mask, FREE, BLOCKED).astype(np.int8)
    return OccupancyGrid(grid=grid, origin_xy=(0.0, 0.0), resolution_m=res, floor_z=0.0,
                         config=OccupancyConfig(resolution_m=res), scene_id="toy",
                         inflated=inflate(grid, radius_m, res))


def test_goal_picking_is_deterministic_reachable_and_long_enough():
    grid = make_grid(np.ones((120, 140), dtype=bool))
    start = (1.0, 1.0)
    a = pick_tour_goals(grid, start, seed=5, n=3, min_total_m=6.0)
    b = pick_tour_goals(grid, start, seed=5, n=3, min_total_m=6.0)
    assert a == b and len(a) >= 3
    total = 0.0
    pose = start
    for goal in a:
        assert grid.is_free(*goal)
        plan = astar(grid, pose, goal)
        assert plan.success
        total += plan.length_m
        pose = goal
    assert total >= 6.0


def test_tour_goes_round_a_wall_and_reaches_every_goal():
    mask = np.ones((100, 100), dtype=bool)
    mask[:, 50] = False          # a wall ...
    mask[40:60, 50] = True       # ... with one gap
    grid = make_grid(mask)
    backend = MockBackend(1)
    backend.reset([(1.0, 1.0, 0.0)])
    goals = [(1.0, 8.0), (8.0, 8.0), (8.0, 1.0)]
    result = run_tour(backend, grid, goals, config=TourConfig(goal_tolerance_m=0.3, max_leg_time_s=90.0,
                                                             max_total_time_s=240.0))
    assert result.success, (result.reached, result.final_distances_m, result.notes)
    assert result.net_displacement_m > 3.0
    # the inflation band is the planner's safety margin, not geometry: a corner cut through it is not a collision
    assert path_is_collision_free(grid, [(x, y) for x, y, _ in result.trace], inflated=False)
    assert len(result.rows) == len(result.commands) == len(result.trace) - 1


def test_unreachable_goal_is_reported_not_faked():
    mask = np.ones((60, 60), dtype=bool)
    mask[:, 30] = False
    grid = make_grid(mask)
    backend = MockBackend(1)
    backend.reset([(1.0, 1.0, 0.0)])
    result = run_tour(backend, grid, [(5.0, 5.0)], config=TourConfig(max_leg_time_s=5.0, max_total_time_s=10.0))
    assert not result.success
    assert result.reached == [False]
    assert result.notes and "unreachable" in result.notes[0]
    assert math.isclose(result.final_distances_m[0], math.hypot(4.0, 4.0), rel_tol=1e-6)


def test_payload_has_the_record_once_fields():
    grid = make_grid(np.ones((40, 40), dtype=bool))
    backend = MockBackend(1)
    backend.reset([(0.5, 0.5, 0.0)])
    result = run_tour(backend, grid, [(2.5, 2.5)], config=TourConfig(goal_tolerance_m=0.3))
    payload = tour_npz_payload(grid, result, scene_id="toy")
    for key in ("t", "pose_xyzw", "cmd", "goal_index", "goal_xy", "fallen", "contact", "goals_xy",
                "reached", "grid", "grid_origin_xy", "grid_resolution_m", "metrics",
                "path_xy", "path_leg", "path_offsets"):
        assert key in payload, key
    n = len(payload["t"])
    assert payload["pose_xyzw"].shape == (n, 7)
    assert payload["cmd"].shape == (n, 3)
    assert payload["grid"].shape == grid.grid.shape
    np.savez_compressed("/tmp/tour_test.npz", **payload)  # must be serialisable as-is
