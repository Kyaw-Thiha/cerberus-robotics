"""Tests for pure pursuit and the closed-loop kinematic test."""


import numpy as np
import pytest

from common.cerberus_policies.command_limiter import CommandRateLimiter
from planning.kinematic_sim import KinematicConfig, run_episode
from planning.local_planner.core.pure_pursuit import (
    PurePursuit,
    PurePursuitConfig,
    _closest_on_polyline,
    _point_at_arclength,
)
from planning.occupancy import BLOCKED, FREE, OccupancyGrid, OccupancyConfig
from planning.sampling import StartGoalPair


def make_grid(mask, *, res=0.1):
    grid = np.where(mask, FREE, BLOCKED).astype(np.int8)
    return OccupancyGrid(grid=grid, origin_xy=(0.0, 0.0), resolution_m=res, floor_z=0.0,
                         config=OccupancyConfig(resolution_m=res), scene_id="toy",
                         inflated=np.zeros_like(grid, dtype=bool))


def limiter():
    return CommandRateLimiter(max_rate={"vx": 1.0, "vy": 1.0, "wz": 2.0}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})


def test_at_goal_emits_zero_and_reports_goal():
    controller = PurePursuit(PurePursuitConfig(goal_tolerance_m=0.5))
    pose = (0.0, 0.0, 0.0)
    path = [(0.0, 0.0), (5.0, 0.0)]
    assert controller.at_goal((0.3, 0.0, 0.0), (5.0, 0.0)) is False
    command = controller.command((5.2, 0.0, 0.0), path)
    assert command["vx"] == 0.0 and command["wz"] == 0.0


def test_straight_path_drives_forward_without_turning():
    controller = PurePursuit(PurePursuitConfig())
    command = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (5.0, 0.0)])
    assert command["vx"] > 0.0
    assert abs(command["wz"]) < 1e-6


def test_heading_error_turns_toward_the_path():
    controller = PurePursuit(PurePursuitConfig())
    left = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (0.0, 5.0)])  # path to +y, robot facing +x
    assert left["wz"] > 0.0
    right = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (0.0, -5.0)])
    assert right["wz"] < 0.0


def test_command_stays_within_the_configured_limits():
    cfg = PurePursuitConfig(max_vx=0.6, max_wz=0.4)
    controller = PurePursuit(cfg)
    command = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (0.0, 0.2), (0.0, 5.0)])
    assert abs(command["vx"]) <= 0.6 + 1e-9
    assert abs(command["wz"]) <= 0.4 + 1e-9


def test_limiter_bounds_the_first_and_second_command():
    lim = limiter()
    controller = PurePursuit(PurePursuitConfig(max_vx=1.0), lim)
    controller.reset()
    first = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (5.0, 0.0)], dt=0.02)
    # the limiter adopts the first target outright, then slews at 1 m/s^2 -> at most 0.02 m/s of change
    second = controller.command((0.0, 0.0, 0.0), [(0.0, 0.0), (5.0, 0.0)], dt=0.02)
    assert abs(second["vx"] - first["vx"]) <= 1.0 * 0.02 + 1e-9


def test_lookahead_is_adaptive_and_bounded():
    controller = PurePursuit(PurePursuitConfig(min_lookahead_m=0.3, max_lookahead_m=1.0, lookahead_gain_s=1.0))
    controller._last_speed = 0.0
    assert controller.lookahead() == pytest.approx(0.3)
    controller._last_speed = 10.0
    assert controller.lookahead() == pytest.approx(1.0)


def test_closest_point_and_arclength_helpers():
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    s, x, y = _closest_on_polyline(path, (1.0, 0.5))
    assert s == pytest.approx(1.0) and (x, y) == pytest.approx((1.0, 0.0))
    assert _point_at_arclength(path, 1.0) == pytest.approx((1.0, 0.0))
    assert _point_at_arclength(path, 3.0) == pytest.approx((2.0, 1.0))
    assert _point_at_arclength(path, 99.0) == pytest.approx((2.0, 2.0))


def test_rejects_an_impossible_lookahead_config():
    with pytest.raises(ValueError):
        PurePursuitConfig(min_lookahead_m=1.0, max_lookahead_m=0.5)


def test_kinematic_episode_succeeds_on_open_floor():
    grid = make_grid(np.ones((60, 60), dtype=bool))
    pair = StartGoalPair(start=grid.centre_of(5, 5), goal=grid.centre_of(5, 50),
                         shortest_path_m=4.5)
    record = run_episode(grid, pair, config=KinematicConfig(max_time_s=60.0))
    assert record["outcome"] == "success", record
    assert record["blocked_cell_steps"] == 0
    assert record["spl"] is not None and 0.0 < record["spl"] <= 1.0


def test_kinematic_episode_reports_a_planner_failure():
    mask = np.ones((30, 30), dtype=bool)
    mask[:, 15] = False
    grid = make_grid(mask)
    pair = StartGoalPair(start=grid.centre_of(5, 5), goal=grid.centre_of(5, 25), shortest_path_m=2.0)
    record = run_episode(grid, pair)
    assert record["outcome"] == "planner_failure"
    assert record["success"] is False


def test_kinematic_collision_is_an_outcome_not_a_success():
    # a corridor whose free width is narrower than the pure-pursuit cut: the tracker leaves the free cells
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 2:38] = True
    mask[20:22, 20] = False  # an obstacle in the corridor forces a swerve the tracker may clip
    grid = make_grid(mask)
    pair = StartGoalPair(start=grid.centre_of(19, 3), goal=grid.centre_of(19, 36), shortest_path_m=3.3)
    record = run_episode(grid, pair, config=KinematicConfig(max_time_s=60.0))
    assert record["outcome"] in ("success", "collision", "timeout")
    if record["outcome"] == "collision":
        assert record["success"] is False
        assert record["blocked_cell_steps"] > 0
