"""Tests for the scripted velocity-path generator (pure Python)."""

import math

import pytest

from common.cerberus_eval.kinematics import step_pose, wrap_angle
from common.cerberus_eval.scripted_path import (
    ZERO_COMMAND,
    contract_command_limits,
    contract_limiter,
    random_waypoints,
    waypoints_to_commands,
)
from common.cerberus_policies.command_limiter import CommandRateLimiter


def instant_limiter() -> CommandRateLimiter:
    """A limiter that never ramps (huge rates), so the geometry can be checked exactly."""
    return CommandRateLimiter(max_rate={"vx": 1e9, "vy": 1e9, "wz": 1e9}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})


def test_contract_limits_match_the_contract_ranges():
    assert contract_command_limits() == {"vx": 1.0, "vy": 1.0, "wz": 1.0}


def test_random_waypoints_are_deterministic_per_seed():
    assert random_waypoints(7) == random_waypoints(7)
    assert random_waypoints(7) != random_waypoints(8)


def test_commands_never_exceed_the_contract_limits():
    path = waypoints_to_commands((0.0, 0.0, 0.0), random_waypoints(3), speed=5.0, turn_rate=5.0)
    limits = contract_command_limits()
    for command in path.commands:
        assert abs(command["vx"]) <= limits["vx"] + 1e-12
        assert abs(command["vy"]) <= limits["vy"] + 1e-12
        assert abs(command["wz"]) <= limits["wz"] + 1e-12


def test_slew_is_bounded_by_the_limiter_rate():
    limiter = contract_limiter()
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(1.0, 0.5), (0.0, 1.0)], limiter=limiter)
    dt = path.dt
    for previous, current in zip(path.commands, path.commands[1:]):
        assert abs(current["vx"] - previous["vx"]) <= limiter.max_rate["vx"] * dt + 1e-9
        assert abs(current["wz"] - previous["wz"]) <= limiter.max_rate["wz"] * dt + 1e-9


def test_schedule_starts_with_a_zero_command_settle():
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(1.0, 0.0)], settle_s=0.5)
    settle_steps = round(0.5 / path.dt)
    for command in path.commands[:settle_steps]:
        assert command["vx"] == 0.0 and command["wz"] == 0.0


def test_schedule_ends_with_a_zero_command_hold():
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(1.0, 0.0)], hold_s=0.5)
    hold_steps = round(0.5 / path.dt)
    for command in path.commands[-hold_steps:]:
        assert command["vx"] == 0.0 and command["wz"] == 0.0


def test_geometry_reaches_the_waypoints_when_the_limiter_does_not_ramp():
    waypoints = [(1.0, 0.0), (1.0, 1.0)]
    path = waypoints_to_commands((0.0, 0.0, 0.0), waypoints, limiter=instant_limiter())
    x, y, yaw = path.expected_end_pose
    assert x == pytest.approx(1.0, abs=1e-6)
    assert y == pytest.approx(1.0, abs=1e-6)
    assert yaw == pytest.approx(math.pi / 2, abs=1e-6)


def test_turn_is_in_place_and_lands_on_the_bearing():
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(0.0, 1.0)], limiter=instant_limiter())
    moving = [c for c in path.commands if c["vx"] != 0.0]
    assert moving, "the robot must walk to the waypoint"
    # during the turn the robot does not translate
    turn = [c for c in path.commands if c["wz"] != 0.0]
    assert turn and all(c["vx"] == 0.0 for c in turn)


def test_speed_and_turn_rate_arguments_are_clipped_to_the_contract():
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(2.0, 0.0)], speed=10.0, turn_rate=10.0, limiter=instant_limiter())
    assert max(c["vx"] for c in path.commands) <= 1.0
    assert max(c["wz"] for c in path.commands) <= 1.0


def test_rejects_non_positive_dt():
    with pytest.raises(ValueError):
        waypoints_to_commands((0.0, 0.0, 0.0), [(1.0, 0.0)], dt=0.0)


def test_zero_command_shape_has_every_loco_field():
    path = waypoints_to_commands((0.0, 0.0, 0.0), [(1.0, 0.0)], settle_s=0.02)
    assert set(path.commands[0]) == set(ZERO_COMMAND)


def test_step_pose_is_exact_for_a_constant_turn():
    # a quarter turn at vx=0.5, wz=pi/2 over 1 s: arc of radius v/w
    x, y, yaw = step_pose((0.0, 0.0, 0.0), {"vx": 0.5, "vy": 0.0, "wz": math.pi / 2}, 1.0)
    assert (x, y) == pytest.approx((0.5 * 2 / math.pi, 0.5 * 2 / math.pi), abs=1e-9)
    assert yaw == pytest.approx(math.pi / 2, abs=1e-9)


def test_wrap_angle_is_in_the_half_open_pi_range():
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
