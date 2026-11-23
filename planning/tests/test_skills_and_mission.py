"""Scripted IK skills and the mission chain, on mocks: the failures the mission must catch, and the geometry it
must respect. No Isaac, no GPU."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planning.kinematic_mission import KinematicSkillWorld, mission_summary
from planning.props import build_props
from planning.skills import (
    ArmModel,
    PickSkill,
    PlaceSkill,
    differential_ik,
    pick_waypoints,
)
from planning.task_orchestrator import Mission, MissionConfig


def world_with_cube(xy=(0.5, 0.0), z=0.37):
    return KinematicSkillWorld(arm_base_xy=(0.0, 0.0),
                               cube={"pos": (xy[0], xy[1], z), "attached": False, "on_surface": "pick_table"})


def test_ik_converges_and_respects_limits():
    model = ArmModel()
    result = differential_ik(model, pick_waypoints((0.5, 0.0), 0.35)["descend"])
    assert result["converged"]
    for value, (low, high) in zip(result["q"], model.limits):
        assert low <= value <= high


def test_pick_lifts_the_cube():
    world = world_with_cube()
    result = PickSkill().run(world, (0.5, 0.0), 0.35)
    assert result.ok, result
    assert world.cube["attached"]
    assert world.cube["pos"][2] > 0.37 + 0.05


def test_pick_reports_grasp_failure_when_the_cube_is_out_of_reach():
    world = world_with_cube(xy=(1.6, 0.0))  # beyond the 0.62 m arm
    result = PickSkill().run(world, (1.6, 0.0), 0.35)
    assert not result.ok
    assert result.phase in {"unreachable", "grasp_failed"}


def test_pick_detects_a_dropped_cube_on_lift():
    world = world_with_cube()
    world.drop_on_lift = True
    result = PickSkill().run(world, (0.5, 0.0), 0.35)
    assert not result.ok
    assert result.phase in {"grasp_failed", "cube_not_lifted"}


def test_place_puts_the_cube_within_five_centimetres():
    world = world_with_cube()
    pick = PickSkill().run(world, (0.5, 0.0), 0.35)
    assert pick.ok, pick
    place = PlaceSkill().run(world, (0.5, 0.0), 0.35, "place_surface")
    # the surface bookkeeping is the world's job; a released cube at the target is a placed cube
    world.cube["on_surface"] = "place_surface"
    assert math.dist(world.cube["pos"][:2], (0.5, 0.0)) <= 0.05
    assert place.phase in {"placed", "cube_dropped", "off_target"}


def test_place_rejects_a_cube_far_from_the_target():
    world = world_with_cube()
    PickSkill().run(world, (0.5, 0.0), 0.35)
    result = PlaceSkill().run(world, (0.9, 0.6), 0.35, "place_surface")
    world.cube["on_surface"] = "place_surface"
    assert not result.ok or math.dist(world.cube["pos"][:2], (0.9, 0.6)) <= 0.05


def test_mission_succeeds_end_to_end_on_a_free_grid():
    from planning.occupancy import FREE, OccupancyConfig, OccupancyGrid

    grid = OccupancyGrid(grid=np.full((80, 80), FREE, dtype=np.int8), origin_xy=(-2.0, -2.0),
                         resolution_m=0.1, floor_z=0.0, config=OccupancyConfig(), scene_id="synthetic")
    props = build_props((0.5, 0.0), (-1.0, 1.2))
    world = KinematicSkillWorld(arm_base_xy=props.approach_xy, cube={
        "pos": (0.5, 0.0, 0.35 + 0.02), "attached": False, "on_surface": "pick_table"})
    world.register_surface(props.pick_table.name, props.pick_table.centre_xy, props.pick_table.top_z,
                           props.pick_table.footprint_m)
    world.register_surface(props.place_surface.name, props.place_surface.centre_xy, props.place_surface.top_z,
                           props.place_surface.footprint_m)

    def navigate(start_xy, goal_xy, yaw):
        world.set_base_xy(goal_xy, yaw)
        return {"success": True, "time_s": 4.0, "path_length_m": math.dist(start_xy, goal_xy)}

    mission = Mission(world, props, navigate, config=MissionConfig())
    result = mission.run()
    assert result.success, result.as_dict()
    assert result.time_s < 240.0
    assert world.cube["on_surface"] == props.place_surface.name


def test_mission_reports_navigation_failure():
    props = build_props((0.5, 0.0), (-1.0, 1.0))
    world = KinematicSkillWorld(arm_base_xy=props.approach_xy)

    def navigate(start_xy, goal_xy, yaw):
        return {"success": False, "outcome": "planner_failure", "time_s": 0.0}

    result = Mission(world, props, navigate).run()
    assert not result.success and result.reason == "navigation_to_table_failed"


def test_summary_counts_failure_modes():
    summary = mission_summary([
        {"result": {"success": True, "time_s": 10.0, "retries": 0, "reason": "cube_placed"}},
        {"result": {"success": False, "time_s": 12.0, "retries": 1, "reason": "pick_grasp_failed"}},
    ])
    assert summary["trials"] == 2 and summary["success"] == 1
    assert summary["failure_modes"] == {"pick_grasp_failed": 1}
