import math

import pytest

from slam.slam.eval.showcase_camera_follow import compute_chase_cam_pose


def test_chase_cam_trails_directly_behind_at_zero_yaw():
    position, orientation_xyzw = compute_chase_cam_pose(robot_position=(0.0, 0.0, 0.0), robot_yaw=0.0)
    x, y, z = position
    assert x < 0.0  # behind along +x heading means camera sits at -x
    assert y == pytest.approx(0.0)
    assert z > 0.0  # above the robot
    assert orientation_xyzw == (0.0, 0.0, 0.0, 1.0)  # level, no yaw offset


def test_chase_cam_orientation_tracks_yaw_only():
    _, orientation_xyzw = compute_chase_cam_pose(robot_position=(0.0, 0.0, 0.0), robot_yaw=math.pi / 2)
    x, y, z, w = orientation_xyzw
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)  # roll/pitch never touched, only yaw (z, w)
    assert z == pytest.approx(math.sin(math.pi / 4))
    assert w == pytest.approx(math.cos(math.pi / 4))


def test_chase_cam_position_follows_yaw_direction():
    position, _ = compute_chase_cam_pose(robot_position=(5.0, 5.0, 0.3), robot_yaw=math.pi / 2)
    x, y, _z = position
    assert x == pytest.approx(5.0)  # cos(pi/2)=0, no x offset at this heading
    assert y < 5.0  # sin(pi/2)=1, camera trails behind along -y
