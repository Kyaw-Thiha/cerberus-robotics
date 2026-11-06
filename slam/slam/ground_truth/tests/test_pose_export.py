import math

import rclpy

from slam.slam.ground_truth.ground_truth_node import GroundTruthPublisher
from slam.slam.ground_truth.pose_export import (
    build_ground_truth_transform,
    build_odometry_fields,
    isaac_quat_to_ros,
    native_isaac_quat_to_ros,
)


def test_isaac_quat_to_ros_reorders_wxyz_to_xyzw():
    assert isaac_quat_to_ros(w=1.0, x=0.0, y=0.0, z=0.0) == (0.0, 0.0, 0.0, 1.0)


def test_isaac_quat_to_ros_nontrivial_rotation():
    half = math.pi / 4
    w_in, x_in, y_in, z_in = math.cos(half), 0.0, 0.0, math.sin(half)
    assert isaac_quat_to_ros(w=w_in, x=x_in, y=y_in, z=z_in) == (x_in, y_in, z_in, w_in)


def test_native_isaac_quat_to_ros_identical_on_both_conventions():
    wxyz = (0.5, 0.1, 0.2, 0.3)
    xyzw = (0.1, 0.2, 0.3, 0.5)
    assert native_isaac_quat_to_ros(wxyz, "wxyz") == (0.1, 0.2, 0.3, 0.5)
    assert native_isaac_quat_to_ros(xyzw, "xyzw") == (0.1, 0.2, 0.3, 0.5)


def test_native_isaac_quat_to_ros_default_uses_env_override(monkeypatch):
    monkeypatch.setenv("CERBERUS_QUAT_CONVENTION", "xyzw")
    assert native_isaac_quat_to_ros((0.1, 0.2, 0.3, 0.5)) == (0.1, 0.2, 0.3, 0.5)
    monkeypatch.setenv("CERBERUS_QUAT_CONVENTION", "wxyz")
    assert native_isaac_quat_to_ros((0.5, 0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3, 0.5)


def test_build_odometry_fields_uses_world_and_base_link_frames():
    fields = build_odometry_fields(
        position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.5, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.1),
    )
    assert fields.frame_id == "world"
    assert fields.child_frame_id == "base_link"
    assert fields.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_build_ground_truth_transform_uses_a_different_child_frame_than_odometry():
    transform = build_ground_truth_transform(position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0))
    assert transform.frame_id == "world"
    assert transform.child_frame_id == "base_link_gt"


def test_ground_truth_publisher_instantiates_and_publishes_without_error():
    rclpy.init()
    try:
        node = GroundTruthPublisher()
        try:
            node.publish_pose(
                sim_time=node.get_clock().now(),
                position=(1.0, 2.0, 0.3), isaac_quat=(1.0, 0.0, 0.0, 0.0),
                linear_velocity=(0.5, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.1),
            )
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
