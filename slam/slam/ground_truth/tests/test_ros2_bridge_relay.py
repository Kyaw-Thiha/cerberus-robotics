import rclpy

from slam.slam.ground_truth.ros2_bridge_relay import Ros2BridgeRelay
from slam.slam.ground_truth.sim_ipc import (
    TOPIC_CAMERA_FRONT_RGB,
    TOPIC_CAMERA_FRONT_RIGHT_RGB,
    ImagePayload,
    ImuPayload,
    PointCloudPayload,
    PosePayload,
    pack_image,
    pack_imu,
    pack_point_cloud,
    pack_pose,
)


def test_handle_pose_republishes_ground_truth_odom_and_tf():
    rclpy.init()
    try:
        relay = Ros2BridgeRelay()
        try:
            published = []
            relay._gt_pub._odom_pub.publish = lambda msg: published.append(msg)
            payload = PosePayload(
                sim_time_ns=1_000_000_000,
                position=(1.0, 2.0, 0.3),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            )
            relay.handle_pose(pack_pose(payload))
            assert len(published) == 1
            assert published[0].pose.pose.position.x == 1.0
        finally:
            relay.destroy_node()
    finally:
        rclpy.shutdown()


def test_handle_lidar_publishes_point_cloud_with_correct_width():
    rclpy.init()
    try:
        relay = Ros2BridgeRelay()
        try:
            published = []
            relay._lidar_pub.publish = lambda msg: published.append(msg)
            payload = PointCloudPayload(sim_time_ns=1, points_xyz=bytes(12 * 3), ring=bytes(2 * 3))
            relay.handle_lidar(pack_point_cloud(payload))
            assert len(published) == 1
            assert published[0].width == 3
        finally:
            relay.destroy_node()
    finally:
        rclpy.shutdown()


def test_handle_imu_publishes_imu_msg():
    rclpy.init()
    try:
        relay = Ros2BridgeRelay()
        try:
            published = []
            relay._imu_pub.publish = lambda msg: published.append(msg)
            payload = ImuPayload(
                sim_time_ns=1,
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.1),
                linear_acceleration=(0.0, 0.0, 9.81),
            )
            relay.handle_imu(pack_imu(payload))
            assert len(published) == 1
            assert published[0].linear_acceleration.z == 9.81
        finally:
            relay.destroy_node()
    finally:
        rclpy.shutdown()


def test_handle_image_dispatches_to_the_matching_topic_publisher():
    rclpy.init()
    try:
        relay = Ros2BridgeRelay()
        try:
            published = []
            relay._image_pubs[TOPIC_CAMERA_FRONT_RGB].publish = lambda msg: published.append(msg)
            payload = ImagePayload(sim_time_ns=1, height=2, width=2, encoding="rgb8", data=bytes(12))
            relay.handle_image(TOPIC_CAMERA_FRONT_RGB, pack_image(payload))
            assert len(published) == 1
            assert published[0].width == 2
        finally:
            relay.destroy_node()
    finally:
        rclpy.shutdown()


def test_handle_image_dispatches_the_stereo_right_camera_topic():
    # multicam_slam's rig (CAMERA_STEREO_SUITE) -- confirms the right-eye
    # topics are wired into _IMAGE_TOPICS the same generic way as every
    # other image topic, not just present in sim_ipc.py's constants.
    rclpy.init()
    try:
        relay = Ros2BridgeRelay()
        try:
            published = []
            relay._image_pubs[TOPIC_CAMERA_FRONT_RIGHT_RGB].publish = lambda msg: published.append(msg)
            payload = ImagePayload(sim_time_ns=1, height=2, width=2, encoding="rgb8", data=bytes(12))
            relay.handle_image(TOPIC_CAMERA_FRONT_RIGHT_RGB, pack_image(payload))
            assert len(published) == 1
            assert published[0].width == 2
        finally:
            relay.destroy_node()
    finally:
        rclpy.shutdown()
