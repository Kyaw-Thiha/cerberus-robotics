"""rclpy-side half of the Isaac-Lab-to-ROS2 bridge -- runs under pixi's
sim-ros2 env (NOT Isaac Sim's own interpreter, see sim_ipc.py's module
docstring for why they're separate processes). Subscribes to the zmq PUB
socket sim_ground_truth_loop.py publishes on, and republishes each payload
onto this repo's real ROS2 topics:
- pose -> GroundTruthPublisher.publish_pose() (slam/slam/ground_truth/
  ground_truth_node.py, unchanged, already correct).
- lidar/camera payloads -> sensor_msgs/PointCloud2 / Image, built by
  sensor_msg_builders.py, published directly (no per-modality adapter
  class needed -- these are raw sensor topics, not the SLAM-backend-output
  contract common_adapter.py handles).

Topic names match slam/slam/eval/record_mission_bag.py's existing
RAW_SENSOR_TOPICS_LIDAR_CAMERA / RAW_SENSOR_TOPICS_CAMERA_ONLY constants --
this is what makes those "PLACEHOLDER, not independently confirmed" topic
names in record_mission_bag.py real.
"""

from __future__ import annotations

import rclpy
import zmq
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu, PointCloud2

from slam.slam.ground_truth.ground_truth_node import GroundTruthPublisher
from slam.slam.ground_truth.pose_export import (
    isaac_quat_to_ros,  # noqa: F401  -- re-exported for callers importing from here
)
from slam.slam.ground_truth.sensor_msg_builders import (
    build_image_msg,
    build_imu_msg,
    build_point_cloud_msg,
)
from slam.slam.ground_truth.sim_ipc import (
    IPC_ADDRESS,
    TOPIC_CAMERA_FRONT_DEPTH,
    TOPIC_CAMERA_FRONT_RGB,
    TOPIC_CAMERA_FRONT_RIGHT_RGB,
    TOPIC_CAMERA_REAR_DEPTH,
    TOPIC_CAMERA_REAR_RGB,
    TOPIC_CAMERA_REAR_RIGHT_RGB,
    TOPIC_IMU,
    TOPIC_LIDAR,
    TOPIC_POSE,
    unpack_image,
    unpack_imu,
    unpack_point_cloud,
    unpack_pose,
)

_IMAGE_TOPICS = {
    TOPIC_CAMERA_FRONT_RGB: ("/front_camera/rgb", "front_camera"),
    TOPIC_CAMERA_FRONT_DEPTH: ("/front_camera/depth", "front_camera"),
    TOPIC_CAMERA_REAR_RGB: ("/rear_camera/rgb", "rear_camera"),
    TOPIC_CAMERA_REAR_DEPTH: ("/rear_camera/depth", "rear_camera"),
    # RIGHT-eye cameras: only populated on CAMERA_STEREO_SUITE (multicam_slam's
    # rig) -- harmless no-op publishers otherwise, since nothing ever sends on
    # these zmq topics unless sim_ground_truth_loop.py finds front_camera_right/
    # rear_camera_right in scene.sensors. See sim_ipc.py's topic comment.
    TOPIC_CAMERA_FRONT_RIGHT_RGB: ("/front_camera_right/rgb", "front_camera_right"),
    TOPIC_CAMERA_REAR_RIGHT_RGB: ("/rear_camera_right/rgb", "rear_camera_right"),
}


class Ros2BridgeRelay(Node):
    def __init__(self) -> None:
        super().__init__("sim_ros2_bridge_relay")
        self._gt_pub = GroundTruthPublisher()
        self._lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", 10)
        self._imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self._image_pubs = {topic: self.create_publisher(Image, ros_topic, 10) for topic, (ros_topic, _) in _IMAGE_TOPICS.items()}

    def handle_pose(self, raw: bytes) -> None:
        payload = unpack_pose(raw)
        self._gt_pub.publish_pose(
            Time(nanoseconds=payload.sim_time_ns),
            payload.position,
            payload.orientation_wxyz,
            payload.linear_velocity,
            payload.angular_velocity,
        )

    def handle_lidar(self, raw: bytes) -> None:
        payload = unpack_point_cloud(raw)
        self._lidar_pub.publish(build_point_cloud_msg(payload, frame_id="lidar"))

    def handle_imu(self, raw: bytes) -> None:
        payload = unpack_imu(raw)
        self._imu_pub.publish(build_imu_msg(payload, frame_id="imu"))

    def handle_image(self, topic: bytes, raw: bytes) -> None:
        _, frame_id = _IMAGE_TOPICS[topic]
        payload = unpack_image(raw)
        self._image_pubs[topic].publish(build_image_msg(payload, frame_id=frame_id))


def run() -> None:
    rclpy.init()
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(IPC_ADDRESS)
    for topic in (TOPIC_POSE, TOPIC_LIDAR, TOPIC_IMU, *_IMAGE_TOPICS):
        socket.setsockopt(zmq.SUBSCRIBE, topic)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    relay = Ros2BridgeRelay()
    try:
        # Publishing via rclpy's create_publisher().publish() sends
        # immediately -- no rclpy.spin() needed, this node has no
        # subscriptions/timers of its own. Poll with a timeout (rather than
        # a blocking recv_multipart()) purely so `while rclpy.ok()` actually
        # gets re-checked between messages, for a clean shutdown.
        while rclpy.ok():
            if not poller.poll(timeout=500):
                continue
            topic, raw = socket.recv_multipart()
            if topic == TOPIC_POSE:
                relay.handle_pose(raw)
            elif topic == TOPIC_LIDAR:
                relay.handle_lidar(raw)
            elif topic == TOPIC_IMU:
                relay.handle_imu(raw)
            elif topic in _IMAGE_TOPICS:
                relay.handle_image(topic, raw)
    finally:
        relay.destroy_node()
        rclpy.shutdown()
        socket.close()
        context.term()


if __name__ == "__main__":
    run()
