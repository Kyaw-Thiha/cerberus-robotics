"""Thin adapter for Multicam-SLAM -- see slam/slam/backends/common_adapter.py
for the shared remap/TF/tagging logic. The old Examples/ROS/ORB_SLAM2 (ROS1,
ros1_bridge) wrapper never published pose/map to a topic at all -- see git
history for that dead end. It's since been superseded by a real, native
ROS2 wrapper vendored at third_party/slam/multicam_slam/ros2_wrapper/,
which DOES publish real odometry: multicam_slam_node.cpp's constructor
calls `odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/multicam_slam/odom", 10)`
(confirmed by reading the source directly, not guessed). No map/point-cloud
publisher exists anywhere in that wrapper -- only the one odometry
publisher -- so native_map_topic is intentionally None here; see
common_adapter.py's docstring for why that's safe (skips the map
subscription/republish rather than waiting on a topic that never
publishes).

This adapter only handles OUTPUT (pose/map remap). For the algorithm's
INPUT side -- it expects 4 synced sensor_msgs/Image topics (front rgb,
front depth, rear rgb, rear depth), depth as 16-bit millimeters
(16UC1, DepthMapFactor ~1000.0) -- see stereo_to_depth_node.py and
stereo_to_depth_launch.py in this same directory. Cerberus feeds it REAL
stereo-computed depth (OpenCV StereoSGBM over the front/rear stereo pairs
from common/cerberus_description/sensors/sensor_configs.py's
CAMERA_STEREO_SUITE), never Isaac Lab's privileged sim ground-truth depth --
see docs/perception.md's stereo-vs-RGB-D decision. That depth computation is
real per-frame logic, not a topic remap, so it deliberately lives in a
separate node rather than here (see common_adapter.py's docstring)."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="multicam_slam_adapter",
        native_odom_topic="/multicam_slam/odom",  # confirmed: ros2_wrapper/src/multicam_slam_node.cpp's create_publisher<nav_msgs::msg::Odometry> call
        native_map_topic=None,  # no map/point-cloud publisher exists in ros2_wrapper -- odom-only
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
