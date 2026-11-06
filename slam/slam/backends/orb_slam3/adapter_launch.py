"""Thin adapter for ORB-SLAM3 -- see slam/slam/backends/common_adapter.py
for the shared remap/TF/tagging logic. The old Examples_old/ROS (ROS1,
ros1_bridge) wrapper never published pose/map to a topic at all -- see
git history for that dead end. It's since been superseded by a real,
native ROS2 wrapper vendored at third_party/slam/orb_slam3/ros2_wrapper/,
which DOES publish real odometry: orb_slam3_node.cpp's constructor calls
`odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/orb_slam3/odom", 10)`
(confirmed by reading the source directly, not guessed). No map/point-cloud
publisher exists anywhere in that wrapper -- only the one odometry
publisher -- so native_map_topic is intentionally None here; see
common_adapter.py's docstring for why that's safe (skips the map
subscription/republish rather than waiting on a topic that never
publishes)."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="orb_slam3_adapter",
        native_odom_topic="/orb_slam3/odom",  # confirmed: ros2_wrapper/src/orb_slam3_node.cpp's create_publisher<nav_msgs::msg::Odometry> call
        native_map_topic=None,  # no map/point-cloud publisher exists in ros2_wrapper -- odom-only
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
