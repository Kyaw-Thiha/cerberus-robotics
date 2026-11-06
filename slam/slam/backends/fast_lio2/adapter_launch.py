"""Thin adapter for FAST-LIO2 -- see slam/slam/backends/common_adapter.py for
the shared remap/TF/tagging logic.

third_party/slam/fast_lio2 (spark_fast_lio) is a NATIVE ROS2 port, not a
ROS1 node behind ros1_bridge -- confirmed by reading
spark_fast_lio.cpp's own create_publisher() calls: pub_odom_ publishes
nav_msgs/Odometry on the relative topic "odometry" (-> /odometry at the
node's root namespace), lowercase, not the ROS1-convention "/Odometry"
this file originally assumed before that assumption was checked against
the real vendored source. pub_cloud_full_ on "cloud_registered" does
happen to match ROS1 spark_fast_lio's own convention, unlike odometry."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(node_name="fast_lio2_adapter", native_odom_topic="/odometry", native_map_topic="/cloud_registered")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
