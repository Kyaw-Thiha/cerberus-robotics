"""Thin adapter for FAST-LIVO2 -- see slam/slam/backends/common_adapter.py
for the shared remap/TF/tagging logic. third_party/slam/fast_livo2 is a
NATIVE ROS2 package (ament_cmake + rclcpp, confirmed via its CMakeLists.txt/
package.xml) -- no ros1_bridge involved, same as fast_lio2 turned out to be
(that package's own adapter carried this identical wrong assumption before
being fixed; this file's own topic names were already independently
confirmed correct, only this docstring's framing was stale). Topic names
below are CONFIRMED against third_party/slam/fast_livo2/src/LIVMapper.cpp:
pubOdomAftMapped advertises nav_msgs::Odometry on "/aft_mapped_to_init" (line
205); pubLaserCloudFullRes advertises sensor_msgs::PointCloud2 on
"/cloud_registered" (line 200)."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="fast_livo2_adapter",
        native_odom_topic="/aft_mapped_to_init",  # confirmed: LIVMapper.cpp:205
        native_map_topic="/cloud_registered",  # confirmed: LIVMapper.cpp:200
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
