"""Thin adapter for BIEVR-LIO's native ROS2 build (bievr_lio_ros2) -- no
ros1_bridge involved, unlike every other backend here.
Topic names below are CONFIRMED (Task 6d Step 7) against
third_party/slam/bievr_lio's own source: PublisherBase's ns="bievr_lio"
prefixes every topic (bievr_ros_common/publisher_base.h), and
process_topics.cpp:45 / process_bag.cpp:53 both construct
`bievr::Publisher(node, pipeline, "bievr_lio")`. pipeline.cpp:369 calls
`publish(odom, header, "odom", ...)` -> "/bievr_lio/odom" (cross-confirmed
by interfaces/ros2/rviz/config.rviz:104, which hardcodes that same topic for
its Odometry display). There is no dedicated map/voxel-map publisher;
pipeline.cpp:351 publishes the accumulated registered cloud via
`publish(..., "points/registered")` -> "/bievr_lio/points/registered", used
here as the closest native analog to a map topic."""

import rclpy

from slam.slam.backends.common_adapter import SlamBackendAdapter


def main() -> None:
    rclpy.init()
    node = SlamBackendAdapter(
        node_name="bievr_lio_adapter",
        native_odom_topic="/bievr_lio/odom",  # confirmed: pipeline.cpp:369 + publisher_base.h ns prefix
        native_map_topic="/bievr_lio/points/registered",  # confirmed: pipeline.cpp:351 + publisher_base.h ns prefix (no dedicated map topic exists)
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
