"""Generic thin SLAM backend adapter -- subscribes to a backend's native
odometry and point-cloud-map topics and republishes them onto this repo's
SLAM topic contract: /slam/odom (nav_msgs/Odometry,
frame ids rewritten to map/base_link), /slam/map (cerberus_msgs/TaggedMap,
map_type carried IN the payload, not a node parameter -- see Task 6a's
rationale), and a TF broadcast map -> base_link derived from the same
odometry pose. Every backend's own adapter_launch.py under
slam/slam/backends/<name>/ is just this class instantiated with that
backend's native topic names -- per §2, "the backend adapter is thin...
does NOT encode backend-specific logic beyond remapping."

native_map_topic is Optional: orb_slam3 and multicam_slam's ros2_wrapper
nodes only publish odometry, no map/point-cloud topic exists in either fork
-- confirmed by reading their ros2_wrapper source directly (only one
create_publisher<nav_msgs::msg::Odometry> call each). Passing None skips
the map subscription/republish entirely instead of subscribing to a topic
that will never publish anything.
"""

from __future__ import annotations

from cerberus_msgs.msg import TaggedMap
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster


class SlamBackendAdapter(Node):
    def __init__(self, node_name: str, native_odom_topic: str, native_map_topic: str | None) -> None:
        super().__init__(node_name)
        self._odom_pub = self.create_publisher(Odometry, "/slam/odom", 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, native_odom_topic, self._on_odom, 10)
        if native_map_topic is not None:
            self._map_pub = self.create_publisher(TaggedMap, "/slam/map", 10)
            self.create_subscription(PointCloud2, native_map_topic, self._on_map, 10)

    def _on_odom(self, msg: Odometry) -> None:
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        self._odom_pub.publish(msg)

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(transform)

    def _on_map(self, msg: PointCloud2) -> None:
        tagged = TaggedMap()
        tagged.map_type = "point_cloud"
        tagged.point_cloud = msg
        self._map_pub.publish(tagged)
