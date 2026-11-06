from __future__ import annotations

from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformBroadcaster

from slam.slam.ground_truth.pose_export import (
    build_ground_truth_transform,
    build_odometry_fields,
)


class GroundTruthPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ground_truth_publisher")
        self._odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

    def publish_pose(self, sim_time: Time, position, isaac_quat, linear_velocity, angular_velocity) -> None:
        odom_fields = build_odometry_fields(position, isaac_quat, linear_velocity, angular_velocity)
        odom_msg = Odometry()
        odom_msg.header.stamp = sim_time.to_msg()
        odom_msg.header.frame_id = odom_fields.frame_id
        odom_msg.child_frame_id = odom_fields.child_frame_id
        odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z = odom_fields.position
        odom_msg.pose.pose.orientation = Quaternion(**dict(zip("xyzw", odom_fields.orientation_xyzw)))
        odom_msg.twist.twist.linear = Vector3(**dict(zip("xyz", odom_fields.linear_velocity)))
        odom_msg.twist.twist.angular = Vector3(**dict(zip("xyz", odom_fields.angular_velocity)))
        self._odom_pub.publish(odom_msg)

        tf_fields = build_ground_truth_transform(position, isaac_quat)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = sim_time.to_msg()
        tf_msg.header.frame_id = tf_fields.frame_id
        tf_msg.child_frame_id = tf_fields.child_frame_id
        tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = tf_fields.translation
        tf_msg.transform.rotation = Quaternion(**dict(zip("xyzw", tf_fields.rotation_xyzw)))
        self._tf_broadcaster.sendTransform(tf_msg)


def main() -> None:
    import rclpy
    rclpy.init()
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
