import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from slam.slam.backends.common_adapter import SlamBackendAdapter


def _make_odom(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = "some_native_frame"  # deliberately NOT "map" -- adapter must overwrite this
    msg.pose.pose = Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))
    return msg


def test_on_odom_republishes_with_contract_frame_ids_and_broadcasts_tf():
    rclpy.init()
    try:
        node = SlamBackendAdapter(node_name="test_adapter", native_odom_topic="/native/odom", native_map_topic="/native/map")
        try:
            published = []
            node._odom_pub.publish = lambda msg: published.append(msg)  # capture instead of a live subscriber
            node._on_odom(_make_odom(1.0, 2.0, 0.3, 0.0, 0.0, 0.0, 1.0))
            assert len(published) == 1
            assert published[0].header.frame_id == "map"
            assert published[0].child_frame_id == "base_link"
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_on_map_wraps_native_point_cloud_with_point_cloud_map_type_tag():
    rclpy.init()
    try:
        node = SlamBackendAdapter(node_name="test_adapter2", native_odom_topic="/native/odom", native_map_topic="/native/map")
        try:
            published = []
            node._map_pub.publish = lambda msg: published.append(msg)
            native_cloud = PointCloud2()
            node._on_map(native_cloud)
            assert len(published) == 1
            assert published[0].map_type == "point_cloud"
            assert published[0].point_cloud is native_cloud
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
