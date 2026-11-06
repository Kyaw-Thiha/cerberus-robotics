from slam.slam.ground_truth.sensor_msg_builders import (
    build_image_msg,
    build_imu_msg,
    build_point_cloud_msg,
)
from slam.slam.ground_truth.sim_ipc import ImagePayload, ImuPayload, PointCloudPayload


def test_build_image_msg_sets_step_from_encoding_and_width():
    payload = ImagePayload(sim_time_ns=1, height=2, width=4, encoding="rgb8", data=bytes(24))
    msg = build_image_msg(payload, frame_id="front_camera")
    assert msg.height == 2
    assert msg.width == 4
    assert msg.encoding == "rgb8"
    assert msg.step == 4 * 3  # width * bytes-per-pixel
    # sensor_msgs/Image's `data` field is a `uint8[]`, which rclpy's message
    # bindings expose as array.array('B', ...), not bytes -- confirmed by an
    # actual AssertionError comparing directly against payload.data (bytes).
    assert bytes(msg.data) == payload.data
    assert msg.header.frame_id == "front_camera"


def test_build_image_msg_depth_encoding_uses_four_bytes_per_pixel():
    payload = ImagePayload(sim_time_ns=1, height=2, width=4, encoding="32FC1", data=bytes(32))
    msg = build_image_msg(payload, frame_id="front_camera")
    assert msg.step == 4 * 4


def test_build_imu_msg_sets_orientation_and_rates_in_ros_xyzw_convention():
    payload = ImuPayload(
        sim_time_ns=1,
        orientation_wxyz=(0.6314712166786194, 0.0013945989776402712, 5.462893022922799e-05, -0.7753980755805969),
        angular_velocity=(0.01, 0.02, 0.03),
        linear_acceleration=(0.0, 0.0, 9.81),
    )
    msg = build_imu_msg(payload, frame_id="imu")
    assert msg.header.frame_id == "imu"
    # isaac_quat_to_ros reorders (w, x, y, z) -> (x, y, z, w)
    assert msg.orientation.w == payload.orientation_wxyz[0]
    assert msg.orientation.x == payload.orientation_wxyz[1]
    assert (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z) == payload.angular_velocity
    assert (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z) == payload.linear_acceleration


def test_build_point_cloud_msg_derives_width_from_byte_length():
    num_points = 5
    points_xyz = bytes(12 * num_points)  # 5 points, 3x float32 each, all zero
    ring = bytes(2 * num_points)  # 5 points, uint16 each, all zero
    payload = PointCloudPayload(sim_time_ns=1, points_xyz=points_xyz, ring=ring)
    msg = build_point_cloud_msg(payload, frame_id="lidar")
    assert msg.height == 1
    assert msg.width == 5
    assert msg.point_step == 32  # velodyne_ros::Point-compatible layout, see sensor_msg_builders.py
    assert msg.row_step == 160
    assert len(msg.fields) == 6
    assert msg.header.frame_id == "lidar"
    assert len(bytes(msg.data)) == 160


def test_build_point_cloud_msg_places_xyz_and_ring_at_the_confirmed_byte_offsets():
    import struct
    points_xyz = struct.pack("<3f", 1.5, -2.5, 3.5)
    ring = struct.pack("<H", 7)
    payload = PointCloudPayload(sim_time_ns=1, points_xyz=points_xyz, ring=ring)
    msg = build_point_cloud_msg(payload, frame_id="lidar")
    data = bytes(msg.data)
    assert struct.unpack_from("<f", data, 0) == (1.5,)
    assert struct.unpack_from("<f", data, 4) == (-2.5,)
    assert struct.unpack_from("<f", data, 8) == (3.5,)
    assert struct.unpack_from("<H", data, 24) == (7,)
