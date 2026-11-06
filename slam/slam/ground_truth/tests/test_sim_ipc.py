from slam.slam.ground_truth.sim_ipc import (
    ImagePayload,
    ImuPayload,
    PointCloudPayload,
    PosePayload,
    pack_image,
    pack_imu,
    pack_point_cloud,
    pack_pose,
    unpack_image,
    unpack_imu,
    unpack_point_cloud,
    unpack_pose,
)


def test_pose_round_trips_through_pack_unpack():
    payload = PosePayload(
        sim_time_ns=123_456_789,
        position=(1.0, 2.0, 0.3),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.1, 0.2, 0.0),
        angular_velocity=(0.0, 0.0, 0.05),
    )
    assert unpack_pose(pack_pose(payload)) == payload


def test_image_round_trips_through_pack_unpack_including_raw_bytes():
    payload = ImagePayload(sim_time_ns=42, height=2, width=2, encoding="rgb8", data=bytes(range(12)))
    assert unpack_image(pack_image(payload)) == payload


def test_imu_round_trips_through_pack_unpack():
    payload = ImuPayload(
        sim_time_ns=99,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.01, 0.02, 0.03),
        linear_acceleration=(0.0, 0.0, 9.81),
    )
    assert unpack_imu(pack_imu(payload)) == payload


def test_point_cloud_round_trips_through_pack_unpack():
    payload = PointCloudPayload(sim_time_ns=7, points_xyz=bytes(range(24)), ring=bytes([0, 0, 1, 0]))
    assert unpack_point_cloud(pack_point_cloud(payload)) == payload
