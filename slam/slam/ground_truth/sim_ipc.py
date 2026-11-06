"""Wire format for the local zmq IPC link between the Isaac-Sim-interpreter
process (sim_ground_truth_loop.py, which owns the env step loop and reads
robot/sensor state) and the rclpy-side process (ros2_bridge_relay.py, which
republishes to ROS2). The two processes can't share one Python interpreter
-- rclpy is only installed into pixi's sim-ros2 env, a different Python
entirely from Isaac Sim's bundled interpreter (docker/remote/Dockerfile's
/workspace/isaaclab/_isaac_sim/python.sh) -- so this module is the only
thing both sides need to agree on, and deliberately imports neither
`isaaclab` nor `rclpy` so it unit-tests locally under the default pixi env.

Transport: one zmq PUB socket (sim side) / SUB socket (relay side) over
localhost, topic-prefixed multipart messages -- `socket.send_multipart([topic,
packed_payload])`. Topics below are the fixed, closed set this bridge
carries; both processes hardcode them, no discovery needed since it's a
fixed one-sim-one-relay pair on one pod.
"""

from __future__ import annotations

from dataclasses import dataclass

import msgpack

IPC_ADDRESS = "tcp://127.0.0.1:5556"

TOPIC_POSE = b"pose"
TOPIC_LIDAR = b"lidar"
TOPIC_IMU = b"imu"
TOPIC_CAMERA_FRONT_RGB = b"camera/front/rgb"
TOPIC_CAMERA_FRONT_DEPTH = b"camera/front/depth"
TOPIC_CAMERA_REAR_RGB = b"camera/rear/rgb"
TOPIC_CAMERA_REAR_DEPTH = b"camera/rear/depth"
# RIGHT-eye cameras: rgb8 only, never depth -- CAMERA_STEREO_SUITE's
# front_camera_right/rear_camera_right (sensor_configs.py) are
# data_types=["rgb"] only, since they only ever feed
# multicam_slam/stereo_to_depth_node.py's own real stereo block matching as
# the second image in the pair, not Isaac Lab's privileged depth channel.
TOPIC_CAMERA_FRONT_RIGHT_RGB = b"camera/front_right/rgb"
TOPIC_CAMERA_REAR_RIGHT_RGB = b"camera/rear_right/rgb"


@dataclass(frozen=True)
class PosePayload:
    sim_time_ns: int
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]


def pack_pose(payload: PosePayload) -> bytes:
    return msgpack.packb(
        {
            "sim_time_ns": payload.sim_time_ns,
            "position": list(payload.position),
            "orientation_wxyz": list(payload.orientation_wxyz),
            "linear_velocity": list(payload.linear_velocity),
            "angular_velocity": list(payload.angular_velocity),
        },
        use_bin_type=True,
    )


def unpack_pose(raw: bytes) -> PosePayload:
    d = msgpack.unpackb(raw, raw=False)
    return PosePayload(
        sim_time_ns=d["sim_time_ns"],
        position=tuple(d["position"]),
        orientation_wxyz=tuple(d["orientation_wxyz"]),
        linear_velocity=tuple(d["linear_velocity"]),
        angular_velocity=tuple(d["angular_velocity"]),
    )


@dataclass(frozen=True)
class ImagePayload:
    sim_time_ns: int
    height: int
    width: int
    encoding: str  # "rgb8" or "32FC1", matching sensor_msgs/Image's `encoding` field
    data: bytes  # already row-major, contiguous, matching `encoding`'s byte layout


def pack_image(payload: ImagePayload) -> bytes:
    return msgpack.packb(
        {
            "sim_time_ns": payload.sim_time_ns,
            "height": payload.height,
            "width": payload.width,
            "encoding": payload.encoding,
            "data": payload.data,
        },
        use_bin_type=True,
    )


def unpack_image(raw: bytes) -> ImagePayload:
    d = msgpack.unpackb(raw, raw=False)
    return ImagePayload(
        sim_time_ns=d["sim_time_ns"],
        height=d["height"],
        width=d["width"],
        encoding=d["encoding"],
        data=d["data"],
    )


@dataclass(frozen=True)
class ImuPayload:
    sim_time_ns: int
    orientation_wxyz: tuple[float, float, float, float]  # from the same
    # articulation-body orientation the IMU is rigidly mounted to (Isaac
    # Lab's Imu sensor exposes `.data.quat_w` for this) -- NOT independently
    # verified against a live Isaac Lab install, see sensor_configs.py's
    # IMU_CFG comment.
    angular_velocity: tuple[float, float, float]  # body frame, rad/s
    linear_acceleration: tuple[float, float, float]  # body frame, m/s^2, gravity included (matches
    # sensor_msgs/Imu's own convention: a stationary IMU reads ~9.81 on its up axis)


def pack_imu(payload: ImuPayload) -> bytes:
    return msgpack.packb(
        {
            "sim_time_ns": payload.sim_time_ns,
            "orientation_wxyz": list(payload.orientation_wxyz),
            "angular_velocity": list(payload.angular_velocity),
            "linear_acceleration": list(payload.linear_acceleration),
        },
        use_bin_type=True,
    )


def unpack_imu(raw: bytes) -> ImuPayload:
    d = msgpack.unpackb(raw, raw=False)
    return ImuPayload(
        sim_time_ns=d["sim_time_ns"],
        orientation_wxyz=tuple(d["orientation_wxyz"]),
        angular_velocity=tuple(d["angular_velocity"]),
        linear_acceleration=tuple(d["linear_acceleration"]),
    )


@dataclass(frozen=True)
class PointCloudPayload:
    sim_time_ns: int
    points_xyz: bytes  # float32 little-endian, row-major (N, 3), contiguous
    ring: bytes  # uint16 little-endian, N contiguous -- per-point scan-line index,
    # 0..channels-1. Required by every FAST-LIO2-family backend's velodyne/ouster
    # point-cloud parser (pcl::fromROSMsg into a struct with a `ring` field) to
    # bucket points into per-layer motion-compensation groups -- confirmed by
    # reading third_party/slam/fast_lio2/spark_fast_lio/include/preprocess.h's
    # POINT_CLOUD_REGISTER_POINT_STRUCT for velodyne_ros::Point/ouster_ros::Point,
    # not guessed. Computed from LidarPatternCfg's known channel-major point
    # ordering (ring = flat_index // azimuth_count) -- confirmed against a real
    # pod's isaaclab.sensors.ray_caster.patterns.patterns.lidar_pattern() source:
    # torch.meshgrid(vertical_angles, horizontal_angles, indexing="ij").reshape(-1, 3)
    # varies the horizontal (azimuth) axis fastest.


def pack_point_cloud(payload: PointCloudPayload) -> bytes:
    return msgpack.packb(
        {"sim_time_ns": payload.sim_time_ns, "points_xyz": payload.points_xyz, "ring": payload.ring},
        use_bin_type=True,
    )


def unpack_point_cloud(raw: bytes) -> PointCloudPayload:
    d = msgpack.unpackb(raw, raw=False)
    return PointCloudPayload(sim_time_ns=d["sim_time_ns"], points_xyz=d["points_xyz"], ring=d["ring"])
