"""Builds sensor_msgs/Image and sensor_msgs/PointCloud2 messages directly
from the raw payloads carried over sim_ipc.py's zmq link -- no cv_bridge
(see pixi.toml's sim-ros2 feature comment for why), just the message
fields' own documented byte-layout contract. Kept separate from
ros2_bridge_relay.py so the field-construction logic is unit-testable
without a live zmq socket or a running rclpy node.

Every builder sets `header.stamp` from `payload.sim_time_ns` -- every
message published without it defaulted to (0, 0), making every scan/frame
look simultaneous. FAST-LIO2's own scan-ordering sanity check
("Lidar loopback detected, clearing buffers") caught this directly on a
real pod run before it was fixed here.
"""

from __future__ import annotations

import numpy as np
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField

from slam.slam.ground_truth.pose_export import isaac_quat_to_ros
from slam.slam.ground_truth.sim_ipc import ImagePayload, ImuPayload, PointCloudPayload

_BYTES_PER_PIXEL = {"rgb8": 3, "32FC1": 4}

# /lidar/points' one wire layout matches velodyne_ros::Point (the struct
# FAST-LIO2-family backends' pcl::fromROSMsg parses into -- see
# third_party/slam/fast_lio2/spark_fast_lio/include/preprocess.h's
# POINT_CLOUD_REGISTER_POINT_STRUCT and sim_ipc.py's PointCloudPayload.ring
# comment) byte-for-byte, NOT guessed: offsets/point_step below were read
# off a real `offsetof`/`sizeof` probe compiled against that exact vendored
# header on a pod (PCL_ADD_POINT4D + EIGEN_ALIGN16 introduce padding a
# field-list alone wouldn't reveal). x/y/z (bytes 0-11) are also a valid
# plain-XYZ cloud on their own, so any consumer that only wants those three
# fields reads the same offsets it always did.
#   sizeof(Point)=32, x@0 y@4 z@8 intensity@16 time@20 ring@24 (uint16)
# intensity is a constant placeholder (no real sim equivalent). `time` was
# originally left at a constant 0.0, reasoning that a last-point time of 0
# would make FAST-LIO2 fall into its own given_offset_time=false path and
# derive per-point timing itself -- WRONG, confirmed on a real pod run:
# that fallback derivation only exists inside preprocess.cpp's
# `feature_enabled` branch of velodyne_handler(); the plain/default branch
# this repo actually runs (feature_enabled defaults false) just does
# `curvature = points[i].time * time_unit_scale` directly, so an all-zero
# `time` field produced a real bug -- every point got curvature 0, tripping
# FAST-LIO2's own "should be close to 100" sanity warning and "No point,
# skip this scan!" on nearly every scan. Fixed by synthesizing a real
# per-point time spread linearly across one scan period (see
# _SCAN_PERIOD_S) -- Isaac Lab's raycast is physically instantaneous (no
# genuine sub-scan sweep to time honestly), so this is an approximation
# for FAST-LIO2's own motion-deskewing math to have SOMETHING plausible to
# work with, not a real hardware timestamp; good enough since it only
# needs to span the right total duration, not match true per-point firing
# order. Bytes 12-15 and 26-31 are unregistered padding, left zero.
#
# Unit is MICROSECONDS. This was briefly changed to seconds mid-session to
# satisfy BIEVR-LIO (whose point-cloud parser hardcodes a field literally
# named "time" as seconds, no per-backend override) -- reverted after
# checking the other two lidar-consuming backends in the same evaluation
# pass: FAST-LIVO2 and GS-LIVO (which shares FAST-LIVO2's preprocess.cpp
# lineage) BOTH hardcode `time / 1000.0` expecting MICROSECONDS-in
# (dividing by 1000 to land on their internal milliseconds convention), no
# config knob either. Three non-FAST-LIO2 backends, two want microseconds,
# one wants seconds -- microseconds is the majority-correct shared
# convention (also satisfies FAST-LIO2 via its own configurable
# `preprocess.timestamp_unit` knob, set to 2 in
# slam/slam/backends/fast_lio2/sim_config.yaml). BIEVR-LIO is the one
# flagged exception -- see slam/slam/backends/bievr_lio/sim_config.yaml's
# own comment for its unresolved status; it needs its own fix (a source
# patch adding a configurable scale, or a small per-backend unit-rescaling
# relay node), not a change to this shared topic's convention.
_POINT_STEP = 32
# microseconds -- see the unit-convention comment above. Matches
# common/cerberus_description/sensors/sensor_configs.py's LIDAR_CFG
# update_period=1/10 (10 Hz -> 100ms -> 100,000us per scan).
_SCAN_PERIOD_US = 100_000.0
_VELODYNE_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "time", "ring"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8, 16, 20, 24],
        "itemsize": _POINT_STEP,
    }
)


def build_image_msg(payload: ImagePayload, frame_id: str) -> Image:
    bytes_per_pixel = _BYTES_PER_PIXEL[payload.encoding]
    msg = Image()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(nanoseconds=payload.sim_time_ns).to_msg()
    msg.height = payload.height
    msg.width = payload.width
    msg.encoding = payload.encoding
    msg.is_bigendian = 0
    msg.step = payload.width * bytes_per_pixel
    msg.data = payload.data
    return msg


def build_imu_msg(payload: ImuPayload, frame_id: str) -> Imu:
    msg = Imu()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(nanoseconds=payload.sim_time_ns).to_msg()
    w, x, y, z = payload.orientation_wxyz
    qx, qy, qz, qw = isaac_quat_to_ros(w=w, x=x, y=y, z=z)
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = qx, qy, qz, qw
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = payload.angular_velocity
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = payload.linear_acceleration
    # Sim ground truth -- zero covariance means "unknown", per sensor_msgs/Imu's
    # documented convention, not "exactly zero noise". Leaving all three
    # covariance arrays at their default all-zero fill matches that.
    return msg


def build_point_cloud_msg(payload: PointCloudPayload, frame_id: str) -> PointCloud2:
    xyz = np.frombuffer(payload.points_xyz, dtype="<f4").reshape(-1, 3)
    ring = np.frombuffer(payload.ring, dtype="<u2")
    num_points = xyz.shape[0]

    buf = np.zeros(num_points, dtype=_VELODYNE_DTYPE)
    buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    buf["ring"] = ring
    buf["time"] = np.linspace(0.0, _SCAN_PERIOD_US, num_points, dtype="<f4", endpoint=True)
    # intensity left at its np.zeros() default -- see _POINT_STEP's comment.

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = Time(nanoseconds=payload.sim_time_ns).to_msg()
    msg.height = 1
    msg.width = num_points
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
        PointField(name="ring", offset=24, datatype=PointField.UINT16, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = _POINT_STEP
    msg.row_step = _POINT_STEP * num_points
    msg.data = buf.tobytes()
    msg.is_dense = True
    return msg
