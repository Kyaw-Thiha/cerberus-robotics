"""Concrete Isaac Lab sensor configs and the two locked `SensorSuiteCfg`
rigs (`LIDAR_CAMERA_SUITE`, `CAMERA_ONLY_SUITE`) from
the sensor-setup design. Imports `isaaclab` -- not installed
locally (see `docs/isaac_lab_workflow.md`), so this module is only
import-time-verifiable via `py_compile` on this machine; functional
verification (does the LiDAR actually raycast, does the stereo pair render)
happens on a pod, same constraint `core/platforms/go2/platform.py` already
lives with.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ImuCfg, RayCasterCfg, patterns

from common.cerberus_compat.quat import to_isaac
from common.cerberus_description.sensors.sensor_suite import (
    SensorSuiteCfg,
    register_suite,
)

# Camera/sensor offset `rot` quaternion order. Verified at Lab v3.0.0-EA:
# isaaclab/sensors/camera/camera_cfg.py documents `OffsetCfg.rot` as
# "Quaternion rotation (x, y, z, w)" with default (0, 0, 0, 1); on Lab 2.x it
# is (w, x, y, z). The literals below are written Cerberus-internal w-first
# (readable) and routed through `to_isaac`, which reorders to the installed
# Lab's native order (override with CERBERUS_QUAT_CONVENTION=wxyz|xyzw).
# `convention="ros"` is the camera-AXIS convention only, independent of order.
# Rotation: ROS optical frame (+Z fwd, -Y up) mounted on a +X-forward body.
_ROS_FRONT_ROT_WXYZ = (0.5, -0.5, 0.5, -0.5)  # facing +X
_ROS_REAR_ROT_WXYZ = (-0.5, 0.5, 0.5, -0.5)  # facing -X (180 deg yaw flip)
_ROS_FRONT_ROT = to_isaac(_ROS_FRONT_ROT_WXYZ)
_ROS_REAR_ROT = to_isaac(_ROS_REAR_ROT_WXYZ)

# 10 Hz matches Livox Mid-360 real hardware rate. Asymmetric vertical FoV
# biased downward for near-field/underbody coverage -- see perception.md's
# Sensor Modalities table for why spinning LiDAR was chosen over solid-state.
LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    update_period=1 / 10,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    # Single-mesh RayCaster against the stock flat ground. A GRScenes house is multi-mesh:
    # slam/slam/eval/scene_env_cfg.py's apply_scene() swaps this for a MultiMeshRayCasterCfg.
    mesh_prim_paths=["/World/ground"],
    # Lab 3.0 replaced the removed `attach_yaw_only=True` flag with this (verified in
    # isaaclab/sensors/ray_caster/ray_caster_cfg.py and kernels.py: "yaw" = sensor position plus
    # yaw-only rotation, no roll/pitch tilt of the scan).
    ray_alignment="yaw",
    pattern_cfg=patterns.LidarPatternCfg(
        channels=100,
        vertical_fov_range=[-25, 15],
        horizontal_fov_range=[-180, 180],
        horizontal_res=1.0,
    ),
    debug_vis=False,
)

# Mounted at the robot base, same rigid body the LiDAR/pose ground-truth
# already reads (root_pos_w/root_quat_w in pose_export.py) -- no separate
# extrinsic to track between the IMU and the frame FAST-LIO2/GS-LIVO/etc.
# fuse it against. 200Hz matches typical low-cost MEMS IMU hardware (well
# above the LiDAR's 10Hz and camera's 30Hz, since LIO/LIVO backends expect
# IMU as their high-rate propagation source between slower sensor updates).
# NOT independently verified against a live Isaac Lab install -- same
# pod-side-only caveat sim_ground_truth_loop.py's header already carries for
# root_lin_vel_w/root_ang_vel_w; confirm ImuCfg's constructor args and the
# resulting Imu sensor's `.data.lin_acc_b`/`.data.ang_vel_b` field names on a
# real pod before depending on them. Verified at Lab v3.0.0-EA
# (isaaclab_physx/sensors/imu/imu_data.py): ang_vel_b/lin_acc_b are ProxyArray
# and `quat_w` no longer exists on ImuData (read robot.data.root_quat_w instead).
IMU_CFG = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    update_period=1 / 200,
    debug_vis=False,
)

# ~12cm stereo baseline (half-offset each side, 0.06 - (-0.06) = 0.12m -- see
# stereo_to_depth_node.py's STEREO_BASELINE_M, which must match this), ~20 deg
# downward tilt. Only wire two cameras (vs.
# one privileged-depth CameraCfg) if the SLAM candidate actually implemented
# wants raw stereo pairs.
#
# LEFT keeps data_types=["rgb", "depth"]: the "depth" channel here is Isaac
# Lab's privileged sim ground-truth depth, which docs/perception.md's stereo-
# vs-RGB-D decision (2026 quadruped SLAM benchmark: stereo beats RGB-D on
# legged platforms, global shutter avoids the motion-blur rolling-shutter
# RGB-D showed under locomotion) says multicam_slam must NOT consume -- it
# gets real depth computed from the stereo pair instead (see
# slam/slam/backends/multicam_slam/stereo_to_depth_node.py). LEFT's privileged
# depth is kept anyway, unused by the SLAM pipeline itself, purely as a
# debug/eval baseline: a free ground-truth reference slam/slam/eval tooling
# can diff the stereo-computed depth against to quantify stereo-matching
# error, without needing a second sim run. RIGHT drops "depth" -- it only
# ever feeds the stereo matcher as the second image in the pair, so paying
# render cost for a depth buffer nothing reads would be pure waste.
_FRONT_CAM_SPAWN = sim_utils.PinholeCameraCfg(
    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
)
FRONT_CAMERA_LEFT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam_left",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(0.32, -0.06, 0.05), rot=_ROS_FRONT_ROT, convention="ros"),
)
FRONT_CAMERA_RIGHT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam_right",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(0.32, 0.06, 0.05), rot=_ROS_FRONT_ROT, convention="ros"),
)

# Mirrored mount, 180 deg yaw flip. Camera-only ablation rig only -- never
# instantiated for LIDAR_CAMERA_SUITE, even inactive, so VRAM budgeting
# between rigs stays directly comparable. Same
# LEFT-keeps-depth-as-debug-baseline / RIGHT-rgb-only split as the front pair.
REAR_CAMERA_LEFT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/rear_cam_left",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(-0.32, -0.06, 0.05), rot=_ROS_REAR_ROT, convention="ros"),
)
REAR_CAMERA_RIGHT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/rear_cam_right",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(-0.32, 0.06, 0.05), rot=_ROS_REAR_ROT, convention="ros"),
)

# Phase 2 placeholder -- do not implement until that phase's own research
# pass. Field exists on SensorSuiteCfg now so Phase 2 doesn't need to
# restructure it.
WRIST_CAMERA_CFG = None

LIDAR_CAMERA_SUITE = SensorSuiteCfg(
    name="lidar_camera",
    lidar=LIDAR_CFG,
    front_camera=FRONT_CAMERA_LEFT_CFG,
    rear_camera=None,
    wrist_camera=WRIST_CAMERA_CFG,
    imu=IMU_CFG,
)
register_suite(LIDAR_CAMERA_SUITE)

CAMERA_ONLY_SUITE = SensorSuiteCfg(
    name="camera_only",
    lidar=None,
    front_camera=FRONT_CAMERA_LEFT_CFG,
    rear_camera=REAR_CAMERA_LEFT_CFG,
    wrist_camera=WRIST_CAMERA_CFG,
)
register_suite(CAMERA_ONLY_SUITE)

# Genuine front+rear STEREO-pair rig, for multicam_slam specifically. Distinct
# from CAMERA_ONLY_SUITE (orb_slam3's rig, single LEFT-only camera per
# position) rather than widening it in place, so orb_slam3/fast_lio2/
# fast_livo2/bievr_lio stay byte-for-byte unchanged -- multicam_slam is the
# only registered candidate whose declared sensor_requirements
# ({"front_camera": "required", "rear_camera": "required"}, see
# slam/slam/core/algorithms.py) actually needs the second eye per position to
# compute its own depth (stereo-not-RGB-D decision). At the
# coarse active_sensors()/sensor_requirements level this suite is
# indistinguishable from CAMERA_ONLY_SUITE (front_camera/rear_camera both
# True) -- validate_sensor_suite.py's compatibility check doesn't need to
# change; front_camera_right/rear_camera_right are extra wiring on top, not a
# new requirement level.
CAMERA_STEREO_SUITE = SensorSuiteCfg(
    name="camera_stereo",
    lidar=None,
    front_camera=FRONT_CAMERA_LEFT_CFG,
    front_camera_right=FRONT_CAMERA_RIGHT_CFG,
    rear_camera=REAR_CAMERA_LEFT_CFG,
    rear_camera_right=REAR_CAMERA_RIGHT_CFG,
    wrist_camera=WRIST_CAMERA_CFG,
)
register_suite(CAMERA_STEREO_SUITE)

# Static offset only -- see slam/slam/eval/showcase_camera_follow.py for the
# actual soft-follow pose math (§8: position tracks the robot, orientation
# stays level). Wiring compute_chase_cam_pose()'s output into a per-physics-
# step prim-transform update on THIS camera is a pod-side Isaac Lab
# integration step, out of scope here (needs a live sim to verify against).
SHOWCASE_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ShowcaseCamera",  # NOT parented under Robot/base -- a rigid
    # parent-child offset would inherit body roll/pitch, contradicting §8's
    # "orientation stays level" requirement. This prim's transform must be
    # set explicitly every step from compute_chase_cam_pose()'s output
    # instead of relying on a fixed offset.
    update_period=1 / 30,
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
    ),
)
