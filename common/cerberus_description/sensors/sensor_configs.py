"""Concrete Isaac Lab sensor configs and the two locked `SensorSuiteCfg`
rigs (`LIDAR_CAMERA_SUITE`, `CAMERA_ONLY_SUITE`) from
`plans/sensor_setup_handoff.md` §2-3. Imports `isaaclab` -- not installed
locally (see `docs/isaac_lab_workflow.md`), so this module is only
import-time-verifiable via `py_compile` on this machine; functional
verification (does the LiDAR actually raycast, does the stereo pair render)
happens on a pod, same constraint `core/platforms/go2/platform.py` already
lives with.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns

from common.cerberus_description.sensors.sensor_suite import (
    SensorSuiteCfg,
    register_suite,
)

# 10 Hz matches Livox Mid-360 real hardware rate. Asymmetric vertical FoV
# biased downward for near-field/underbody coverage -- see perception.md's
# Sensor Modalities table for why spinning LiDAR was chosen over solid-state.
LIDAR_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    update_period=1 / 10,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
    mesh_prim_paths=["/World/ground"],  # extend per-scene once Phase 1/2 BEHAVIOR-1K scenes are wired up
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=100,
        vertical_fov_range=[-25, 15],
        horizontal_fov_range=[-180, 180],
        horizontal_res=1.0,
    ),
    debug_vis=False,
)

# ~12cm stereo baseline (half-offset each side), ~20 deg downward tilt. See
# sensor_setup_handoff.md §2.2: only wire two cameras (vs. one privileged-depth
# CameraCfg) if the SLAM candidate actually implemented wants raw stereo pairs.
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
    offset=CameraCfg.OffsetCfg(pos=(0.32, -0.06, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
)
FRONT_CAMERA_RIGHT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_cam_right",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(0.32, 0.06, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
)

# Mirrored mount, 180 deg yaw flip. Camera-only ablation rig only -- never
# instantiated for LIDAR_CAMERA_SUITE, even inactive, so VRAM budgeting
# between rigs stays directly comparable (sensor_setup_handoff.md §2.3).
REAR_CAMERA_LEFT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/rear_cam_left",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(-0.32, -0.06, 0.05), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"),
)
REAR_CAMERA_RIGHT_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/rear_cam_right",
    update_period=1 / 30,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=_FRONT_CAM_SPAWN,
    offset=CameraCfg.OffsetCfg(pos=(-0.32, 0.06, 0.05), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"),
)

# Phase 2 placeholder -- do not implement until that phase's own research
# pass. Field exists on SensorSuiteCfg now so Phase 2 doesn't need to
# restructure it (sensor_setup_handoff.md §2.4).
WRIST_CAMERA_CFG = None

LIDAR_CAMERA_SUITE = SensorSuiteCfg(
    name="lidar_camera",
    lidar=LIDAR_CFG,
    front_camera=FRONT_CAMERA_LEFT_CFG,
    rear_camera=None,
    wrist_camera=WRIST_CAMERA_CFG,
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
