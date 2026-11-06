"""Generic sensor-suite composition contract. Mirrors
`policy/locomotion/core/platforms/registry.py`'s split: this module stays
free of any `isaaclab`/`isaaclab.sensors` import (so it unit-tests locally,
no pod needed) while the actual Isaac Lab sensor objects (`RayCasterCfg`,
`CameraCfg`, ...) live in `sensor_configs.py`, which does import isaaclab and
is therefore only import-time-verifiable on a pod.

Field types are `Any | None` rather than `RayCasterCfg | None` /
`CameraCfg | None` for the same reason `PlatformCfg.flat_agent_cfg` in
registry.py is typed `Any`: importing the real Isaac Lab type here would
force every caller (including this module's own unit tests) to have
isaaclab installed just to construct a `SensorSuiteCfg`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SensorSuiteCfg:
    """One named, composable sensor rig. A fork adding a third rig (say,
    LiDAR + stereo + thermal) adds one more `SensorSuiteCfg` instance
    somewhere that imports the real Isaac Lab configs (see
    `sensor_configs.py` for the reference example) -- nothing here or
    downstream needs to change unless it specifically consumes the new
    modality.
    """

    name: str  # e.g. "lidar_camera" -- used for --sensor_profile, registry key

    lidar: Any | None  # RayCasterCfg | None
    front_camera: Any  # CameraCfg -- always present, both locked rigs. Doubles as the LEFT
    # camera of a stereo pair on rigs that also populate front_camera_right (see below).
    rear_camera: Any | None  # CameraCfg | None -- camera-only rig only. Same LEFT-of-pair role
    # as front_camera on rigs that also populate rear_camera_right.
    wrist_camera: Any | None = None  # CameraCfg | None -- Phase 2+ placeholder, both rigs once implemented
    imu: Any | None = None  # ImuCfg | None -- lidar_camera rig only. The LIO/LIVO backend family
    # (fast_lio2, fast_livo2, bievr_lio, gs_livo) is lidar-inertial, not lidar-only -- see
    # slam/slam/core/algorithms.py's sensor_requirements for those four. orb_slam3 is explicitly
    # "no IMU" and multicam_slam doesn't declare it either, so camera_only/camera_stereo leave
    # this None, same populated-only-where-needed pattern as rear_camera.
    # RIGHT-eye counterparts for a genuine stereo-pair rig (multicam_slam's camera_stereo
    # suite -- see sensor_configs.py). `None` on every other suite (lidar_camera,
    # camera_only): those rigs only ever populate front_camera/rear_camera as a single
    # camera each, never a pair, so leaving these off `active_sensors()`/`attach_sensor_suite`
    # for them is a no-op, not a behavior change. Kept as separate optional fields rather than
    # widening front_camera/rear_camera to accept a (left, right) tuple, because
    # `attach_sensor_suite` sets these directly as named attributes on an Isaac Lab
    # `InteractiveSceneCfg`, which expects one sensor object per attribute -- a tuple there
    # wouldn't resolve to two real USD camera prims.
    front_camera_right: Any | None = None  # CameraCfg | None
    rear_camera_right: Any | None = None  # CameraCfg | None
    # Video/demo aid, independent of sensor_profile -- not consumed by any
    # SLAM/RL/IL algorithm. Excluded from every sensor_requirements manifest
    # and RL observation-space gating. Only turned on via with_showcase_camera()
    # for single/few-env eval/recording runs, never during massively-parallel
    # RL training.
    showcase_camera: Any | None = None  # CameraCfg | None

    def active_sensors(self) -> dict[str, bool]:
        """Which named sensor slots are actually populated on this rig, as a
        plain bool map -- the same shape `system/cerberus_bringup`'s
        `sensor_profiles.yaml` records on the ROS2 side, so the two can be
        compared/kept in sync without either side importing the other's
        objects (see the platform-selection precedent in
        `docs/superpowers/specs/2026-09-09-multi-platform-config-design.md`
        for why that cross-language split is deliberate, not an oversight).

        Deliberately excludes front_camera_right/rear_camera_right: those are a
        presence-only detail of *how* front_camera/rear_camera are populated on a
        stereo-pair rig (still True/True here), not a distinct modality any
        `sensor_requirements` manifest gates on -- see sensor_suite.py's field
        docstring and slam/slam/core/algorithms.py's multicam_slam entry.
        """
        return {
            "lidar": self.lidar is not None,
            "front_camera": self.front_camera is not None,
            "rear_camera": self.rear_camera is not None,
            "wrist_camera": self.wrist_camera is not None,
            "imu": self.imu is not None,
        }


def with_showcase_camera(suite: SensorSuiteCfg, camera: Any) -> SensorSuiteCfg:
    """Returns a copy of `suite` with showcase_camera attached -- the locked
    Phase-1 suites (LIDAR_CAMERA_SUITE/CAMERA_ONLY_SUITE) stay untouched;
    eval scripts use this to opt in per-run instead."""
    return dataclasses.replace(suite, showcase_camera=camera)


def attach_sensor_suite(scene_cfg: Any, suite: SensorSuiteCfg) -> None:
    """Sets each populated sensor slot on `scene_cfg` as an attribute
    (`scene_cfg.lidar = suite.lidar`, etc.), skipping `None` slots. Isaac
    Lab's `@configclass`-wrapped scene/env cfgs are plain mutable
    dataclasses (this repo's own `core/platforms/go2/env_cfg.py` already
    mutates `self.scene`/`self.events` post-`__post_init__` the same way),
    so this needs no `isaaclab` import -- works on any object with matching
    attribute names, real or a test double. A future env cfg calls this
    once from its own `__post_init__`:
    `attach_sensor_suite(self.scene, get_suite(profile_name))`.
    """
    for name in (
        "lidar",
        "front_camera",
        "rear_camera",
        "wrist_camera",
        "front_camera_right",
        "rear_camera_right",
        "imu",
    ):
        sensor = getattr(suite, name)
        if sensor is not None:
            setattr(scene_cfg, name, sensor)


_SUITES: dict[str, SensorSuiteCfg] = {}


def register_suite(cfg: SensorSuiteCfg) -> None:
    if cfg.name in _SUITES:
        raise ValueError(f"sensor suite '{cfg.name}' already registered")
    _SUITES[cfg.name] = cfg


def get_suite(name: str) -> SensorSuiteCfg:
    try:
        return _SUITES[name]
    except KeyError:
        raise ValueError(f"unknown sensor suite '{name}', available: {sorted(_SUITES)}") from None


def available_suites() -> list[str]:
    return sorted(_SUITES)
