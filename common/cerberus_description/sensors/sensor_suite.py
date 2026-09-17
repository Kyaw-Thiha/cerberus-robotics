"""Generic sensor-suite composition contract. Mirrors
`policy/locomotion/core/platforms/registry.py`'s split: this module stays
free of any `isaaclab`/`isaaclab.sensors` import (so it unit-tests locally,
no pod needed) while the actual Isaac Lab sensor objects (`RayCasterCfg`,
`CameraCfg`, ...) live in `sensor_configs.py`, which does import isaaclab and
is therefore only import-time-verifiable on a pod. See
`plans/sensor_setup_handoff.md` for the design this implements.

Field types are `Any | None` rather than `RayCasterCfg | None` /
`CameraCfg | None` for the same reason `PlatformCfg.flat_agent_cfg` in
registry.py is typed `Any`: importing the real Isaac Lab type here would
force every caller (including this module's own unit tests) to have
isaaclab installed just to construct a `SensorSuiteCfg`.
"""

from __future__ import annotations

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
    front_camera: Any  # CameraCfg -- always present, both locked rigs
    rear_camera: Any | None  # CameraCfg | None -- camera-only rig only
    wrist_camera: Any | None = None  # CameraCfg | None -- Phase 2+ placeholder, both rigs once implemented

    def active_sensors(self) -> dict[str, bool]:
        """Which named sensor slots are actually populated on this rig, as a
        plain bool map -- the same shape `system/cerberus_bringup`'s
        `sensor_profiles.yaml` records on the ROS2 side, so the two can be
        compared/kept in sync without either side importing the other's
        objects (see the platform-selection precedent in
        `docs/superpowers/specs/2026-09-09-multi-platform-config-design.md`
        for why that cross-language split is deliberate, not an oversight).
        """
        return {
            "lidar": self.lidar is not None,
            "front_camera": self.front_camera is not None,
            "rear_camera": self.rear_camera is not None,
            "wrist_camera": self.wrist_camera is not None,
        }


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
    for name in ("lidar", "front_camera", "rear_camera", "wrist_camera"):
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
