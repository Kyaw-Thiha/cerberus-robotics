"""SLAM-eval env cfgs: the Go2 Play env (policy/locomotion/core/platforms/
go2/env_cfg.py's UnitreeGo2FlatCerberusEnvCfg_PLAY) with one of the two
locked sensor suites (common/cerberus_description/sensors/sensor_configs.py)
attached, for stepping a trained locomotion checkpoint while a SLAM backend consumes
the published sensor stream. The cfgs run on Isaac Lab's stock flat terrain
(/World/ground); a GRScenes scene is swapped in at runtime, per run, by
scene_env_cfg.apply_scene() (not one gym id per scene). Physics is PhysX with the
legacy actuator path: the parent Cerberus Go2 cfg pins it through
common/cerberus_compat/physics.py. One
class + gym task id per suite (mirrors registry.py's own
one-task-id-per-variant convention) rather than a runtime-selectable field,
so `--task` alone picks the rig, same as `--terrain flat|rough` already does
for training.

Registered via a direct gym.register() call, not through
policy/locomotion/core/platforms/registry.py's register_platform() -- that
registry is about robot IDENTITY (a new platform: new URDF, new agent cfg,
new push-disturbance tuning), not sensor equipping. A sensor-suite variant
of an existing platform's already-registered Play env doesn't fit that
shape; this uses the same underlying gym.register() call registry.py
itself does, just for one axis (sensors) instead of four (flat/rough x
train/play).

NOT independently verified against a live Isaac Lab install (none exists
locally, see docs/isaac_lab_workflow.md) -- needs a pod-side smoke test,
same constraint every other module directly subclassing Isaac Lab env cfgs
in this repo already lives with (e.g. env_cfg.py's own viewer.origin_type
comments).
"""

from __future__ import annotations

import gymnasium as gym
from isaaclab.utils import configclass

from common.cerberus_description.sensors.sensor_configs import (
    CAMERA_ONLY_SUITE,
    CAMERA_STEREO_SUITE,
    LIDAR_CAMERA_SUITE,
    SHOWCASE_CAMERA_CFG,
)
from common.cerberus_description.sensors.sensor_suite import attach_sensor_suite
from policy.locomotion.core.platforms.go2.env_cfg import (
    UnitreeGo2FlatCerberusEnvCfg_PLAY,
)
from policy.locomotion.core.platforms.go2.platform import GO2
from slam.slam.eval.sim_run_args import (
    TASK_ID_CAMERA_ONLY,
    TASK_ID_CAMERA_STEREO,
    TASK_ID_LIDAR_CAMERA,
)


@configclass
class UnitreeGo2SlamEvalLidarCameraEnvCfg(UnitreeGo2FlatCerberusEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        attach_sensor_suite(self.scene, LIDAR_CAMERA_SUITE)
        # attach_sensor_suite() deliberately never wires showcase_camera --
        # see with_showcase_camera()'s own docstring ("eval scripts use this
        # to opt in per-run"), so it's set directly on the scene cfg here.
        self.scene.showcase_camera = SHOWCASE_CAMERA_CFG
        # The PLAY cfg's asset_root follow-cam viewer tracking crashes on
        # teardown here (AttributeError: ManagerBasedRLEnv has no attribute
        # scene, inside viewport_camera_controller.py's per-event tracking
        # callback) -- confirmed by hitting this exact traceback on a real
        # pod run. The showcase camera above is a real CameraCfg sensor
        # prim, driven explicitly every step via compute_chase_cam_pose()
        # (see sim_ground_truth_loop.py), independent of this viewport
        # follow-cam mechanism -- so reverting to the plain fixed viewer
        # here doesn't affect showcase-camera video capture.
        self.viewer.origin_type = "world"


@configclass
class UnitreeGo2SlamEvalCameraOnlyEnvCfg(UnitreeGo2FlatCerberusEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        attach_sensor_suite(self.scene, CAMERA_ONLY_SUITE)
        self.scene.showcase_camera = SHOWCASE_CAMERA_CFG
        self.viewer.origin_type = "world"  # see UnitreeGo2SlamEvalLidarCameraEnvCfg's comment


@configclass
class UnitreeGo2SlamEvalCameraStereoEnvCfg(UnitreeGo2FlatCerberusEnvCfg_PLAY):
    """multicam_slam's rig -- front+rear genuine stereo pairs (CAMERA_STEREO_SUITE),
    not the single-camera-per-position CAMERA_ONLY_SUITE. No LiDAR, so no
    lidar to re-point at a scene (scene_env_cfg.apply_scene) -- cameras don't
    raycast against scene geometry."""

    def __post_init__(self):
        super().__post_init__()
        attach_sensor_suite(self.scene, CAMERA_STEREO_SUITE)
        self.scene.showcase_camera = SHOWCASE_CAMERA_CFG
        self.viewer.origin_type = "world"  # see UnitreeGo2SlamEvalLidarCameraEnvCfg's comment


_registered = False


def register_slam_eval_tasks() -> None:
    """Idempotent -- gym.register() itself raises on a duplicate id, and a
    pod-side eval script may import this module more than once indirectly
    (e.g. once directly, once via a shared import chain)."""
    global _registered
    if _registered:
        return
    for task_id, env_cfg_cls in (
        (TASK_ID_LIDAR_CAMERA, UnitreeGo2SlamEvalLidarCameraEnvCfg),
        (TASK_ID_CAMERA_ONLY, UnitreeGo2SlamEvalCameraOnlyEnvCfg),
        (TASK_ID_CAMERA_STEREO, UnitreeGo2SlamEvalCameraStereoEnvCfg),
    ):
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{env_cfg_cls.__name__}",
                "rsl_rl_cfg_entry_point": GO2.flat_agent_cfg,
            },
        )
    _registered = True

