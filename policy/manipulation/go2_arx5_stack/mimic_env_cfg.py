"""Isaac Lab Mimic env cfg for the fixed-base Go2 + ARX5 stack task: grasp cube_2, then place it on cube_1.

Imports isaaclab at module level: only import after the Isaac app has launched (loaded through the string entry point
registered in `register.py`).

Based on (Isaac Lab tag `v3.0.0-EA`):
- `isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env_cfg.py` (`FrankaCubeStackIKRelMimicEnvCfg`: datagen_config values,
  SubTaskConfig fields; the Franka task has four subtasks, this one has two)
- `isaaclab/envs/mimic_env_cfg.py` (`MimicEnvCfg`, `DataGenConfig`, `SubTaskConfig`)

Contract with the env: subtask 0 ends when the `grasp_1` observation in the `subtask_terms` group fires; the last
subtask has `subtask_term_signal=None` and offset (0, 0); `terminations.success` exists in the env cfg.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from policy.manipulation.go2_arx5_stack.env_cfg import Go2Arx5StackEnvCfg
from policy.manipulation.go2_arx5_stack.layout import (
    MIMIC_EEF_NAME,
    SUBTASK_GRASP_SIGNAL,
)


@configclass
class Go2Arx5StackMimicEnvCfg(Go2Arx5StackEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for the Go2 + ARX5 stack IK-Rel env."""

    def __post_init__(self):
        # post init of the env cfg parent (it does not chain to MimicEnvCfg, same as the Franka cfg)
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_go2_arx5_stack_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="cube_2",
                # Binary indicator in "datagen_info" that signals when this subtask is finished (0 -> 1 edge).
                subtask_term_signal=SUBTASK_GRASP_SIGNAL,
                # Random offsets added to the termination boundary when splitting a trajectory into segments.
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube",
                next_subtask_description="Place red cube on top of blue cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                # End of the final subtask does not need to be detected.
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place red cube on top of blue cube",
            )
        )
        self.subtask_configs[MIMIC_EEF_NAME] = subtask_configs
