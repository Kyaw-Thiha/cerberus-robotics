"""Isaac Lab Mimic env for the fixed-base Go2 + ARX5 stack task.

Imports isaaclab at module level: only import after the Isaac app has launched (registered through
`register.py` as a string entry point).

Based on (Isaac Lab tag `v3.0.0-EA`):
- `isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py` (`FrankaCubeStackIKRelMimicEnv`; methods copied, one EEF)
- `isaaclab/envs/manager_based_rl_mimic_env.py` (`ManagerBasedRLMimicEnv`: abstract methods, `get_object_poses`
  reads `scene.get_state(is_relative=True)['rigid_object']`, so tracked objects must be `RigidObject`s: cube_1/cube_2)

Quaternions: Isaac Lab 3.0 is XYZW. Nothing here indexes quaternion components; all conversions go through
`isaaclab.utils.math` (`matrix_from_quat`, `quat_from_matrix`, `axis_angle_from_quat`, `quat_from_angle_axis`),
which follow the installed convention.

Action semantics are the stock Franka ones (see `waypoints.py`): the first six action entries are the raw
relative-IK command in metres/radians (the IK term multiplies them by its scale), the last entry is the gripper.

UNVERIFIED: that Mimic generation reaches high success with this arm. `target_eef_pose_to_action` keeps the stock
convention (action = pose error, unscaled), so the commanded step is `IK_ACTION_SCALE` times the error; only a pod
run of annotate/generate can show whether the DLS IK tracks the generated targets (first thing to tune: action_noise,
num_interpolation_steps in `mimic_env_cfg.py`).
"""

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from policy.manipulation.go2_arx5_stack.layout import SUBTASK_GRASP_SIGNAL


class Go2Arx5StackMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for the Go2 + ARX5 stack IK-Rel env (single end effector)."""

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Current TCP pose as 4x4 matrices, in the frame the relative IK actions are expressed in.

        Args:
            eef_name: Name of the end effector (unused: there is exactly one).
            env_ids: Environment indices; None means all envs.

        Returns:
            Tensor of shape (len(env_ids), 4, 4).
        """
        if env_ids is None:
            env_ids = slice(None)

        # Pose from the observation buffer (`ee_frame_pos` is world minus env origin, `ee_frame_quat` is world).
        # The fixed robot base sits at the env origin with identity rotation, so this is also the base frame
        # orientation the IK term uses.
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Turn a target TCP pose and gripper action into an env action (relative pose delta + gripper).

        Args:
            target_eef_pose_dict: 4x4 target pose for each end effector.
            gripper_action_dict: Gripper action for each end effector.
            action_noise_dict: Per-eef noise amplitude, or None for no noise.
            env_id: Environment index to compute the action for.

        Returns:
            Action tensor of shape (7,), compatible with `env.step()`.
        """
        eef_name = next(iter(self.cfg.subtask_configs.keys()))

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # delta position, and delta rotation as an axis-angle vector
        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # gripper action for the single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to the pose part
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Inverse of `target_eef_pose_to_action`: the TCP pose an env action points at.

        Args:
            action: Env action(s), shape (num_envs, action_dim).

        Returns:
            {eef_name: 4x4 target pose tensor}.
        """
        eef_name = next(iter(self.cfg.subtask_configs.keys()))

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # target position
        target_pos = curr_pos + delta_position

        # axis-angle -> rotation matrix; the axis is zeroed where the angle is ~0 (division by zero)
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        is_close_to_zero_angle = torch.isclose(
            delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
        ).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(
            delta_rotation_axis
        )[is_close_to_zero_angle]

        # Same as the stock Franka env minus its `.squeeze(0)`, which only mattered for a single env.
        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        )
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Gripper part (last action entry) of a sequence of env actions.

        Args:
            actions: Env actions, shape (num_envs, num steps in a demo, action_dim).

        Returns:
            {eef_name: tensor of shape (num_envs, num steps, 1)}.
        """
        return {next(iter(self.cfg.subtask_configs.keys())): actions[:, -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Subtask termination flags from the `subtask_terms` observation group.

        Only `grasp_1` is needed: the last subtask (place on cube_1) has no termination signal.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = {}
        subtask_terms = self.obs_buf["subtask_terms"]
        signals[SUBTASK_GRASP_SIGNAL] = subtask_terms[SUBTASK_GRASP_SIGNAL][env_ids]
        return signals
