"""Go2 flat locomotion with the Hwangbo-curriculum push disturbance added on top
of Isaac Lab's stock UnitreeGo2FlatEnvCfg. Rewards, PPO hyperparameters, and
network architecture are untouched -- see REFERENCES.md.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from .push_disturbance_cfg import add_push_disturbance


@configclass
class UnitreeGo2FlatCerberusEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_push_disturbance(self)


@configclass
class UnitreeGo2FlatCerberusEnvCfg_PLAY(UnitreeGo2FlatCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing events -- push recovery is evaluated separately
        # (policy/locomotion/eval/), not exercised during a play/visualization run
        self.events.base_external_force_torque = None
        self.events.push_disturbance = None
        # the curriculum term references push_disturbance above; must go too
        self.curriculum.push_disturbance_curriculum = None

        # Follow-cam: the default viewer is a fixed world-space camera that
        # never tracks the robot, so footage drifts out of frame once a push
        # (or just plain walking) moves the robot away from its spawn point --
        # see REFERENCES.md's camera-framing writeup. asset_root origin makes
        # Isaac Lab recompute eye/lookat from the robot's current root pose
        # every step instead of once. Needs a pod-side smoke test to confirm
        # (not verifiable from source alone -- Isaac Lab isn't installed here).
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (-2.5, -2.5, 1.3)
        self.viewer.lookat = (0.0, 0.0, 0.3)
