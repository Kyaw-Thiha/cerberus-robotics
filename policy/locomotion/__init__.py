"""Cerberus Phase 0 locomotion task registration.

Importing this module registers four Gym environments, each Isaac Lab's stock
Go2 config plus the Hwangbo-curriculum push disturbance (see REFERENCES.md):

    Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0        (training)
    Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0   (eval)
    Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0       (training)
    Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0  (eval)

PPO hyperparameters are Isaac Lab's stock UnitreeGo2FlatPPORunnerCfg /
UnitreeGo2RoughPPORunnerCfg, reused unmodified -- not redefined here.
"""

import gymnasium as gym
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.core.go2_flat_env_cfg:UnitreeGo2FlatCerberusEnvCfg",
        "rsl_rl_cfg_entry_point": UnitreeGo2FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.core.go2_flat_env_cfg:UnitreeGo2FlatCerberusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": UnitreeGo2FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.core.go2_rough_env_cfg:UnitreeGo2RoughCerberusEnvCfg",
        "rsl_rl_cfg_entry_point": UnitreeGo2RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.core.go2_rough_env_cfg:UnitreeGo2RoughCerberusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": UnitreeGo2RoughPPORunnerCfg,
    },
)
