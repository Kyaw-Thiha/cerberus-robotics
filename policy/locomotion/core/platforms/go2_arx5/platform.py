"""``go2_arx5`` platform registration: the fused Go2 + ARX5 body as a first-class platform.

Same four task ids' pattern as ``go2``; the ids differ so a legs-only checkpoint and an arm checkpoint can never be
confused in a run name, a checkpoint meta file or a report.
"""

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)

from policy.locomotion.core.platforms.registry import PlatformCfg, register_platform

# Same recipe as the legs-only GA retrain (docs/reports/ga_retrain_and_gate_report.md): the agent cfg is reused
# unchanged, so PPO hyperparameters, network shape and the 24-step rollout are identical; only the body differs.
GO2_ARX5 = PlatformCfg(
    name="go2_arx5",
    flat_task_id="Isaac-Velocity-Flat-Unitree-Go2-Arx5-Cerberus-v0",
    flat_play_task_id="Isaac-Velocity-Flat-Unitree-Go2-Arx5-Cerberus-Play-v0",
    rough_task_id="Isaac-Velocity-Rough-Unitree-Go2-Arx5-Cerberus-v0",
    rough_play_task_id="Isaac-Velocity-Rough-Unitree-Go2-Arx5-Cerberus-Play-v0",
    flat_env_cfg_entry_point="policy.locomotion.core.platforms.go2_arx5.env_cfg:UnitreeGo2Arx5FlatCerberusEnvCfg",
    rough_env_cfg_entry_point="policy.locomotion.core.platforms.go2_arx5.env_cfg:UnitreeGo2Arx5RoughCerberusEnvCfg",
    flat_agent_cfg=UnitreeGo2FlatPPORunnerCfg,
    rough_agent_cfg=UnitreeGo2RoughPPORunnerCfg,
    # same push as the legs-only platform: the arm changes the body, not the disturbance the policy must survive
    push_peak_force_n=120.0,
    push_peak_torque_nm=18.0,
    root_body_name="base",
    stairs_step_height_scale=0.5,
    num_steps_per_env=24,
)

register_platform(GO2_ARX5)
