"""Go2's PlatformCfg instance -- the single source of truth for every
Go2-specific value the training/eval/CLI stack needs. See
core/platforms/registry.py for the field contract every platform must
satisfy.
"""

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
    UnitreeGo2RoughPPORunnerCfg,
)

from policy.locomotion.core.platforms.registry import PlatformCfg, register_platform

# Push peak force anchored to isaaclab-go2-locomotion's validated Go2 number
# (87.5% recovery at 120N peak vs 0% for the untrained baseline). Torque has
# no verified source anywhere in the Hwangbo/Lee/Miki lineage or the
# reference repo; it's a reasoned ESTIMATE (peak force x ~0.15m, an
# approximate Go2 body-frame lever arm), not a verified figure. See
# policy/locomotion/REFERENCES.md.
#
# stairs_step_height_scale: stock UnitreeGo2RoughEnvCfg already rescales
# "boxes"/"random_rough" for Go2's smaller body size, but leaves
# "pyramid_stairs"/"pyramid_stairs_inv" at their stock (ANYmal-scale)
# step_height_range. Rescaled by the same 0.5x factor stock's own boxes
# rescale used -- a reasoned estimate following that established pattern,
# not independently verified. See policy/locomotion/RUN_001.md and
# REFERENCES.md.
GO2 = PlatformCfg(
    name="go2",
    flat_task_id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-v0",
    flat_play_task_id="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    rough_task_id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0",
    rough_play_task_id="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    flat_env_cfg_entry_point="policy.locomotion.core.platforms.go2.env_cfg:UnitreeGo2FlatCerberusEnvCfg",
    rough_env_cfg_entry_point="policy.locomotion.core.platforms.go2.env_cfg:UnitreeGo2RoughCerberusEnvCfg",
    flat_agent_cfg=UnitreeGo2FlatPPORunnerCfg,
    rough_agent_cfg=UnitreeGo2RoughPPORunnerCfg,
    push_peak_force_n=120.0,
    push_peak_torque_nm=18.0,
    root_body_name="base",
    stairs_step_height_scale=0.5,
    num_steps_per_env=24,
)

register_platform(GO2)
