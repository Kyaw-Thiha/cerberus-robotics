"""Articulation config for the fused Go2 + ARX5 (X5A) robot: 12 leg joints, joint1..6, joint7/joint8 fingers.

Imports isaaclab: only import lazily (after the app is launched).
Derived from the stock Go2 config so leg numbers are never copied.

Exports `GO2_ARX5_CFG` (floating base), `GO2_ARX5_FIXED_BASE_CFG` (root link fixed, for fixed-base demos) and
`make_go2_arx5_cfg(fix_base)`. The effort-limit kwarg differs between Isaac Lab 2.3.2 (`effort_limit_sim`) and the
3.0 line (`joint_effort_limit`); `_actuator_kwargs` picks it by inspecting the dataclass fields.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from common.cerberus_description.go2_arx5_spec import (
    ARX5_ACTUATOR_GAINS,
    ARX5_ARM_JOINTS,
    ARX5_STOWED_JOINT_POS,
    GRIPPER_ACTUATOR_GAINS,
    GRIPPER_JOINT_NAMES,
    fused_usd_path,
    implicit_actuator_kwargs,
)


def _actuator_kwargs(joint_names_expr, stiffness, damping, effort_limit, armature=None) -> dict:
    """ImplicitActuatorCfg kwargs that work on both Isaac Lab lines (see implicit_actuator_kwargs)."""
    return implicit_actuator_kwargs(
        ImplicitActuatorCfg, joint_names_expr, stiffness, damping, effort_limit, armature=armature
    )


# Arm drive stiffness/damping in the USD are zero, so the gains must come from
# here. One actuator group per arm joint. prim_path is left to callers.
# Verified on pod 2026-10-05 (Isaac Lab 2.3.2 / Sim 5.1): `effort_limit_sim` is accepted by ImplicitActuatorCfg.
# UNVERIFIED: the `joint_effort_limit` name on the Isaac Lab 3.0 line (taken from the shared contract).
_ARM_ACTUATORS = {
    f"arx5_{joint}": ImplicitActuatorCfg(
        **_actuator_kwargs(
            [joint],
            ARX5_ACTUATOR_GAINS[joint].stiffness,
            ARX5_ACTUATOR_GAINS[joint].damping,
            ARX5_ACTUATOR_GAINS[joint].effort_limit,
            armature=ARX5_ACTUATOR_GAINS[joint].armature,
        )
    )
    for joint in ARX5_ARM_JOINTS
}

# Both fingers in one group; the fingers are independent prismatic joints (no mimic).
# PROVISIONAL gains (see GRIPPER_ACTUATOR_GAINS), not tuned.
_GRIPPER_ACTUATORS = {
    "arx5_gripper": ImplicitActuatorCfg(
        **_actuator_kwargs(
            list(GRIPPER_JOINT_NAMES),
            GRIPPER_ACTUATOR_GAINS.stiffness,
            GRIPPER_ACTUATOR_GAINS.damping,
            GRIPPER_ACTUATOR_GAINS.effort_limit,
        )
    )
}


def make_go2_arx5_cfg(fix_base: bool = False):
    """Fused Go2 + ARX5 ArticulationCfg; `fix_base=True` fixes the root link (Go2 `base`) to the world.

    Stock soft_joint_pos_limit_factor is kept (not overridden).
    """
    spawn = UNITREE_GO2_CFG.spawn.replace(usd_path=str(fused_usd_path()))
    if fix_base:
        if hasattr(spawn, "fix_root_link"):
            # Lab 3.0 GA: `fix_root_link` is a field of the spawner cfg; `articulation_props` is a list of schema fragments.
            spawn = spawn.replace(fix_root_link=True)
        else:
            # Lab 2.x: `fix_root_link` lives on ArticulationRootPropertiesCfg.
            props = spawn.articulation_props
            if props is None:
                from isaaclab.sim import ArticulationRootPropertiesCfg

                props = ArticulationRootPropertiesCfg()
            spawn = spawn.replace(articulation_props=props.replace(fix_root_link=True))
    return UNITREE_GO2_CFG.replace(
        spawn=spawn,
        init_state=UNITREE_GO2_CFG.init_state.replace(
            joint_pos={**UNITREE_GO2_CFG.init_state.joint_pos, **ARX5_STOWED_JOINT_POS}
        ),
        actuators={**UNITREE_GO2_CFG.actuators, **_ARM_ACTUATORS, **_GRIPPER_ACTUATORS},
    )


GO2_ARX5_CFG = make_go2_arx5_cfg(fix_base=False)
GO2_ARX5_FIXED_BASE_CFG = make_go2_arx5_cfg(fix_base=True)
