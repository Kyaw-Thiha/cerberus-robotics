"""``go2_arx5`` locomotion env cfgs: the same Cerberus rough/flat tasks on the fused Go2 + ARX5 body.

The policy must stay the *same contract* as the legs-only checkpoint: 12 leg joints by name, in the asset's own
order (hips, thighs, calves per leg), the same 235-D observation and the same 12-D action, so the exported ONNX
policy still applies. The arm and gripper are held at the stowed pose by their implicit PD actuators and are never
part of the action vector.

Why the overrides are needed: Isaac Lab's stock velocity cfg uses ``joint_names=[".*"]`` for the action term and no
``asset_cfg`` for the joint observations, which on a 20-DoF body would silently widen obs to 256 and actions to 20.

Arm robustness during the retrain (docs/cerberus_project_context.md): per-episode randomised arm joint offsets
(the arm's position target is stowed + offset, and the joints are reset to it) and a randomised torso payload mass.
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from common.cerberus_description.go2_arx5_cfg import GO2_ARX5_CFG
from common.cerberus_description.go2_arx5_spec import (
    ARX5_ARM_JOINTS,
    ARX5_STOWED_JOINT_POS,
    GO2_ARX5_BASE_PRIM_REL_PATH,
    GRIPPER_JOINT_NAMES,
)
from common.cerberus_eval.scene_assertions import contract_joint_order
from policy.locomotion.core.platforms.go2.env_cfg import (
    UnitreeGo2FlatCerberusEnvCfg,
    UnitreeGo2FlatCerberusEnvCfg_PLAY,
    UnitreeGo2RoughCerberusEnvCfg,
    UnitreeGo2RoughCerberusEnvCfg_PLAY,
)
from policy.locomotion.core.platforms.go2_arx5.arm_hold import (
    randomize_arm_hold,
    randomize_payload_mass,
)

# The policy contract's joint order (hips, thighs, calves) -- the fused USD's own order is per leg, so positional
# data would be silently permuted without pinning this list.
CONTRACT_LEG_JOINTS: tuple[str, ...] = tuple(contract_joint_order())
ARM_JOINT_NAMES: tuple[str, ...] = (*ARX5_ARM_JOINTS, *GRIPPER_JOINT_NAMES)
STOWED_ARM_TARGET_RAD: dict[str, float] = dict(ARX5_STOWED_JOINT_POS)

# Retrain-time randomisation ranges (docs/cerberus_project_context.md: the arm is 3.35 kg with a 1.5 kg rated
# payload). The offsets are deliberately small: the arm is a load, not something the legs policy is asked to use.
ARM_OFFSET_RAD = 0.12  # +-0.12 rad on each arm joint, +-5 mm on each finger
PAYLOAD_KG = (0.0, 1.5)  # extra mass on the torso, uniform


def _leg_asset_cfg() -> SceneEntityCfg:
    """Leg joints by name **in contract order** (``preserve_order=True``), as a fresh object per term.

    The fused USD's emission order is per leg (FR, FL, RR, RL) because it comes from the merged URDF, not from
    Isaac Lab's stock Go2 asset (hips, thighs, calves). The policy contract is positional, so the order has to be
    pinned by name; reusing one cfg object across terms also breaks, because Isaac Lab resolves (and mutates)
    ``joint_ids`` in place.
    """
    return SceneEntityCfg("robot", joint_names=list(CONTRACT_LEG_JOINTS), preserve_order=True)


def _apply_arm_embodiment(env_cfg) -> None:
    """Shared overrides for every go2_arx5 variant."""
    env_cfg.scene.robot = GO2_ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # height scan frame: the fused USD keeps the Go2 root body under Geometry/base
    env_cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + GO2_ARX5_BASE_PRIM_REL_PATH

    env_cfg.actions.joint_pos.joint_names = list(CONTRACT_LEG_JOINTS)
    env_cfg.actions.joint_pos.asset_name = "robot"
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.observations.policy.joint_pos.params = {"asset_cfg": _leg_asset_cfg()}
    env_cfg.observations.policy.joint_vel.params = {"asset_cfg": _leg_asset_cfg()}
    # the critic (asymmetric) stack, when present, sees the same joints
    critic = getattr(env_cfg.observations, "critic", None)
    if critic is not None:
        for term_name in ("joint_pos", "joint_vel"):
            term = getattr(critic, term_name, None)
            if term is not None:
                term.params = {"asset_cfg": _leg_asset_cfg()}


def _add_arm_events(env_cfg) -> None:
    """Per-episode arm offsets and a one-off payload draw; ``CERBERUS_ARM_EVENTS=0`` is the bisect switch."""
    if os.environ.get("CERBERUS_ARM_EVENTS") == "0":
        env_cfg.events.arm_hold_offset = None
        env_cfg.events.payload_mass = None
        return
    robot = SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES))
    env_cfg.events.arm_hold_offset = EventTerm(
        func=randomize_arm_hold,
        mode="reset",
        params={"asset_cfg": robot, "max_offset_rad": ARM_OFFSET_RAD,
                "stowed": [STOWED_ARM_TARGET_RAD[j] for j in ARM_JOINT_NAMES]},
    )
    if os.environ.get("CERBERUS_PAYLOAD_EVENTS", "1") == "1":
        env_cfg.events.payload_mass = EventTerm(
            func=randomize_payload_mass,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
                    "mass_range_kg": PAYLOAD_KG},
        )
    else:
        env_cfg.events.payload_mass = None


@configclass
class UnitreeGo2Arx5RoughCerberusEnvCfg(UnitreeGo2RoughCerberusEnvCfg):
    """Training cfg: rough terrain, push curriculum, arm stowed with per-episode offsets and payload."""

    def __post_init__(self):
        super().__post_init__()
        _apply_arm_embodiment(self)
        _add_arm_events(self)


@configclass
class UnitreeGo2Arx5RoughCerberusEnvCfg_PLAY(UnitreeGo2RoughCerberusEnvCfg_PLAY):
    """Eval cfg: no randomisation, no pushes, arm stowed -- the shape the scene runner builds."""

    def __post_init__(self):
        super().__post_init__()
        _apply_arm_embodiment(self)
        self.events.arm_hold_offset = None
        self.events.payload_mass = None


@configclass
class UnitreeGo2Arx5FlatCerberusEnvCfg(UnitreeGo2FlatCerberusEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_arm_embodiment(self)
        _add_arm_events(self)


@configclass
class UnitreeGo2Arx5FlatCerberusEnvCfg_PLAY(UnitreeGo2FlatCerberusEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _apply_arm_embodiment(self)
        self.events.arm_hold_offset = None
        self.events.payload_mass = None
