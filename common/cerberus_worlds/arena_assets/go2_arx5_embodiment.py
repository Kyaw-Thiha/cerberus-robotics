"""Out-of-tree Isaac Lab Arena embodiment for the FIXED-BASE Go2 + ARX5 (X5A), plus a kitchen pick-and-place environment
factory (W2b: Mimic demo in a Lightwheel RoboCasa kitchen composed through Arena).

Everything Arena/isaaclab related is imported lazily (inside functions / ``_build_arena_classes``), so this module
imports with plain python. The Arena classes are created on first use and cached; creating them runs Arena's
``@register_asset`` decorator, which is the registration. Call ``register_go2_arx5_assets()`` AFTER the Isaac app has
launched (Arena pulls in ``pxr``/Kit modules). ``go2_embodiment.register_go2_arx5_assets`` is the public hook and
calls this module.

What Arena has for Mimic at ``main`` 311d0f0fc (read from ``third_party/IsaacLab-Arena``, nothing was run)
--------------------------------------------------------------------------------------------------------
PRESENT
  * ``ArenaEnvBuilderCfg(mimic=True)``: ``ArenaEnvBuilder.compose_manager_cfg`` then builds an
    ``IsaacArenaManagerBasedMimicEnvCfg(IsaacLabArenaManagerBasedRLEnvCfg, MimicEnvCfg)`` from
    ``task.get_mimic_env_cfg(arm_mode=embodiment.arm_mode)`` (datagen_config, subtask_configs,
    task_constraint_configs, mimic_recorder_config) and ``get_entry_point`` returns ``embodiment.get_mimic_env()``,
    i.e. the embodiment must expose a ``ManagerBasedRLMimicEnv`` subclass in ``self.mimic_env`` (Franka: a copy
    of ``FrankaCubeStackIKRelMimicEnv`` in ``embodiments/franka/franka.py``). Metrics, recorders and
    ``episode_recorders`` are excluded from the mimic env cfg.
  * ``PickAndPlaceTask`` (``tasks/pick_and_place_task.py``): ``get_mimic_env_cfg`` returns ``PickPlaceMimicEnvCfg``
    (two subtasks, eef key ``"robot"`` for ``ArmMode.SINGLE_ARM``; subtask 0 ends on signal ``grasp_1``) or whatever
    the ``mimic_env_cfg_factory`` argument returns. Success = ``object_on_destination`` (contact-sensor based).
  * ``isaaclab_arena/environments/isaaclab_interop.py::environment_registration_callback``: the documented way to
    run Isaac Lab's own ``record_demos.py`` / ``annotate_demos.py`` / ``generate_dataset.py`` on an Arena env
    (``--external_callback``, ``--task <env name>``, ``--mimic``, ``--external_environment_class_path module:Class``
    for an out-of-tree ``ExampleEnvironmentBase``). Docs: ``docs/pages/example_workflows/static_manipulation/``.
ABSENT / NOT USABLE ON THIS COMMIT
  * Arena ships no annotate/generate script of its own (docs call Lab's scripts in ``submodules/IsaacLab``).
  * No ``subtask_terms`` observation group in any embodiment: ``grasp_1`` is only a signal NAME in the task's
    ``SubTaskConfig``, so automatic annotation needs the embodiment (here) to provide the group and the mimic env to
    read it (``get_subtask_term_signals``). Done below / in ``policy/manipulation/go2_arx5_stack/mimic_env.py``.
  * ``PickPlaceMimicEnvCfg`` dereferences ``destination_object.name``; pass ``destination_object`` or a
    ``mimic_env_cfg_factory`` (we do the latter to get the W2a datagen settings).
  * The Lab ``annotate_demos.py`` at the pinned Lab commit (bb0c8e1b9) reads ``args_cli.headless`` which the pinned
    ``AppLauncher.add_app_launcher_args`` no longer registers (``--viz`` replaced it): it would raise
    ``AttributeError`` before doing anything, and fixing that means patching Lab. ``scripts/arena_mimic_go2_arx5.py
    annotate`` therefore runs an in-process port of its auto mode by default.

Design (thin wrapper over what ``policy/manipulation/go2_arx5_stack`` already defines)
--------------------------------------------------------------------------------------
* Robot cfg = ``make_go2_arx5_cfg(fix_base=True)`` (prim ``{ENV_REGEX_NS}/Robot``, root body ``base`` fixed).
* Actions: ONE ``DifferentialInverseKinematicsActionCfg`` (joint1..joint6, body ``TCP_BODY_NAME``, relative pose, DLS,
  scale 0.5) + ONE ``BinaryJointPositionActionCfg`` (joint7/joint8, open ``GRIPPER_OPEN_M``, close ``GRIPPER_CLOSED_M``).
  NO leg term: legs keep the default-pose joint targets written by ``reset_go2_arx5_to_default``.
  Copied from ``policy/manipulation/go2_arx5_stack/env_cfg.py`` (its builders are inline in ``__post_init__``);
  ``test_go2_arx5_embodiment.py`` checks the constants against that file.
* Observations ``policy``: actions, joint_pos, joint_vel, eef_pos, eef_quat, gripper_pos (absolute finger positions);
  ``subtask_terms``: ``grasp_1`` (only when the pick object name is known). Scene: ``robot`` + ``ee_frame``.
* FRAME CAVEAT (Mimic + stock ``ee_frame_pos`` = world minus env origin, no base rotation): relative IK actions are
  expressed in the robot ROOT frame, so the robot must keep an identity yaw (``--robot_xy`` has no yaw on purpose).
* Kitchen environment (``Go2Arx5KitchenPickAndPlaceEnvironment``, name ``go2_arx5_kitchen_pick_and_place``): the arm
  base is 0.405 m above the floor, a RoboCasa counter top (~0.9 m) is out of reach, so the task sits on a procedural
  0.30 m static riser block placed in front of the robot (same geometry as the W2a table, ``layout.py``) with a
  Lightwheel-SDK object (default ``sweet_potato``) as the pick object and a thin kinematic procedural plate as the
  destination. Object poses are randomized by Arena ``PoseRange`` reset events inside the W2a cube envelope.

# UNVERIFIED: none of the Arena-side classes has been instantiated (needs a pod with Sim 6.1 / Lab 3.0 line). In
# particular: ``is_fixed_base`` with ``fix_root_link``, the robot position in each kitchen
# (the default ``DEFAULT_ROBOT_XY`` is a guess from the viewer camera of the Lightwheel kitchens, floor assumed z=0),
# the grasp of a real object (not a cube) by the 0.088 m gripper, and that ``object_on_destination`` fires for a
# procedural kinematic plate.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from common.cerberus_description.go2_arx5_spec import (
    ARX5_ARM_JOINTS,
    GO2_ARX5_EMBODIMENT,
    GRIPPER_CLOSED_M,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_M,
    TCP_BODY_NAME,
)
from common.cerberus_worlds.arena_assets.go2_embodiment import (
    GO2_ARX5_ARENA_NAME,
    ROBOT_PRIM_PATH,
    ROBOT_SCENE_KEY,
)

# Pure python (spec-only imports); the heavy W2a modules (env_cfg, mimic_env) are imported lazily below.
from policy.manipulation.go2_arx5_stack import layout

# ---------------------------------------------------------------------------------------------------------
# Constants (mirror policy/manipulation/go2_arx5_stack/env_cfg.py; checked by the unit test)
# ---------------------------------------------------------------------------------------------------------
EE_FRAME_NAME = "ee_frame"
EE_TARGET_NAME = "end_effector"
ROBOT_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/{layout.BASE_PRIM_RELATIVE_PATH}"  # FrameTransformer source frame (the fixed root body)
EE_TARGET_PRIM_PATH = f"{ROBOT_PRIM_PATH}/{layout.EEF_PRIM_RELATIVE_PATH}"
ARM_ACTION_NAME = "arm_action"
GRIPPER_ACTION_NAME = "gripper_action"
ACTION_DIM = layout.ACTION_DIM  # 6 (relative pose) + 1 (gripper)

# Isaac Lab DifferentialInverseKinematicsActionCfg / DifferentialIKControllerCfg / BinaryJointPositionActionCfg kwargs.
ARM_ACTION_SPEC: dict[str, Any] = {
    "asset_name": ROBOT_SCENE_KEY,
    "joint_names": list(ARX5_ARM_JOINTS),
    "body_name": TCP_BODY_NAME,
    "scale": layout.IK_ACTION_SCALE,
    "body_offset": None,  # eef_link is itself the TCP body
    "controller": {
        "command_type": "pose",
        "use_relative_mode": True,
        "ik_method": "dls",
    },
}
GRIPPER_ACTION_SPEC: dict[str, Any] = {
    "asset_name": ROBOT_SCENE_KEY,
    "joint_names": list(GRIPPER_JOINT_NAMES),
    "open_command_expr": {name: GRIPPER_OPEN_M for name in GRIPPER_JOINT_NAMES},
    "close_command_expr": {name: GRIPPER_CLOSED_M for name in GRIPPER_JOINT_NAMES},
}
OBSERVATION_TERMS = (
    "actions",
    "joint_pos",
    "joint_vel",
    "eef_pos",
    "eef_quat",
    "gripper_pos",
)
SUBTASK_TERM_NAMES = (layout.SUBTASK_GRASP_SIGNAL,)

# Control rate of the W2a env cfg (the scripted waypoint action magnitudes assume it): 100 Hz physics, 20 Hz control.
SIM_DT = 0.01
DECIMATION = 5
RENDER_INTERVAL = 2

# ---------------------------------------------------------------------------------------------------------
# Kitchen environment options (pure python)
# ---------------------------------------------------------------------------------------------------------
ENV_NAME = "go2_arx5_kitchen_pick_and_place"
ENV_FACTORY_CLASS = "Go2Arx5KitchenPickAndPlaceEnvironment"
ENV_FACTORY_PATH = f"{__name__}:{ENV_FACTORY_CLASS}"
DEFAULT_KITCHEN_REF = "lightwheel:1:1"  # the pair whose anonymous Lightwheel SDK download was checked (arena log)
# Registered Arena objects whose USD comes from the Lightwheel SDK (``LightwheelLazyPath`` in object_library.py);
# every other registered object loads from Arena's Nucleus/S3 path instead.
LIGHTWHEEL_SDK_OBJECTS = ("sweet_potato", "broccoli", "jug", "beer_bottle")
DEFAULT_PICK_OBJECT = "sweet_potato"
# UNVERIFIED guess: floor point in the kitchen frame in front of the counter wall (viewer cameras of the Lightwheel
# kitchens look from y=-5.5 towards y=-1.4 at x=2.75); robot faces +x with identity yaw (see FRAME CAVEAT above).
DEFAULT_ROBOT_XY = (1.5, -2.6)
DEFAULT_OBJECT_HEIGHT_M = (
    0.04  # UNVERIFIED: generic grasp-height proxy for the selected SDK object
)
DEFAULT_PICK_FOOTPRINT_M = (0.08, 0.08)
DEFAULT_PLATE_SIZE_M = (0.10, 0.10)
DEFAULT_PLATE_THICKNESS_M = 0.01
DEFAULT_EPISODE_LENGTH_S = 30.0
PICK_SPAWN_CLEARANCE_M = 0.01  # UNVERIFIED: spawn this far above the riser and assume the selected USD settles by itself
# Keep the sampled support points within the tested W2a table-reach region and <= 0.39 m horizontal radius.
PICK_X_RANGE: tuple[float, float] = (0.32, 0.36)
PICK_Y_RANGE: tuple[float, float] = (0.06, 0.12)
PLATE_X_RANGE: tuple[float, float] = (0.32, 0.36)
PLATE_Y_RANGE: tuple[float, float] = (-0.12, -0.08)
PICK_YAW_RANGE: tuple[float, float] = (
    layout.CUBE_YAW_RANGE[0],
    layout.CUBE_YAW_RANGE[0],
)
# UNVERIFIED: one-task, hardware-limits-aware test pose; not a generalization to other objects or tasks.
PLATE_NAME = "go2_arx5_plate"
RISER_NAME = "go2_arx5_riser"
# UNVERIFIED: Lightwheel SDK objects' actual collision footprints and settling/grasp behavior have not been measured;
# the 8 cm square is a conservative geometry proxy used to keep the default pick region clear of the plate.


@dataclass(frozen=True)
class KitchenOptions:
    """Everything the kitchen factory needs, with defaults that match the CLI flags."""

    kitchen_scene: str = DEFAULT_KITCHEN_REF
    pick_object: str = DEFAULT_PICK_OBJECT
    robot_xy: tuple[float, float] = DEFAULT_ROBOT_XY
    riser_height_m: float = layout.TABLE_TOP_Z
    object_height_m: float = DEFAULT_OBJECT_HEIGHT_M
    plate_size_m: tuple[float, float] = DEFAULT_PLATE_SIZE_M
    plate_thickness_m: float = DEFAULT_PLATE_THICKNESS_M
    episode_length_s: float = DEFAULT_EPISODE_LENGTH_S

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_environment_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the kitchen environment flags (pure argparse; also used by Arena's external-environment CLI path)."""
    d = KitchenOptions()
    group = parser.add_argument_group("Go2+ARX5 kitchen pick-and-place environment")
    group.add_argument(
        "--kitchen_scene",
        type=str,
        default=d.kitchen_scene,
        help="Lightwheel kitchen: 'lightwheel:<layout 1-10>:<style 1-10>' or 'lightwheel_kitchen_<layout>_<style>'",
    )
    group.add_argument(
        "--pick_object",
        type=str,
        default=d.pick_object,
        choices=LIGHTWHEEL_SDK_OBJECTS,
        help="Arena registry object fetched through the Lightwheel SDK",
    )
    group.add_argument(
        "--robot_xy",
        type=float,
        nargs=2,
        default=list(d.robot_xy),
        metavar=("X", "Y"),
        help="robot base xy on the kitchen floor (kitchen frame, identity yaw); UNVERIFIED default",
    )
    group.add_argument(
        "--riser_height_m",
        type=float,
        default=d.riser_height_m,
        help="top height of the static riser block in front of the robot (0 = objects on the floor)",
    )
    group.add_argument(
        "--object_height_m",
        type=float,
        default=d.object_height_m,
        help="pick object height proxy for the place height",
    )
    group.add_argument(
        "--plate_size_m",
        type=float,
        nargs=2,
        default=list(d.plate_size_m),
        metavar=("SX", "SY"),
    )
    group.add_argument("--plate_thickness_m", type=float, default=d.plate_thickness_m)
    group.add_argument("--episode_length_s", type=float, default=d.episode_length_s)


def options_from_namespace(ns: Any) -> KitchenOptions:
    """KitchenOptions from a parsed namespace (missing attributes keep their defaults)."""
    d = KitchenOptions()

    def get(name: str, default: Any) -> Any:
        value = getattr(ns, name, None)
        return default if value is None else value

    return KitchenOptions(
        kitchen_scene=str(get("kitchen_scene", d.kitchen_scene)),
        pick_object=str(get("pick_object", d.pick_object)),
        robot_xy=tuple(float(v) for v in get("robot_xy", d.robot_xy)),  # type: ignore[arg-type]
        riser_height_m=float(get("riser_height_m", d.riser_height_m)),
        object_height_m=float(get("object_height_m", d.object_height_m)),
        plate_size_m=tuple(float(v) for v in get("plate_size_m", d.plate_size_m)),  # type: ignore[arg-type]
        plate_thickness_m=float(get("plate_thickness_m", d.plate_thickness_m)),
        episode_length_s=float(get("episode_length_s", d.episode_length_s)),
    )


def validate_options(opts: KitchenOptions) -> list[str]:
    """Hard problems (the env cannot be built or cannot work)."""
    problems = []
    if opts.pick_object not in LIGHTWHEEL_SDK_OBJECTS:
        problems.append(
            f"pick_object {opts.pick_object!r} is not one of {list(LIGHTWHEEL_SDK_OBJECTS)}"
        )
    if opts.riser_height_m < 0:
        problems.append("riser_height_m must be >= 0")
    if not 0.01 <= opts.object_height_m <= 0.08:
        problems.append(
            "object_height_m must be in [0.01, 0.08] (the open gripper is 0.088 m wide)"
        )
    if min(opts.plate_size_m) <= 0 or opts.plate_thickness_m <= 0:
        problems.append("plate_size_m and plate_thickness_m must be > 0")
    if opts.episode_length_s <= 0:
        problems.append("episode_length_s must be > 0")
    return problems


def option_warnings(opts: KitchenOptions) -> list[str]:
    """Soft problems: runs, but outside what was verified offline."""
    warnings = []
    if opts.riser_height_m == 0:
        warnings.append(
            "riser_height_m=0: floor-level grasps were never checked against the arm's reach envelope"
        )
    elif abs(opts.riser_height_m - layout.TABLE_TOP_Z) > 0.02:
        warnings.append(
            f"riser_height_m={opts.riser_height_m} differs from the W2a table top {layout.TABLE_TOP_Z} m: the offline "
            "IK reach scan (layout.py) only covered table tops of 0.28-0.30 m"
        )
    return warnings


def kitchen_geometry(opts: KitchenOptions) -> dict[str, Any]:
    """Poses of everything we place, in the kitchen (= env-local) frame. Pure python.

    The robot keeps an identity yaw, so robot-frame offsets from ``layout.py`` just add to ``robot_xy``.
    """
    x0, y0 = opts.robot_xy
    top = opts.riser_height_m
    plate_z = top + opts.plate_thickness_m / 2.0 + layout.CUBE_REST_MARGIN_M
    pick_z = top + opts.object_height_m / 2.0 + PICK_SPAWN_CLEARANCE_M
    geometry: dict[str, Any] = {
        "robot_pos": (x0, y0, layout.ROBOT_BASE_Z),
        "robot_yaw_rad": 0.0,
        "support_top_z": top,
        "riser": None,
        "pick_range_min": (x0 + PICK_X_RANGE[0], y0 + PICK_Y_RANGE[0], pick_z),
        "pick_range_max": (x0 + PICK_X_RANGE[1], y0 + PICK_Y_RANGE[1], pick_z),
        "pick_yaw_range": PICK_YAW_RANGE,
        "pick_object_footprint": DEFAULT_PICK_FOOTPRINT_M,
        "plate_size": (
            opts.plate_size_m[0],
            opts.plate_size_m[1],
            opts.plate_thickness_m,
        ),
        "plate_range_min": (x0 + PLATE_X_RANGE[0], y0 + PLATE_Y_RANGE[0], plate_z),
        "plate_range_max": (x0 + PLATE_X_RANGE[1], y0 + PLATE_Y_RANGE[1], plate_z),
    }
    if top > 0:
        geometry["riser"] = {
            "size": (layout.TABLE_DEPTH_X, layout.TABLE_WIDTH_Y, top),
            "center": (
                x0 + layout.TABLE_CENTER[0],
                y0 + layout.TABLE_CENTER[1],
                top / 2.0,
            ),
        }
    return _round_nested(geometry)


def _round_nested(value: Any, digits: int = 6) -> Any:
    """Round every float in nested dict/tuple values (keeps tuples as tuples)."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round_nested(v, digits) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_round_nested(v, digits) for v in value)
    return value


def place_reference_pos(
    plate_pos: Sequence[float], plate_thickness_m: float, object_height_m: float
) -> tuple[float, float, float]:
    """Position to hand ``ScriptedPickPlace.reset(place_pos=...)`` so the object is released onto the plate.

    The state machine releases at ``place_pos.z + cube_size + grasp_z_offset + place_margin`` (``cube_size`` is set to
    ``object_height_m``), designed for a cube-centre ``place_pos``. The held object centre should end at plate top +
    object_height/2, so ``place_pos.z = plate_z + plate_thickness/2 - object_height/2``.
    """
    return (
        float(plate_pos[0]),
        float(plate_pos[1]),
        float(plate_pos[2]) + plate_thickness_m / 2.0 - object_height_m / 2.0,
    )


# W2a datagen settings (policy/manipulation/go2_arx5_stack/mimic_env_cfg.py), applied on top of Arena's
# PickPlaceMimicEnvCfg (whose defaults are generation_relative False / noise 0.005).
MIMIC_DATAGEN_OVERRIDES: dict[str, Any] = {
    "name": "demo_src_go2_arx5_kitchen_pick_and_place_D0",
    "generation_guarantee": True,
    "generation_keep_failed": False,
    "generation_num_trials": 10,
    "generation_select_src_per_subtask": True,
    "generation_transform_first_robot_pose": False,
    "generation_interpolate_from_last_target_pose": True,
    "generation_relative": True,
    "max_num_failures": 25,
    "seed": 1,
}
MIMIC_SUBTASK_OVERRIDES: dict[str, Any] = {
    "action_noise": 0.03,
    "num_interpolation_steps": 5,
    "num_fixed_steps": 0,
    "apply_noise_during_interpolation": False,
}
MIMIC_SUBTASK_DESCRIPTIONS = ("Grasp the object", "Place the object on the plate")


def apply_mimic_overrides(cfg: Any) -> Any:
    """Apply the W2a datagen/subtask settings to a Mimic env cfg (anything with ``datagen_config``/``subtask_configs``)."""
    for key, value in MIMIC_DATAGEN_OVERRIDES.items():
        setattr(cfg.datagen_config, key, value)
    for subtasks in cfg.subtask_configs.values():
        for index, subtask in enumerate(subtasks):
            for key, value in MIMIC_SUBTASK_OVERRIDES.items():
                setattr(subtask, key, value)
            if index < len(MIMIC_SUBTASK_DESCRIPTIONS):
                subtask.description = MIMIC_SUBTASK_DESCRIPTIONS[index]
    return cfg


# ---------------------------------------------------------------------------------------------------------
# Functions used as Isaac Lab manager terms (module level so Lab can resolve them; torch imported lazily)
# ---------------------------------------------------------------------------------------------------------
def reset_go2_arx5_to_default(
    env: Any, env_ids: Any, asset_name: str = ROBOT_SCENE_KEY
) -> None:
    """Reset ONLY the robot to its default state and its joint targets to the default pose.

    The 12 leg joints have no action term; this is what makes them hold the default pose. Lab 3.0 ``*_index`` write
    API as in ``isaaclab.envs.mdp.reset_scene_to_default(reset_joint_targets=True)`` (which would also reset every
    other rigid object to its default pose and fight the objects' own reset events).
    """
    from common.cerberus_compat.isaac_data import as_tensor

    robot = env.scene[asset_name]
    root_pose = as_tensor(robot.data.default_root_pose)[env_ids].clone()
    root_vel = as_tensor(robot.data.default_root_vel)[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
    joint_pos = as_tensor(robot.data.default_joint_pos)[env_ids].clone()
    joint_vel = as_tensor(robot.data.default_joint_vel)[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)


def object_grasped_by_gripper(
    env: Any,
    robot_cfg: Any,
    ee_frame_cfg: Any,
    object_cfg: Any,
    diff_threshold: float = layout.GRASP_DIFF_THRESHOLD,
    gripper_joint_names: Sequence[str] = GRIPPER_JOINT_NAMES,
    gripper_open_val: float = GRIPPER_OPEN_M,
    gripper_threshold: float = layout.GRIPPER_THRESHOLD,
) -> Any:
    """Subtask signal ``grasp_1``: TCP within `diff_threshold` of the object and every finger off its open value.

    Same logic as ``isaaclab_tasks.contrib.stack.mdp.object_grasped`` minus its reads of ``env.cfg.gripper_*``
    attributes (an Arena env cfg is composed by the builder and has no place for them).
    """
    import torch

    from common.cerberus_compat.isaac_data import as_tensor

    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]
    obj_pos = as_tensor(obj.data.root_pos_w)
    tcp_pos = as_tensor(ee_frame.data.target_pos_w)[:, 0, :]
    grasped = torch.linalg.vector_norm(obj_pos - tcp_pos, dim=1) < diff_threshold
    joint_ids, _ = robot.find_joints(list(gripper_joint_names), preserve_order=True)
    finger_pos = as_tensor(robot.data.joint_pos)[:, joint_ids]
    moved = (finger_pos - gripper_open_val).abs() > gripper_threshold
    return grasped & moved.all(dim=1)


# ---------------------------------------------------------------------------------------------------------
# Config builders (lazy isaaclab)
# ---------------------------------------------------------------------------------------------------------
def make_robot_cfg() -> Any:
    """Fixed-base fused robot cfg for manipulation, at the env origin and with gravity compensated."""
    import isaaclab.sim as sim_utils

    from common.cerberus_description.go2_arx5_cfg import make_go2_arx5_cfg

    robot = make_go2_arx5_cfg(fix_base=True)
    # Fixed-base relative IK has no gravity compensation; with gravity enabled the arm sags and each command
    # re-anchors on the sagged pose. Keep this override scoped to the Arena manipulation embodiment.
    rigid_props = robot.spawn.rigid_props
    rigid_props = (
        rigid_props.replace(disable_gravity=True)
        if rigid_props is not None
        else sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    )
    robot = robot.replace(spawn=robot.spawn.replace(rigid_props=rigid_props))
    return robot.replace(
        prim_path=ROBOT_PRIM_PATH,
        init_state=robot.init_state.replace(pos=layout.ROBOT_BASE_POS),
    )


def make_arm_action_cfg() -> Any:
    from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import (
        DifferentialInverseKinematicsActionCfg,
    )

    spec = copy.deepcopy(ARM_ACTION_SPEC)
    controller = DifferentialIKControllerCfg(**spec.pop("controller"))
    return DifferentialInverseKinematicsActionCfg(controller=controller, **spec)


def make_gripper_action_cfg() -> Any:
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

    return BinaryJointPositionActionCfg(**copy.deepcopy(GRIPPER_ACTION_SPEC))


def make_ee_frame_cfg() -> Any:
    from isaaclab_tasks.contrib.stack.stack_env_cfg import make_ee_frame_cfg as _make

    return _make(
        base_prim_path=ROBOT_BASE_PRIM_PATH,
        target_specs=[(EE_TARGET_PRIM_PATH, EE_TARGET_NAME, (0.0, 0.0, 0.0))],
        marker_scale=(0.05, 0.05, 0.05),
    )


# ---------------------------------------------------------------------------------------------------------
# Arena classes (lazy)
# ---------------------------------------------------------------------------------------------------------
_ARENA_CLASSES: dict[str, type] | None = None
_LAZY_NAMES = ("Go2Arx5Embodiment", "ProceduralBoxObject", ENV_FACTORY_CLASS)


def _build_arena_classes() -> dict[str, type]:
    """Define (and thereby register) the Arena classes. Needs Arena + isaaclab importable (app launched)."""
    import isaaclab.envs.mdp as mdp_isaac_lab
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
        FrameTransformerCfg,
    )
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object_base import RootedObjectBase
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PoseRange
    from isaaclab_arena_environments.example_environment_base import (
        ExampleEnvironmentBase,
    )
    from isaaclab_tasks.contrib.stack.mdp.observations import (
        ee_frame_pos,
        ee_frame_quat,
    )

    @configclass
    class Go2Arx5SceneCfg:
        """Scene additions from the embodiment: the robot and the TCP frame."""

        robot: ArticulationCfg | None = None
        ee_frame: FrameTransformerCfg | None = None

    @configclass
    class Go2Arx5ActionsCfg:
        """One relative-pose IK term on joint1..joint6, one binary gripper term. Action dim 7. No leg term."""

        arm_action: Any = None
        gripper_action: Any = None

    @configclass
    class Go2Arx5EventCfg:
        reset_robot = EventTermCfg(
            func=reset_go2_arx5_to_default,
            mode="reset",
            params={"asset_name": ROBOT_SCENE_KEY},
        )

    @configclass
    class Go2Arx5SubtaskCfg(ObsGroup):
        """Mimic reads this group as ``obs_buf['subtask_terms']``."""

        grasp_1: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class Go2Arx5ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            actions = ObsTerm(func=mdp_isaac_lab.last_action)
            joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
            eef_pos = ObsTerm(func=ee_frame_pos)
            eef_quat = ObsTerm(func=ee_frame_quat)
            # Absolute finger positions of joint7/joint8 (both open towards +0.044, see W2a env_cfg).
            gripper_pos = ObsTerm(
                func=mdp_isaac_lab.joint_pos,
                params={
                    "asset_cfg": SceneEntityCfg(
                        ROBOT_SCENE_KEY, joint_names=list(GRIPPER_JOINT_NAMES)
                    )
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()
        subtask_terms: Go2Arx5SubtaskCfg | None = None

    def make_observations_cfg(
        pick_up_object_name: str | None, concatenate: bool
    ) -> Any:
        obs = Go2Arx5ObservationsCfg()
        obs.policy.concatenate_terms = concatenate
        if pick_up_object_name is not None:
            subtasks = Go2Arx5SubtaskCfg()
            subtasks.grasp_1 = ObsTerm(
                func=object_grasped_by_gripper,
                params={
                    "robot_cfg": SceneEntityCfg(ROBOT_SCENE_KEY),
                    "ee_frame_cfg": SceneEntityCfg(EE_FRAME_NAME),
                    "object_cfg": SceneEntityCfg(pick_up_object_name),
                    "diff_threshold": layout.GRASP_DIFF_THRESHOLD,
                    "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
                    "gripper_open_val": GRIPPER_OPEN_M,
                    "gripper_threshold": layout.GRIPPER_THRESHOLD,
                },
            )
            obs.subtask_terms = subtasks
        return obs

    @register_asset
    class Go2Arx5Embodiment(EmbodimentBase):
        """Fixed-base Go2 + ARX5 (X5A) with relative-IK arm actions, a binary gripper and Mimic support."""

        name = GO2_ARX5_ARENA_NAME
        # Arena `main` (Lab 3.0 GA line) registers by `registry_name`; the older pin reads `name`. Keep both.
        registry_name = GO2_ARX5_ARENA_NAME
        tags = ["embodiment"]
        default_arm_mode = ArmMode.SINGLE_ARM

        def __init__(
            self,
            enable_cameras: bool = False,
            initial_pose: Pose | None = None,
            concatenate_observation_terms: bool = False,
            arm_mode: ArmMode | None = None,
            pick_up_object_name: str | None = None,
        ):
            super().__init__(
                enable_cameras, initial_pose, concatenate_observation_terms, arm_mode
            )
            from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
                IsaacLabArenaManagerBasedRLMimicEnv,
            )

            from policy.manipulation.go2_arx5_stack.mimic_env import (
                Go2Arx5StackMimicEnv,
            )

            class ArenaCompatibleGo2Arx5StackMimicEnv(
                IsaacLabArenaManagerBasedRLMimicEnv, Go2Arx5StackMimicEnv
            ):
                """Arena's env services (progress tracker, ArenaWorld, rest-pose recorder, episode recorders)
                plus this embodiment's Mimic methods. Arena `main` (Lab 3.0 GA) task success runs through the
                progress tracker, so the old hand-rolled shim (ArenaWorld + recorder only) is not enough."""

            self.scene_config = Go2Arx5SceneCfg(
                robot=make_robot_cfg(), ee_frame=make_ee_frame_cfg()
            )
            self.action_config = Go2Arx5ActionsCfg(
                arm_action=make_arm_action_cfg(),
                gripper_action=make_gripper_action_cfg(),
            )
            self.observation_config = make_observations_cfg(
                pick_up_object_name, self.concatenate_observation_terms
            )
            self.event_config = Go2Arx5EventCfg()
            self.mimic_env = ArenaCompatibleGo2Arx5StackMimicEnv
            self.joint_groups = dict(GO2_ARX5_EMBODIMENT.joint_groups)

        def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
            return EE_FRAME_NAME

        def get_command_body_name(self) -> str:
            return TCP_BODY_NAME

    class ProceduralBoxObject(RootedObjectBase):
        """A procedural cuboid (no USD): static collider (``rigid=False``) or kinematic/dynamic rigid body.

        Not an Arena ``Object`` on purpose: ``Object.get_contact_sensor_cfg`` of the pick object searches the
        destination's USD for a rigid body when the destination is an ``Object``; as a plain ``ObjectBase`` the
        sensor filters on this object's prim path instead.
        """

        def __init__(
            self,
            name: str,
            size: tuple[float, float, float],
            rigid: bool = True,
            kinematic: bool = True,
            mass_kg: float = 0.2,
            color: tuple[float, float, float] = (0.55, 0.42, 0.30),
            friction: float = 1.0,
            initial_pose: Pose | PoseRange | None = None,
        ):
            self.size = tuple(float(s) for s in size)
            self.kinematic = kinematic
            self.mass_kg = mass_kg
            self.color = color
            self.friction = friction
            super().__init__(
                name=name,
                prim_path=None,
                object_type=ObjectType.RIGID if rigid else ObjectType.BASE,
                tags=["object", "procedural"],
            )
            self.object_cfg = self._init_object_cfg()
            if initial_pose is not None:
                self.set_initial_pose(initial_pose)

        def _spawn(self, rigid: bool) -> Any:
            kwargs: dict[str, Any] = {}
            if rigid:
                kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=self.kinematic
                )
                if not self.kinematic:
                    kwargs["mass_props"] = sim_utils.MassPropertiesCfg(
                        mass=self.mass_kg
                    )
            return sim_utils.CuboidCfg(
                size=self.size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.friction, dynamic_friction=self.friction
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.color),
                **kwargs,
            )

        def _apply_pose(self, cfg: Any) -> Any:
            pose = self._get_initial_pose_as_pose()
            if pose is not None:
                cfg.init_state.pos = pose.position_xyz
                cfg.init_state.rot = pose.rotation_xyzw
            return cfg

        def _generate_rigid_cfg(self) -> Any:
            return self._apply_pose(
                RigidObjectCfg(prim_path=self.prim_path, spawn=self._spawn(rigid=True))
            )

        def _generate_articulation_cfg(self) -> Any:
            raise NotImplementedError("ProceduralBoxObject is never an articulation")

        def _generate_base_cfg(self) -> Any:
            return self._apply_pose(
                AssetBaseCfg(prim_path=self.prim_path, spawn=self._spawn(rigid=False))
            )

        def get_bounding_box(self) -> AxisAlignedBoundingBox:
            half = tuple(s / 2.0 for s in self.size)
            return AxisAlignedBoundingBox(
                min_point=tuple(-h for h in half), max_point=half
            )

    def set_control_rate(env_cfg: Any) -> Any:
        """``env_cfg_callback``: the W2a control rate (Arena's default is 15 Hz)."""
        env_cfg.sim.dt = SIM_DT
        env_cfg.decimation = DECIMATION
        env_cfg.sim.render_interval = RENDER_INTERVAL
        return env_cfg

    class Go2Arx5KitchenPickAndPlaceEnvironment(ExampleEnvironmentBase):
        """Lightwheel kitchen + fixed-base Go2 + ARX5 + riser, Lightwheel-SDK pick object and a plate destination."""

        name: str = ENV_NAME

        def get_env(self, args_cli: argparse.Namespace):
            from isaaclab_arena.environments.isaaclab_arena_environment import (
                IsaacLabArenaEnvironment,
            )
            from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

            from common.cerberus_worlds import registry

            opts = options_from_namespace(args_cli)
            problems = validate_options(opts)
            if problems:
                raise ValueError("; ".join(problems))
            geometry = kitchen_geometry(opts)

            # Scenes and embodiments only through the backend registry (it adds a dome light to the scene).
            backend = registry.get_backend("arena")
            backend.register_embodiment(GO2_ARX5_EMBODIMENT)
            scene = backend.load_scene(opts.kitchen_scene)
            background = next(
                a for a in scene.assets.values() if isinstance(a, Background)
            )

            pick = self.asset_registry.get_asset_by_name(opts.pick_object)()
            pick.set_initial_pose(
                PoseRange(
                    position_xyz_min=geometry["pick_range_min"],
                    position_xyz_max=geometry["pick_range_max"],
                    rpy_min=(0.0, 0.0, geometry["pick_yaw_range"][0]),
                    rpy_max=(0.0, 0.0, geometry["pick_yaw_range"][1]),
                )
            )
            plate = ProceduralBoxObject(
                PLATE_NAME,
                size=geometry["plate_size"],
                rigid=True,
                kinematic=True,
                color=(0.9, 0.9, 0.9),
            )
            plate.set_initial_pose(
                PoseRange(
                    position_xyz_min=geometry["plate_range_min"],
                    position_xyz_max=geometry["plate_range_max"],
                )
            )
            extra = [plate, pick]
            if geometry["riser"] is not None:
                extra.append(
                    ProceduralBoxObject(
                        RISER_NAME,
                        size=geometry["riser"]["size"],
                        rigid=False,
                        initial_pose=Pose(
                            position_xyz=tuple(geometry["riser"]["center"])
                        ),
                    )
                )
            scene.add_assets(extra)

            robot_pos = geometry["robot_pos"]
            embodiment = self.asset_registry.get_asset_by_name(GO2_ARX5_ARENA_NAME)(
                enable_cameras=bool(getattr(args_cli, "enable_cameras", False)),
                initial_pose=Pose(
                    position_xyz=tuple(robot_pos), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
                ),
                pick_up_object_name=pick.name,
            )

            def mimic_cfg_factory(arm_mode: Any) -> Any:
                from isaaclab_arena.tasks.pick_and_place_task import (
                    PickPlaceMimicEnvCfg,
                )

                cfg = PickPlaceMimicEnvCfg(
                    arm_mode=arm_mode,
                    pick_up_object_name=pick.name,
                    destination_location_name=plate.name,
                )
                return apply_mimic_overrides(cfg)

            task = PickAndPlaceTask(
                pick_up_object=pick,
                destination_location=plate,
                background_scene=background,
                destination_object=plate,
                episode_length_s=opts.episode_length_s,
                mimic_env_cfg_factory=mimic_cfg_factory,
            )
            return IsaacLabArenaEnvironment(
                name=ENV_NAME,
                scene=scene,
                embodiment=embodiment,
                task=task,
                env_cfg_callback=set_control_rate,
            )

        @staticmethod
        def add_cli_args(parser: argparse.ArgumentParser) -> None:
            add_environment_cli_args(parser)

    return {
        "Go2Arx5Embodiment": Go2Arx5Embodiment,
        "ProceduralBoxObject": ProceduralBoxObject,
        ENV_FACTORY_CLASS: Go2Arx5KitchenPickAndPlaceEnvironment,
    }


def get_arena_classes() -> dict[str, type]:
    """The Arena classes (``Go2Arx5Embodiment``, ``ProceduralBoxObject``, the kitchen environment factory)."""
    global _ARENA_CLASSES
    if _ARENA_CLASSES is None:
        _ARENA_CLASSES = _build_arena_classes()
    return _ARENA_CLASSES


def __getattr__(
    name: str,
) -> Any:  # PEP 562: ``--external_environment_class_path <this module>:<class>``
    if name in _LAZY_NAMES:
        return get_arena_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_go2_arx5_assets() -> dict[str, type]:
    """Register the fixed-base Go2 + ARX5 Arena embodiment (asset name ``go2_arx5``). Idempotent.

    Imports Arena lazily: call after the Isaac app has launched.
    """
    return get_arena_classes()
