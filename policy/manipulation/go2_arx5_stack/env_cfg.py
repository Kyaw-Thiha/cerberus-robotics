"""Fixed-base Go2 + ARX5 (X5A) "put cube_2 on cube_1" task, relative-IK actions (W2a, Isaac Lab 3.0 line).

Imports isaaclab at module level: only import this after the Isaac app has launched (it is loaded lazily through
the gymnasium `env_cfg_entry_point` string registered in `register.py`).

Based on (Isaac Lab tag `v3.0.0-EA`):
- `isaaclab_tasks/contrib/stack/stack_env_cfg.py` (scene/obs/termination/event structure, `PhysicsCfg`,
  `make_ee_frame_cfg`)
- `isaaclab_tasks/contrib/stack/config/franka/stack_joint_pos_env_cfg.py` and `stack_ik_rel_env_cfg.py`
  (gripper bookkeeping attributes, `DifferentialInverseKinematicsActionCfg`, `BinaryJointPositionActionCfg`)
- `isaaclab_tasks/contrib/stack/mdp/{observations,terminations,stack_events}.py` (reused as is)

Differences from the Franka task: two cubes (no cube_3); a procedural table and procedural cubes (no Nucleus assets);
the robot is the fused Go2 + ARX5 with its root link fixed; the IK action drives `joint1..joint6` on the `eef_link`
TCP body (body offset zero); NO leg action term: the 12 leg joints keep the default-pose target written by
`reset_scene_to_default(reset_joint_targets=True)`; the success term allows for the 0.044 m prismatic finger limit.

Geometry and its derivation: see `layout.py`.

UNVERIFIED (nothing here has run in Isaac Sim; check on the first pod session, in this order):
- `isaaclab_tasks.contrib.stack` is the v3.0.0-EA home of the stack task (it was `manager_based/manipulation/stack` on
  2.3.x); the GA release may move it again.
- `make_go2_arx5_cfg(fix_base=True)` really gives `Articulation.is_fixed_base == True`:
  `DifferentialInverseKinematicsAction` indexes the Jacobian with `body_idx - 1` only in that case.
- The robot's default init rotation is identity (we only override `pos`), as the IK/Mimic frames assume.
- Grasp physics: cube mass/friction, the provisional finger PD gains (`GRIPPER_ACTUATOR_GAINS`), a top-down grasp with
  the TCP at the cube centre, and fingers reaching 0.044 m +- `GRIPPER_OPEN_ATOL` after release (success term).
- Legs: with no leg action term they hold the default pose only through the joint targets written by
  `reset_scene_to_default(reset_joint_targets=True)`; the fixed base keeps the 12 leg joints loaded only by gravity.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.mdp import stack_events
from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg, make_ee_frame_cfg

from common.cerberus_compat.isaac_data import as_tensor
from common.cerberus_description.go2_arx5_cfg import make_go2_arx5_cfg
from common.cerberus_description.go2_arx5_spec import (
    ARX5_ARM_JOINTS,
    GRIPPER_CLOSED_M,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_M,
    TCP_BODY_NAME,
)
from policy.manipulation.go2_arx5_stack import layout

##
# Observation functions (two-cube versions of the stack task's three-cube ones)
##


def cube_positions_env_frame(
    env,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Cube centres in the env frame (world minus env origin), cube_1 then cube_2. Shape (num_envs, 6)."""
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    origins = env.scene.env_origins
    return torch.cat(
        (
            as_tensor(cube_1.data.root_pos_w) - origins,
            as_tensor(cube_2.data.root_pos_w) - origins,
        ),
        dim=1,
    )


def cube_orientations_world_frame(
    env,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Cube quaternions (Isaac Lab's native order: XYZW on the 3.0 line), cube_1 then cube_2. Shape (num_envs, 8)."""
    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    return torch.cat(
        (as_tensor(cube_1.data.root_quat_w), as_tensor(cube_2.data.root_quat_w)), dim=1
    )


##
# Scene
##

_CUBE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


def _cube_cfg(
    name: str, pos: tuple[float, float, float], color: tuple[float, float, float]
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos, rot=(0.0, 0.0, 0.0, 1.0)
        ),  # identity, XYZW
        spawn=sim_utils.CuboidCfg(
            size=(layout.CUBE_SIZE,) * 3,
            rigid_props=_CUBE_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=layout.CUBE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=layout.CUBE_FRICTION,
                dynamic_friction=layout.CUBE_FRICTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class Go2Arx5StackSceneCfg(InteractiveSceneCfg):
    """Ground, light, a solid table block in front of the fixed robot, two cubes, the robot and the TCP frame."""

    # populated in the env cfg's __post_init__
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=layout.TABLE_CENTER),
        spawn=sim_utils.CuboidCfg(
            size=layout.TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),  # no rigid body API: a static collider
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.42, 0.30)
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # cube_1 (blue) is the base; cube_2 (red) is picked and placed on it. Same naming as the Franka task.
    cube_1 = _cube_cfg("Cube_1", layout.CUBE_1_DEFAULT_POS, (0.1, 0.2, 0.9))
    cube_2 = _cube_cfg("Cube_2", layout.CUBE_2_DEFAULT_POS, (0.9, 0.1, 0.1))


##
# MDP
##


@configclass
class ActionsCfg:
    """One relative-pose IK term on joint1..joint6, one binary gripper term. Action dim 7. No leg term."""

    arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_positions = ObsTerm(func=cube_positions_env_frame)
        cube_orientations = ObsTerm(func=cube_orientations_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # Absolute finger positions (m) of joint7/joint8. Not `mdp.gripper_pos`, which negates the second finger
        # for the Franka's mirrored joint convention; here both fingers open towards +0.044.
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(GRIPPER_JOINT_NAMES)
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Mimic reads this group as `obs_buf['subtask_terms']`."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
                "diff_threshold": layout.GRASP_DIFF_THRESHOLD,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": layout.CUBE_DROP_MIN_Z,
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": layout.CUBE_DROP_MIN_Z,
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )

    # cube_2 stacked on cube_1 with the gripper released. `cubes_stacked` with cube_3_cfg=None checks: cube_1 below
    # cube_2 (z), xy distance < xy_threshold, z distance within height_threshold of `height_diff` (= cube side),
    # and every finger within `atol` of gripper_open_val.
    success = DoneTerm(
        func=mdp.cubes_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "cube_3_cfg": None,
            "xy_threshold": layout.STACK_XY_THRESHOLD,
            "height_threshold": layout.STACK_HEIGHT_THRESHOLD,
            "height_diff": layout.CUBE_SIZE,
            "atol": layout.GRIPPER_OPEN_ATOL,
            "rtol": 0.0,
        },
    )


@configclass
class EventCfg:
    # Everything back to the default (stowed arm, default leg pose, open gripper) incl. joint targets: the leg joints
    # have no action term, so this is what keeps them holding the default pose.
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_cube_positions = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": layout.CUBE_X_RANGE,
                "y": layout.CUBE_Y_RANGE,
                "z": (layout.CUBE_Z, layout.CUBE_Z),
                "yaw": layout.CUBE_YAW_RANGE,
            },
            "min_separation": layout.CUBE_MIN_SEPARATION,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2")],
        },
    )


@configclass
class Go2Arx5StackEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base Go2 + ARX5 stack task with relative-pose IK actions."""

    scene: Go2Arx5StackSceneCfg = Go2Arx5StackSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        # general settings (as in the stock stack task)
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100 Hz physics, 20 Hz control
        self.sim.render_interval = 2
        self.sim.physics = PhysicsCfg()

        # Robot: fixed base at the env origin (identity rotation), root link fixed.
        # Gravity is disabled on the robot (as the stock Franka stack task does, and as the real ARX controller
        # compensates arm gravity): with relative IK commands and the soft ARX5 PD gains (joint4: 30 N*m/rad) the arm
        # otherwise sags continuously (pod 2026-10-05: joint4 -0.97 rad in 60 zero-action steps, IK tracking error
        # > 0.25 m everywhere), because each relative command re-anchors the target on the sagged pose.
        robot = make_go2_arx5_cfg(fix_base=True)
        spawn = robot.spawn
        rigid_props = spawn.rigid_props
        rigid_props = (
            rigid_props.replace(disable_gravity=True)
            if rigid_props is not None
            else sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        )
        # Pod tuning: the 1,000 N/m provisional actuator stopped at 0.023 m/finger when pinching a 0.04 m cube,
        # then the cube slipped during lift (trace at close_pause_steps=40). Raise stiffness to reach the existing
        # 50 N effort cap and improve holding force; the force limit itself is unchanged.
        actuators = dict(robot.actuators)
        actuators["arx5_gripper"] = actuators["arx5_gripper"].replace(stiffness=5.0e3)
        self.scene.robot = robot.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=spawn.replace(rigid_props=rigid_props),
            actuators=actuators,
            init_state=robot.init_state.replace(pos=layout.ROBOT_BASE_POS),
        )

        # Actions. joint_names are explicit: DiffIK resolves them with `find_joints` (full-match regex), and the
        # fingers must not be in the arm group. body_name must match exactly one body: the TCP frame body.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=list(ARX5_ARM_JOINTS),
            body_name=TCP_BODY_NAME,
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=layout.IK_ACTION_SCALE,
            body_offset=None,  # eef_link is itself the TCP body (GO2_ARX5_TCP_OFFSET_XYZ from link6)
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(GRIPPER_JOINT_NAMES),
            open_command_expr={name: GRIPPER_OPEN_M for name in GRIPPER_JOINT_NAMES},
            close_command_expr={name: GRIPPER_CLOSED_M for name in GRIPPER_JOINT_NAMES},
        )
        # UNVERIFIED: finger PD gains hold a 0.04 m cube without crushing/slipping (provisional GRIPPER_ACTUATOR_GAINS).
        # Read by mdp.object_grasped / cubes_stacked / randomize_joint_by_gaussian_offset.
        self.gripper_joint_names = list(GRIPPER_JOINT_NAMES)
        self.gripper_open_val = GRIPPER_OPEN_M
        self.gripper_threshold = layout.GRIPPER_THRESHOLD

        # TCP frame: target frame = the eef_link body, source = the fixed root body.
        # Prim paths come from the URDF-converted fused USD (layout.BASE_PRIM_RELATIVE_PATH / EEF_PRIM_RELATIVE_PATH).
        eef_prim_path = f"{{ENV_REGEX_NS}}/Robot/{layout.EEF_PRIM_RELATIVE_PATH}"
        self.scene.ee_frame = make_ee_frame_cfg(
            base_prim_path=f"{{ENV_REGEX_NS}}/Robot/{layout.BASE_PRIM_RELATIVE_PATH}",
            target_specs=[(eef_prim_path, "end_effector", (0.0, 0.0, 0.0))],
            marker_scale=(0.05, 0.05, 0.05),
        )
