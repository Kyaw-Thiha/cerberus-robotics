"""Spec for the fused Unitree Go2 + ARX5 (X5) robot ("go2_arx5"). Pure python.

Sources:
- LeggedManip_Lab go2_arx5 USD inspection: arm mount pose and PD gains.
- arx5-sdk ``X5.urdf``: arm joint limits (the X5A URDF only has +-10 rad placeholders).
- ARX's official ``X5A.urdf`` (github.com/ARXroboticsX/ARX_Model, ``X5/X5A``): link masses, the two prismatic
  finger joints (``joint7``/``joint8``) and the geometry the TCP offset is derived from.
- LeggedManip_Lab MuJoCo model ctrlrange: arm effort limits.

Decision: the fused robot keeps the stock Isaac Lab Go2 body
(``UNITREE_GO2_CFG``, root body ``base``) so legs-only baselines stay
comparable. The repo's ROS xacro Go2 is NOT used as the physics body.

Arm role: legs-only RL policy + IK-driven arm, so the arm
uses simple implicit PD actuators. The gripper is two independent prismatic
fingers (no mimic joint) driven by their own implicit PD actuator group; the
whole robot is 20 DoF: 12 legs + joint1..6 + joint7, joint8.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from os import environ
from pathlib import Path

from common.cerberus_description.embodiment import EmbodimentCfg, compose_embodiments

GO2_ROOT_BODY = "base"
GO2_LEG_JOINTS: tuple[str, ...] = tuple(
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
)

ARX5_ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7))
# X5A gripper: two independent prismatic fingers, mirrored axes, no <mimic> tag in the URDF.
GRIPPER_JOINT_NAMES: tuple[str, ...] = ("joint7", "joint8")
GRIPPER_OPEN_M = 0.044  # per finger
GRIPPER_CLOSED_M = 0.0  # per finger
ARX5_FINGER_LINKS: tuple[str, ...] = ("link7", "link8")
ARX5_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    *ARX5_FINGER_LINKS,
    "eef_link",
)
# Tool-centre-point body: a fixed child of link6 added by the URDF patch (X5A has none).
TCP_BODY_NAME = "eef_link"
ARX5_EEF_BODY = TCP_BODY_NAME

# Arm joint limits taken from the SDK X5.urdf. X5A.urdf only has +-10 rad placeholders for joint1..6, which
# the URDF patch replaces with these.
ARX5_JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "joint1": (-10.0, 10.0),
    "joint2": (0.0, 3.14),
    "joint3": (0.0, 3.14),
    "joint4": (-1.57079632679, 1.57079632679),
    "joint5": (-1.67, 1.67),
    "joint6": (-1.57, 1.57),
}
# PROVISIONAL: the SDK X5.urdf has 10 rad/s on two joints and a 1000 placeholder on the rest; one conservative
# value for all six, replacing the X5A 1000 placeholder. Not from the ARX datasheet.
ARX5_JOINT_VELOCITY_LIMIT_RAD_S = 10.0


@dataclass(frozen=True)
class FingerSpec:
    """One prismatic gripper finger (X5A.urdf joint7 / joint8)."""

    link: str
    lower_m: float
    upper_m: float
    axis: tuple[float, float, float]
    mass_kg: float
    # PROVISIONAL: X5A.urdf has effort 100 / velocity 1000 placeholders; effort follows the gripper actuator
    # effort (50 N), velocity is a guess. Check against the ARX datasheet before relying on grip force.
    effort_n: float = 50.0
    velocity_m_s: float = 1.0


# joint7 slides along +y, joint8 along -y, so q=0 is closed (fingers touching at y=0 in the link6 frame) and
# q=0.044 is open for both.
ARX5_FINGER_SPECS: dict[str, FingerSpec] = {
    "joint7": FingerSpec("link7", 0.0, 0.044, (0.0, 1.0, 0.0), 0.0648),
    "joint8": FingerSpec("link8", 0.0, 0.044, (0.0, -1.0, 0.0), 0.0648),
}

# X5A.urdf link masses (kg); eef_link is massless there and excluded. Total with fingers: 3.454 kg
# (the fingers use 0.064798 kg each in the URDF, rounded to 0.0648 in ARX5_FINGER_SPECS).
ARX5_LINK_MASSES_KG: dict[str, float] = {
    "base_link": 0.43995,
    "link1": 0.066982,
    "link2": 1.0795,
    "link3": 0.54534,
    "link4": 0.11714,
    "link5": 0.63488,
    "link6": 0.44089,
    "link7": 0.064798,
    "link8": 0.064798,
}
# The old SDK X5.urdf (gripper baked into link6; no finger joints), kept for the X5 variant only.
ARX5_SDK_X5_LINK_MASSES_KG: dict[str, float] = {
    "base_link": 0.428,
    "link1": 0.064,
    "link2": 1.165,
    "link3": 0.546,
    "link4": 0.080971,
    "link5": 0.655,
    "link6": 0.581,
}

# Fingertip midpoint in the link6 frame (metres), derived from X5A geometry. Derivation (recomputed by
# test_go2_arx5_spec.py::test_tcp_offset_matches_x5a_geometry):
#   - joint7/joint8 origins in X5A.urdf, expressed in link6: (0.08657, +0.024896, -0.0002436) and
#     (0.08657, -0.0249, -0.00024366), no rotation, finger joints at q=0 (closed);
#   - the fingertip pad of each finger is the part of its binary STL (link7.STL / link8.STL, in metres, in the
#     finger link frame) with x >= x_max - 0.015 (x_max = 0.071, so x in [0.056, 0.071], 47 vertices);
#   - TCP = midpoint of the two pad bounding-box centres, each shifted by its joint origin:
#       pad centres in link6: (0.150185, +0.001498, -0.00003) and (0.150185, -0.001502, -0.00003)
#       midpoint: (0.150185, -0.000002, -0.00003) -> rounded to 0.1 mm: (0.1502, 0.0, 0.0).
# The pads touch at y=0 when closed, so the TCP is on the grasp centre line, ~7.1 cm past the link6 mesh end.
GO2_ARX5_TCP_OFFSET_XYZ: tuple[float, float, float] = (0.1502, 0.0, 0.0)
# Constants the derivation test recomputes from (kept here so the number and its inputs live together).
ARX5_FINGER_JOINT_ORIGINS_LINK6: dict[str, tuple[float, float, float]] = {
    "joint7": (0.08657, 0.024896, -0.0002436),
    "joint8": (0.08657, -0.0249, -0.00024366),
}
ARX5_FINGER_PAD_DEPTH_M = 0.015

# Stowed arm (same as LeggedManip_Lab's default; note joint2/joint3 lower limit is 0.0) with the gripper open.
ARX5_STOWED_JOINT_POS: dict[str, float] = {
    **{j: 0.0 for j in ARX5_ARM_JOINTS},
    **{j: GRIPPER_OPEN_M for j in GRIPPER_JOINT_NAMES},
}


@dataclass(frozen=True)
class ArmActuatorGains:
    stiffness: float
    damping: float
    armature: float
    effort_limit: float


# Stiffness/damping from LeggedManip_Lab. Effort limits (N*m) are taken from
# LeggedManip_Lab's MuJoCo ctrlrange; the ARX datasheet was NOT verified.
ARX5_ACTUATOR_GAINS: dict[str, ArmActuatorGains] = {
    "joint1": ArmActuatorGains(50.0, 3.0, 0.01, 20.0),
    "joint2": ArmActuatorGains(50.0, 2.0, 0.01, 20.0),
    "joint3": ArmActuatorGains(80.0, 3.0, 0.01, 20.0),
    "joint4": ArmActuatorGains(30.0, 3.0, 0.01, 7.0),
    "joint5": ArmActuatorGains(30.0, 2.5, 0.01, 5.0),
    "joint6": ArmActuatorGains(20.0, 1.0, 0.01, 5.0),
}


@dataclass(frozen=True)
class GripperActuatorGains:
    stiffness: float  # N/m
    damping: float  # N*s/m
    effort_limit: float  # N


# PROVISIONAL: untuned starting values (stiffness ~1e3 N/m, damping ~1e2 N*s/m, 50 N), not from a datasheet.
# Tune on a pod once the gripper test phase of spawn_hold_go2_arx5.py gives real settling data.
GRIPPER_ACTUATOR_GAINS = GripperActuatorGains(
    stiffness=1.0e3, damping=1.0e2, effort_limit=50.0
)

# Arm base_link origin in the Go2 `base` frame. Decision: 0.07 m. With z=0 the arm base sits fully inside the Go2
# body shell; LeggedManip_Lab uses 0.07 (USD) and 0.06 (MuJoCo). The real mounting-plate thickness was not checked
# in hardware docs.
GO2_ARX5_MOUNT_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.07)
GO2_ARX5_MOUNT_RPY_DEG: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Prim paths in the fused USD, relative to the robot default prim (`/go2_arx5`, i.e. under `{ENV_REGEX_NS}/Robot/`
# once spawned). The USD is produced by `UrdfConverter` from the kinematic chain of the merged URDF (one prim per
# link, nested along the chain); verified by the convert script's conformance check and by the offline test
# `test_go2_arx5_spec.py::test_default_fused_usd_structure`.
GO2_ARX5_BASE_PRIM_REL_PATH = "Geometry/base"
GO2_ARX5_TCP_PRIM_REL_PATH = (
    "Geometry/base/base_link/link1/link2/link3/link4/link5/link6/eef_link"
)

GO2_ARX5_USD_ENV = "CERBERUS_GO2_ARX5_USD"


def default_fused_usd_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "assets"
        / "go2_arx5"
        / "go2_arx5.usd"
    )


def fused_usd_path() -> Path:
    override = environ.get(GO2_ARX5_USD_ENV)
    return Path(override) if override else default_fused_usd_path()


# Name of the effort-limit field on ImplicitActuatorCfg, by Isaac Lab line: `effort_limit_sim` (2.3.2) and
# `joint_effort_limit` (3.0 line). Detected from the dataclass fields, never from a version string.
ACTUATOR_EFFORT_LIMIT_FIELDS: tuple[str, ...] = (
    "effort_limit_sim",
    "joint_effort_limit",
)


def implicit_actuator_kwargs(
    cfg_cls: type,
    joint_names_expr: Sequence[str],
    stiffness: float,
    damping: float,
    effort_limit: float,
    armature: float | None = None,
) -> dict:
    """Constructor kwargs for an ImplicitActuatorCfg-like dataclass on either Isaac Lab line.

    Picks the effort-limit field name that `dataclasses.fields(cfg_cls)` offers (first match in
    ACTUATOR_EFFORT_LIMIT_FIELDS); `armature` is only passed when given and the class has it.
    Raises TypeError if `cfg_cls` is not a dataclass and ValueError if no effort-limit field exists.
    """
    if not is_dataclass(cfg_cls):
        raise TypeError(f"{cfg_cls!r} is not a dataclass; cannot inspect its fields")
    names = {f.name for f in fields(cfg_cls)}
    effort_field = next((n for n in ACTUATOR_EFFORT_LIMIT_FIELDS if n in names), None)
    if effort_field is None:
        raise ValueError(
            f"{cfg_cls.__name__} has none of the effort-limit fields {list(ACTUATOR_EFFORT_LIMIT_FIELDS)}"
        )
    kwargs = {
        "joint_names_expr": list(joint_names_expr),
        "stiffness": stiffness,
        "damping": damping,
        effort_field: effort_limit,
    }
    if armature is not None and "armature" in names:
        kwargs["armature"] = armature
    return kwargs


GO2_ARX5_ARTICULATION_ENTRY_POINT = (
    "common.cerberus_description.go2_arx5_cfg:GO2_ARX5_CFG"
)

GO2_EMBODIMENT = EmbodimentCfg(
    "go2", "isaaclab_assets.robots.unitree:UNITREE_GO2_CFG", {"legs": GO2_LEG_JOINTS}
)
# The arm is only ever spawned fused, hence the fused entry point.
ARX5_EMBODIMENT = EmbodimentCfg(
    "arx5",
    GO2_ARX5_ARTICULATION_ENTRY_POINT,
    {"arm": ARX5_ARM_JOINTS, "gripper": GRIPPER_JOINT_NAMES},
)
GO2_ARX5_EMBODIMENT = compose_embodiments(
    "go2_arx5", GO2_EMBODIMENT, ARX5_EMBODIMENT, GO2_ARX5_ARTICULATION_ENTRY_POINT
)
