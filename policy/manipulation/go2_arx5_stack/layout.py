"""Geometry and shared constants of the W2a fixed-base pick-and-place scene. Pure python (no isaaclab).

Frame convention: the env origin is the Go2 `base` origin projected to the ground (x forward, z up). The fixed
robot base sits at (0, 0, ROBOT_BASE_Z) with identity rotation, which Isaac Lab's stack observations
(`ee_frame_pos` = world minus env origin, no base rotation) and Mimic's pose helpers rely on.

Where the numbers come from (none of this is verified in simulation; see the UNVERIFIED notes):

- Standing height. Go2 thigh/calf length 0.213 m, foot sphere radius 0.022 m, default leg pose thigh 0.8 rad
  (front) / 1.0 rad (rear), calf -1.5 rad. Hip-to-sole height = L*cos(thigh) + L*cos(thigh + calf) + r, i.e.
  0.333 m front and 0.324 m rear. The base is fixed at 0.335 m so the feet rest on (or just above) the ground.
- Arm base. `GO2_ARX5_MOUNT_XYZ` puts the ARX `base_link` 0.07 m above the Go2 `base`, so 0.405 m above ground.
- Reach envelope. The X5A kinematics (joint origins/axes from `X5A.urdf`, limits from `ARX5_JOINT_LIMITS_RAD`,
  TCP at `GO2_ARX5_TCP_OFFSET_XYZ`) were scanned offline with a damped-least-squares numerical IK (random restarts,
  position + orientation, no collision model): with the arm base at z = 0.405 m and a top-down gripper
  (`GRASP_PITCH_RAD`), every point with x in [0.30, 0.46] m and |y| <= 0.14 m was reachable at all four
  waypoint heights of the pick-and-place (grasp, grasp + hover, place + hover, release) for table tops 0.28-0.30 m;
  x = 0.50 m was not. A horizontal (stowed-orientation) grasp was only reachable for x >= 0.55 m at this height,
  which is why the gripper approaches from above.
- Clearance. Go2 front feet reach x = hip 0.193 - 0.016 + 0.022 = about 0.20 m; the table front edge is 0.05 m
  beyond `GO2_FRONT_EXTENT_X`.
"""

from __future__ import annotations

import math

from common.cerberus_description.go2_arx5_spec import (
    GO2_ARX5_BASE_PRIM_REL_PATH,
    GO2_ARX5_MOUNT_XYZ,
    GO2_ARX5_TCP_PRIM_REL_PATH,
)

# --- robot placement -----------------------------------------------------------------------------------------
ROBOT_BASE_Z = 0.335
ROBOT_BASE_POS: tuple[float, float, float] = (0.0, 0.0, ROBOT_BASE_Z)
ARM_BASE_Z = ROBOT_BASE_Z + GO2_ARX5_MOUNT_XYZ[2]
# Prim paths of the root body and the TCP body relative to the robot prim (single source: the spec).
BASE_PRIM_RELATIVE_PATH = GO2_ARX5_BASE_PRIM_REL_PATH
EEF_PRIM_RELATIVE_PATH = GO2_ARX5_TCP_PRIM_REL_PATH

# Go2 geometry used for the standing-height derivation and the table clearance (metres / radians).
GO2_LINK_LENGTH = 0.213
GO2_FOOT_RADIUS = 0.022
GO2_DEFAULT_THIGH = {"front": 0.8, "rear": 1.0}
GO2_DEFAULT_CALF = -1.5
GO2_FRONT_EXTENT_X = 0.21  # front foot sole + margin, forward of the Go2 base origin


def go2_standing_height(thigh: float, calf: float = GO2_DEFAULT_CALF) -> float:
    """Hip-to-ground height of a Go2 leg with the given thigh/calf angles (foot sphere resting on the ground)."""
    return (
        GO2_LINK_LENGTH * math.cos(thigh)
        + GO2_LINK_LENGTH * math.cos(thigh + calf)
        + GO2_FOOT_RADIUS
    )


# --- table -----------------------------------------------------------------------------------------------------
TABLE_TOP_Z = 0.30
TABLE_FRONT_X = 0.26
TABLE_DEPTH_X = 0.34
TABLE_WIDTH_Y = 0.60
TABLE_SIZE: tuple[float, float, float] = (
    TABLE_DEPTH_X,
    TABLE_WIDTH_Y,
    TABLE_TOP_Z,
)  # solid block on the ground
TABLE_CENTER: tuple[float, float, float] = (
    TABLE_FRONT_X + TABLE_DEPTH_X / 2.0,
    0.0,
    TABLE_TOP_Z / 2.0,
)
TABLE_MIN_CLEARANCE_TO_GO2_X = 0.04

# --- cubes -----------------------------------------------------------------------------------------------------
CUBE_SIZE = (
    0.04  # fits the open gripper (2 * 0.044 m finger travel) with 2+ cm slack per side
)
CUBE_MASS_KG = 0.05
CUBE_FRICTION = 5.0  # tuned after 0/21 picks: cubes slipped from the closed X5A fingers during vertical lift.
CUBE_REST_MARGIN_M = (
    0.001  # spawn this far above the table to avoid initial penetration
)
CUBE_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + CUBE_REST_MARGIN_M
CUBE_X_RANGE: tuple[float, float] = (0.32, 0.44)
CUBE_Y_RANGE: tuple[float, float] = (-0.12, 0.12)
CUBE_YAW_RANGE: tuple[float, float] = (
    -0.3,
    0.3,
)  # rad; modest so the top-down gripper straddles the cube
CUBE_MIN_SEPARATION = 0.10
CUBE_DROP_MIN_Z = TABLE_TOP_Z - 0.05
# Default (pre-randomization) cube positions in the env frame.
CUBE_1_DEFAULT_POS: tuple[float, float, float] = (0.38, -0.08, CUBE_Z)
CUBE_2_DEFAULT_POS: tuple[float, float, float] = (0.38, 0.08, CUBE_Z)

# --- reach envelope for a top-down grasp (offline IK scan, see module docstring) ---------------------------------
GRASP_PITCH_RAD = (
    math.pi / 2.0
)  # rotation of the stowed TCP frame about the world y axis (x axis -> -z)
TOPDOWN_REACH_X: tuple[float, float] = (0.30, 0.46)
TOPDOWN_REACH_ABS_Y = 0.14
REACH_MARGIN_M = 0.02  # cube range must stay at least this far inside the envelope
HOVER_HEIGHT = 0.10

# --- control / task constants shared by env cfg, Mimic env and the scripted recorder -----------------------------
IK_ACTION_SCALE = 0.5
GRIPPER_THRESHOLD = (
    0.005  # `object_grasped`: finger must be this far from the open value
)
GRIPPER_OPEN_ATOL = (
    0.002  # success term: fingers must be within this of GRIPPER_OPEN_M (released)
)
GRASP_DIFF_THRESHOLD = 0.05  # `object_grasped`: max TCP-to-cube-centre distance
STACK_XY_THRESHOLD = 0.04
STACK_HEIGHT_THRESHOLD = 0.005
SUBTASK_GRASP_SIGNAL = "grasp_1"
MIMIC_EEF_NAME = "go2_arx5"
ACTION_DIM = 7  # 3 position + 3 rotation (relative IK) + 1 gripper (<0 closes)
