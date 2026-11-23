"""Arm embodiment: the 20-DoF body keeps the 12-joint contract order, and the arm events clamp to the limits.

Pure-python checks only (no Isaac): the joint-order rule and the offset sampler are what could silently corrupt a
run, so they are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.cerberus_eval.scene_assertions import check_joint_order_embodiment

CONTRACT = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]


def test_legs_only_body_still_requires_exact_equality():
    assert check_joint_order_embodiment(CONTRACT, "go2") == []
    swapped = list(CONTRACT)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert check_joint_order_embodiment(swapped, "go2")


def test_arm_body_accepts_the_twenty_dof_order():
    actual = CONTRACT + ARM
    assert check_joint_order_embodiment(actual, "go2_arx5") == []


def test_arm_body_rejects_missing_legs_or_arm():
    assert check_joint_order_embodiment(CONTRACT, "go2_arx5")          # no arm joints
    assert check_joint_order_embodiment(ARM + CONTRACT[:-1], "go2_arx5")  # one leg missing


def test_arm_body_accepts_a_different_asset_order():
    """The fused URDF emits joints per leg; presence is what the check can verify (the env cfg pins the policy
    order by name with preserve_order=True)."""
    per_leg = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
               "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
    assert check_joint_order_embodiment(per_leg + ARM, "go2_arx5") == []


def test_offset_bounds_clamp_to_joint_limits():
    from common.cerberus_description.arm_hold_math import arm_offset_bounds

    lows, highs = arm_offset_bounds([0.0], 0.5, limits=[(-0.2, 0.3)])
    assert lows == [-0.2] and highs == [0.3]
