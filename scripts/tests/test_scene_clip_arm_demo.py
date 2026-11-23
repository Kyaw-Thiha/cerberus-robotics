"""The scripted arm motion for the showcase clip: phases, ranges and a deterministic cycle (no Isaac)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: F401  (kept for the runner's fixture discovery)
from scene_clip import arm_demo_targets

STOWED = {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
          "joint7": 0.044, "joint8": 0.044}


def test_starts_and_ends_stowed():
    assert arm_demo_targets(0.0, stowed=STOWED) == STOWED
    assert arm_demo_targets(4.0 * 0.75, stowed=STOWED) == STOWED


def test_reaches_forward_and_turns_the_wrist_when_fully_extended():
    target = arm_demo_targets(4.0 * 0.45 - 1e-6, stowed=STOWED)
    assert target["joint1"] > STOWED["joint1"] + 0.4     # shoulder forward
    assert target["joint2"] < STOWED["joint2"] - 0.5     # lifted
    assert target["joint5"] > STOWED["joint5"] + 0.7     # wrist turned


def test_gripper_closes_then_opens():
    closed = arm_demo_targets(4.0 * 0.52, stowed=STOWED)
    assert closed["joint7"] == 0.0 and closed["joint8"] == 0.0
    opened = arm_demo_targets(4.0 * 0.65, stowed=STOWED)
    assert opened["joint7"] == STOWED["joint7"] and opened["joint8"] == STOWED["joint8"]


def test_cycle_is_deterministic_and_bounded():
    for t in (0.1, 0.9, 1.7, 2.5, 3.3):
        a = arm_demo_targets(t, stowed=STOWED)
        b = arm_demo_targets(t, stowed=STOWED)
        assert a == b
        assert abs(a["joint1"]) <= 0.5 and abs(a["joint2"]) <= 0.7
