"""Tests for the pure-Python inference core (no rclpy needed)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from common.cerberus_policies.command_limiter import CommandRateLimiter
from locomotion_policy.core import LocomotionCore, PolicySpec, build_limiter, zero_loco_command
from locomotion_policy.observation_assembly import SHAPE_BLIND, SHAPE_PERCEPTIVE

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pixi.toml").exists())
NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
DEFAULTS = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)


def spec(shape=SHAPE_PERCEPTIVE) -> PolicySpec:
    return PolicySpec(NAMES, DEFAULTS, action_scale=0.25, observation_shape=shape, height_scan_size=187)


def constant_policy(value=0.4):
    return lambda obs: np.full(12, value, dtype=np.float32)


def step_kwargs(shape=SHAPE_PERCEPTIVE, positions=None):
    return dict(
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.array([0.0, 0.0, 0.1]),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        joint_positions=dict(zip(NAMES, DEFAULTS)) if positions is None else positions,
        joint_velocities={n: 0.0 for n in NAMES},
        height_scan=np.zeros(187) if shape == SHAPE_PERCEPTIVE else None,
        command=zero_loco_command(),
    )


def test_perceptive_step_assembles_235_and_maps_targets_by_name():
    core = LocomotionCore(spec(), constant_policy())
    out = core.step(**step_kwargs())
    assert out.observation.shape == (235,) and out.observation.dtype == np.float32
    assert out.actions.shape == (12,)
    assert out.joint_targets["FL_hip_joint"] == pytest.approx(0.1 + 0.25 * 0.4)
    assert out.joint_targets["FR_hip_joint"] == pytest.approx(-0.1 + 0.25 * 0.4)
    assert list(out.joint_targets) == list(NAMES)


def test_blind_step_assembles_48_and_ignores_a_stray_scan():
    core = LocomotionCore(spec(SHAPE_BLIND), constant_policy())
    out = core.step(**step_kwargs(SHAPE_BLIND))
    assert out.observation.shape == (48,)
    # the blind shape has no scan term; a scan arriving anyway must not change the vector
    # (fresh core: `last_action` legitimately differs after a step)
    with_scan = LocomotionCore(spec(SHAPE_BLIND), constant_policy()).step(
        **{**step_kwargs(SHAPE_BLIND), "height_scan": np.ones(187)}
    )
    np.testing.assert_array_equal(out.observation, with_scan.observation)


def test_joint_values_are_ordered_by_name_not_position():
    core = LocomotionCore(spec(SHAPE_BLIND), lambda obs: obs[12:24])  # echoes joint_pos_rel
    scrambled = {n: v for n, v in zip(reversed(NAMES), reversed(DEFAULTS))}
    out = core.step(**{**step_kwargs(SHAPE_BLIND), "joint_positions": scrambled,
                       "joint_velocities": {n: 0.0 for n in NAMES}})
    # joint_pos_rel is zero regardless of input order (values equal defaults, by name)
    np.testing.assert_allclose(out.observation[12:24], np.zeros(12), atol=1e-6)
    with pytest.raises(ValueError, match="missing=.*FL_hip_joint"):
        bad = dict(zip(NAMES, DEFAULTS))
        del bad["FL_hip_joint"]
        core.step(**{**step_kwargs(SHAPE_BLIND), "joint_positions": bad})


def test_height_scan_size_and_action_count_are_validated():
    core = LocomotionCore(spec(), constant_policy())
    with pytest.raises(ValueError, match="height scan has 10 cells"):
        core.step(**{**step_kwargs(), "height_scan": np.zeros(10)})
    bad_policy = LocomotionCore(spec(), lambda obs: np.zeros(11))
    with pytest.raises(ValueError, match="policy returned 11 actions"):
        bad_policy.step(**step_kwargs())


def test_command_limiter_is_applied_and_reset():
    limiter = CommandRateLimiter(max_rate={"vx": 1.0, "vy": 1.0, "wz": 2.0}, max_abs={"vx": 1.0, "vy": 1.0, "wz": 1.0})
    core = LocomotionCore(spec(), constant_policy(), limiter=limiter)
    first = core.command({**zero_loco_command(), "vx": 1.0}, dt=0.1)  # adopts the clamped target
    assert first["vx"] == pytest.approx(1.0)
    second = core.command({**zero_loco_command(), "vx": -1.0}, dt=0.1)
    assert second["vx"] == pytest.approx(0.9)
    core.reset()
    assert core.command({**zero_loco_command(), "vx": -1.0}, dt=0.1)["vx"] == pytest.approx(-1.0)


def test_command_validation_and_passthrough():
    core = LocomotionCore(spec(), constant_policy())
    out = core.command({**zero_loco_command(), "vx": 5.0, "body_height": 0.3, "has_body_height": True}, dt=0.1)
    assert out["vx"] == 5.0 and out["body_height"] == 0.3 and out["has_body_height"] is True
    with pytest.raises(ValueError, match="missing"):
        core.command({"vx": 0.0}, dt=0.1)
    with pytest.raises(ValueError, match="must be a number"):
        core.command({**zero_loco_command(), "vx": "fast"}, dt=0.1)


def test_spec_validation():
    with pytest.raises(ValueError, match="joint names vs"):
        PolicySpec(NAMES, DEFAULTS[:-1])
    with pytest.raises(ValueError, match="duplicates"):
        PolicySpec(NAMES[:11] + ("FL_hip_joint",), DEFAULTS)
    with pytest.raises(ValueError, match="action_scale"):
        PolicySpec(NAMES, DEFAULTS, action_scale=0.0)
    with pytest.raises(ValueError, match="observation_shape"):
        PolicySpec(NAMES, DEFAULTS, observation_shape="depth")


def test_build_limiter_reads_the_shipped_config():
    limiter = build_limiter(REPO_ROOT / "common/cerberus_policies/config/command_limiter.yaml")
    assert dict(limiter.max_abs) == {"vx": 1.0, "vy": 1.0, "wz": 1.0}


def test_launch_config_joint_order_matches_the_contract():
    import yaml

    platforms = yaml.safe_load((REPO_ROOT / "system/cerberus_bringup/config/platforms.yaml").read_text())
    contract = yaml.safe_load((REPO_ROOT / "common/cerberus_policies/contracts/heightscan_ppo_v1.yaml").read_text())
    go2 = platforms["go2"]
    assert go2["joint_names"] == contract["joints"]["order"]
    assert [float(v) for v in go2["default_joint_pos"]] == [contract["joints"]["default_joint_pos_rad"][n] for n in go2["joint_names"]]
    assert platforms["go2"]["locomotion_modes"]["perceptive"]["backend"] == "heightscan_ppo"


def test_core_imports_without_ros():
    script = (
        "import sys\n"
        "import locomotion_policy.core\n"
        "bad = [m for m in ('rclpy', 'cerberus_msgs', 'torch', 'onnxruntime') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT}/policy/locomotion_policy", "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("BAD:"), out.stdout
