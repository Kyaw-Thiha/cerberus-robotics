"""Tests for the Phase 0.5 gate arithmetic (policy/locomotion/adversarial_robustness/attack_math.py).

Loaded by file path: the module itself is pure python, but importing it through
`policy.locomotion.adversarial_robustness` would pull in the package chain that needs Isaac Lab.

These tests pin the definitions the gate depends on: what "combined magnitude" means, that the
rate cap really binds, that the reward penalises a surviving robot, and that the pass/fail boundary
is strict (a boundary exactly at the trained range is a FAIL, not a pass).
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "attack_math.py"
_spec = importlib.util.spec_from_file_location("_attack_math_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod  # dataclasses resolve their module through sys.modules during exec
_spec.loader.exec_module(_mod)


def bounds(**overrides):
    base = {"force_peak_n": 120.0, "command_peak_mps": math.hypot(1.0, 1.0), "rate_limit_fraction": 0.1}
    base.update(overrides)
    return _mod.PerturbationBounds(**base)


class TestBoundsFromConfig:
    def test_derives_peak_force_and_command_envelope(self):
        platform_cfg = SimpleNamespace(push_peak_force_n=120.0, push_peak_torque_nm=18.0)
        ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
        out = _mod.PerturbationBounds.from_config(platform_cfg, ranges)
        assert out.force_peak_n == 120.0
        assert out.command_peak_mps == pytest.approx(math.hypot(1.0, 1.0))
        assert out.yaw_peak_rad_s == pytest.approx(1.0)
        assert out.force_peak_torque_nm == 18.0

    def test_asymmetric_ranges_use_the_corner(self):
        platform_cfg = SimpleNamespace(push_peak_force_n=80.0)
        ranges = SimpleNamespace(lin_vel_x=(0.0, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-2.0, 1.0))
        out = _mod.PerturbationBounds.from_config(platform_cfg, ranges)
        assert out.command_peak_mps == pytest.approx(math.hypot(1.5, 0.5))
        assert out.yaw_peak_rad_s == 2.0

    def test_missing_peak_force_is_an_error_not_a_default(self):
        with pytest.raises(ValueError, match="push_peak_force_n"):
            _mod.PerturbationBounds.from_config(SimpleNamespace(), SimpleNamespace(lin_vel_x=(-1, 1), lin_vel_y=(-1, 1)))


class TestBoundsValidation:
    def test_rejects_non_positive_force(self):
        with pytest.raises(ValueError, match="force_peak_n"):
            _mod.PerturbationBounds(force_peak_n=0.0, command_peak_mps=1.0)

    def test_rejects_out_of_range_rate_fraction(self):
        with pytest.raises(ValueError, match="rate_limit_fraction"):
            _mod.PerturbationBounds(force_peak_n=120.0, command_peak_mps=1.0, rate_limit_fraction=1.5)

    def test_default_rate_cap_is_a_tenth_of_the_range(self):
        b = bounds()
        assert b.max_force_delta_per_step == pytest.approx(12.0)
        assert b.max_command_delta_per_step == pytest.approx(b.command_peak_mps * 0.1)


class TestRateLimitedChannel:
    def test_first_step_is_capped_not_jumped(self):
        channel = _mod.RateLimitedChannel(peak=120.0, max_delta_per_step=12.0)
        # Asking for the full 120 N in one step must produce 12 N: that is the 0.1x cap.
        assert channel.step([120.0, -120.0]) == [pytest.approx(12.0), pytest.approx(-12.0)]

    def test_ramps_over_successive_steps(self):
        channel = _mod.RateLimitedChannel(peak=120.0, max_delta_per_step=12.0)
        values = [channel.step([120.0, 0.0])[0] for _ in range(10)]
        assert values == pytest.approx([12.0 * (i + 1) for i in range(10)])
        assert channel.step([120.0, 0.0])[0] == pytest.approx(120.0)  # saturates at the channel peak

    def test_clamps_to_peak(self):
        channel = _mod.RateLimitedChannel(peak=10.0, max_delta_per_step=10.0)
        assert channel.step([1000.0, -1000.0]) == [pytest.approx(10.0), pytest.approx(-10.0)]

    def test_reset_returns_to_zero(self):
        channel = _mod.RateLimitedChannel(peak=10.0, max_delta_per_step=5.0)
        channel.step([5.0, 5.0])
        channel.reset()
        assert channel.state == [0.0, 0.0]

    def test_rejects_3d_action(self):
        channel = _mod.RateLimitedChannel(peak=10.0, max_delta_per_step=1.0)
        with pytest.raises(ValueError, match="2-D"):
            channel.step([1.0, 2.0, 3.0])


class TestMagnitude:
    def test_trained_force_alone_is_one(self):
        out = _mod.normalised_magnitude([120.0, 0.0], [0.0, 0.0], bounds())
        assert out["force_norm"] == pytest.approx(1.0)
        assert out["combined"] == pytest.approx(1.0)

    def test_force_and_command_combine_in_quadrature(self):
        b = bounds()
        out = _mod.normalised_magnitude([120.0, 0.0], [b.command_peak_mps, 0.0], b)
        assert out["command_norm"] == pytest.approx(1.0)
        assert out["combined"] == pytest.approx(math.hypot(1.0, 1.0))

    def test_zero_perturbation_is_zero(self):
        out = _mod.normalised_magnitude([0.0, 0.0], [0.0, 0.0], bounds())
        assert out["combined"] == 0.0

    def test_direction_does_not_change_magnitude(self):
        b = bounds()
        a = _mod.normalised_magnitude([60.0, 60.0], [0.0, 0.0], b)["combined"]
        c = _mod.normalised_magnitude([-60.0, 60.0], [0.0, 0.0], b)["combined"]
        assert a == pytest.approx(c)

    def test_modality_evidence_splits_channels(self):
        b = bounds()
        out = _mod.modality_evidence([120.0, 0.0], [0.0, b.command_peak_mps], b)
        assert out["force_only"] == pytest.approx(1.0)
        assert out["command_only"] == pytest.approx(1.0)
        assert out["combined"] == pytest.approx(math.hypot(1.0, 1.0))


class TestAdversaryReward:
    def test_surviving_robot_is_penalised(self):
        out = _mod.adversary_reward(
            alive=True,
            projected_gravity_z=-1.0,
            base_angular_xy=(0.0, 0.0),
            joint_torques=[0.0] * 12,
            joint_torque_limits=[23.5] * 12,
        )
        assert out["alive_term"] == pytest.approx(-1.0)
        assert out["reward"] < 0  # alive + upright: nothing for the adversary to like

    def test_fallen_robot_earns_no_alive_penalty(self):
        out = _mod.adversary_reward(
            alive=False,
            projected_gravity_z=1.0,
            base_angular_xy=(0.0, 0.0),
            joint_torques=[0.0] * 12,
            joint_torque_limits=[23.5] * 12,
        )
        assert out["alive_term"] == 0.0
        assert out["orient"] == pytest.approx(1.0)
        assert out["reward"] > 0

    def test_orientation_term_rewards_tipping(self):
        upright = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[], joint_torque_limits=[]
        )
        tipping = _mod.adversary_reward(
            alive=True, projected_gravity_z=0.5, base_angular_xy=(0, 0), joint_torques=[], joint_torque_limits=[]
        )
        assert tipping["orient"] > upright["orient"]

    def test_shake_term_uses_roll_pitch_rates(self):
        quiet = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[], joint_torque_limits=[]
        )
        shaking = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(1.0, 2.0), joint_torques=[], joint_torque_limits=[]
        )
        assert shaking["shake"] == pytest.approx(0.05 * (1.0 + 4.0))
        assert shaking["shake"] > quiet["shake"]

    def test_torque_term_only_counts_excess_over_limit(self):
        under = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[10.0], joint_torque_limits=[23.5]
        )
        over = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[47.0], joint_torque_limits=[23.5]
        )
        assert under["torque"] == 0.0
        assert over["torque"] == pytest.approx(0.1 * 1.0)

    def test_lipschitz_penalty_is_subtracted(self):
        without = _mod.adversary_reward(
            alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[], joint_torque_limits=[]
        )
        with_penalty = _mod.adversary_reward(
            alive=True,
            projected_gravity_z=-1.0,
            base_angular_xy=(0, 0),
            joint_torques=[],
            joint_torque_limits=[],
            layer_inf_norms=(1000.0, 2000.0),
        )
        assert with_penalty["lipschitz"] > 0
        assert with_penalty["reward"] < without["reward"]

    def test_torque_list_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="differ"):
            _mod.adversary_reward(
                alive=True, projected_gravity_z=-1.0, base_angular_xy=(0, 0), joint_torques=[1.0], joint_torque_limits=[1.0, 2.0]
            )


def step_evaluator(threshold: float, trials: int = 100):
    """Deterministic 'fall rate' that is 0 below ``threshold`` and 1 at/above it."""

    def evaluate(scale):
        return {"fall_rate": 1.0 if scale >= threshold else 0.0, "trials": trials}

    return evaluate


class TestFindMinimumInducing:
    def test_finds_a_step_boundary(self):
        out = _mod.find_minimum_inducing(step_evaluator(0.6), max_scale=1.0, tolerance=0.01)
        assert out["found"] is True
        assert 0.6 <= out["scale"] < 0.6 + 0.02
        assert out["fall_rate"] == 1.0

    def test_returns_not_found_when_the_maximum_does_not_break_it(self):
        out = _mod.find_minimum_inducing(step_evaluator(1.5), max_scale=1.0)
        assert out["found"] is False
        assert "did not reach the fall threshold" in out["reason"]

    def test_minimum_scale_already_failing_is_reported(self):
        out = _mod.find_minimum_inducing(step_evaluator(0.0), max_scale=1.0)
        assert out["found"] is True
        assert out["scale"] == 0.0
        assert "not the limiting factor" in out["note"]

    def test_records_its_search_history(self):
        out = _mod.find_minimum_inducing(step_evaluator(0.4), max_scale=1.0, tolerance=0.01)
        assert out["iterations"] == len(out["history"]) > 0
        assert all("scale" in h for h in out["history"])

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="max_scale"):
            _mod.find_minimum_inducing(step_evaluator(0.5), max_scale=0)
        with pytest.raises(ValueError, match="fall_threshold"):
            _mod.find_minimum_inducing(step_evaluator(0.5), max_scale=1.0, fall_threshold=2.0)


class TestGateDecision:
    def test_above_trained_range_passes(self):
        out = _mod.gate_decision(1.35)
        assert out["verdict"] == "PASS"
        assert out["margin"] == pytest.approx(0.35)

    def test_exactly_at_trained_range_fails(self):
        # Strictly greater: matching the trained strength is the failure the gate exists to catch.
        assert _mod.gate_decision(1.0)["verdict"] == "FAIL"

    def test_below_trained_range_fails(self):
        assert _mod.gate_decision(0.8)["verdict"] == "FAIL"

    def test_no_boundary_at_all_fails(self):
        out = _mod.gate_decision(None)
        assert out["verdict"] == "FAIL"
        assert "no fall induced" in out["reason"]

    def test_reports_relative_margin(self):
        out = _mod.gate_decision(1.5)
        assert out["relative_margin"] == pytest.approx(0.5)


class TestScaleBounds:
    def test_scales_both_channels(self):
        scaled = _mod.scale_bounds(bounds(), 0.5)
        assert scaled.force_peak_n == pytest.approx(60.0)
        assert scaled.command_peak_mps == pytest.approx(bounds().command_peak_mps * 0.5)

    def test_keeps_the_rate_fraction(self):
        scaled = _mod.scale_bounds(bounds(), 0.25)
        assert scaled.max_force_delta_per_step == pytest.approx(scaled.force_peak_n * 0.1)

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="scale must be a positive"):
            _mod.scale_bounds(bounds(), 0.0)

    def test_scaled_force_at_full_range_reports_the_scale(self):
        base = bounds()
        scaled = _mod.scale_bounds(base, 0.5)
        magnitude = _mod.normalised_magnitude([scaled.force_peak_n, 0.0], [0.0, 0.0], base)
        assert magnitude["combined"] == pytest.approx(0.5)


class TestRateLimitedStepPure:
    def test_reaches_at_most_the_cap_per_step(self):
        assert _mod.rate_limited_step([0.0, 0.0], [100.0, -100.0], peak=50.0, max_delta_per_step=5.0) == [5.0, -5.0]

    def test_never_exceeds_the_peak(self):
        prev = [48.0, 0.0]
        assert _mod.rate_limited_step(prev, [999.0, 0.0], peak=50.0, max_delta_per_step=5.0)[0] == 50.0

    def test_can_return_toward_zero(self):
        assert _mod.rate_limited_step([50.0, 0.0], [0.0, 0.0], peak=50.0, max_delta_per_step=5.0) == [45.0, 0.0]

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            _mod.rate_limited_step([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], peak=1.0, max_delta_per_step=0.1)


class TestModalityChannels:
    def test_force_only_allows_force(self):
        assert _mod.channels_for("force_only") == frozenset({"force"})

    def test_command_only_allows_command(self):
        assert _mod.channels_for("command_only") == frozenset({"command"})

    def test_combined_allows_both(self):
        assert _mod.channels_for("combined") == frozenset({"force", "command"})

    def test_unknown_modality_rejected(self):
        with pytest.raises(ValueError, match="unknown modality"):
            _mod.channels_for("observation")

    def test_every_listed_modality_resolves(self):
        for modality in _mod.MODALITIES:
            assert _mod.channels_for(modality)


class TestModalitySummary:
    def test_combination_succeeds_where_singles_fail(self):
        out = _mod.summarise_modalities(
            {
                "force_only": {"minimum_score": None},
                "command_only": {"minimum_score": 1.8},
                "combined": {"minimum_score": 0.9},
            }
        )
        assert out["force_only"]["breaks_policy"] is False
        assert out["command_only"]["breaks_policy"] is False
        assert out["combined"]["breaks_policy"] is True
        assert out["combination_succeeds_where_singles_fail"] is True

    def test_single_modality_breaking_the_policy_is_not_the_paper_finding(self):
        out = _mod.summarise_modalities(
            {"force_only": {"minimum_score": 0.7}, "command_only": {"minimum_score": None}, "combined": {"minimum_score": 0.5}}
        )
        assert out["force_only"]["breaks_policy"] is True
        assert out["combination_succeeds_where_singles_fail"] is False
