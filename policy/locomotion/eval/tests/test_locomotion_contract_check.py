"""Tests for the P1 contract check (policy/locomotion/eval/contract_check.py).

The module is loaded by file path: importing `policy.locomotion` runs the platform auto-discovery,
which needs isaaclab_tasks (pod-only). The checker itself is pure python, so its behaviour -- and
specifically its ability to catch an observation reorder, an action-scale change and a control-rate
change -- is verifiable here without Isaac.

Reference-run tests skip when `policy/locomotion/runs/<run>/params/` is absent (it is gitignored).
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "contract_check.py"
_spec = importlib.util.spec_from_file_location("_contract_check_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REPO_ROOT = Path(__file__).resolve().parents[4]
SEPTEMBER_RUN = REPO_ROOT / "policy" / "locomotion" / "runs" / "2026-09-08_19-28-32_analysis"

needs_september_run = pytest.mark.skipif(
    not (SEPTEMBER_RUN / "params" / "env.yaml").is_file(),
    reason="September run's params/ tree is gitignored; present only on a workstation that had the run",
)


def make_reference(**overrides):
    """A complete, self-consistent contract dict (the shape load_reference returns)."""
    base = {
        "run_dir": "synthetic",
        "source": "test",
        "obs_group": "policy",
        "obs_term_names": ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands"],
        "obs_term_funcs": ["base_lin_vel", "base_ang_vel", "projected_gravity", "generated_commands"],
        "obs_term_widths": {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3},
        "obs_dim": 12,
        "enable_corruption": True,
        "action_dim": 12,
        "action_scale": 0.25,
        "action_offset": 0.0,
        "action_use_default_offset": True,
        "decimation": 4,
        "sim_dt": 0.005,
        "step_dt": 0.02,
        "command_ranges": {"lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
        "standing_env_fraction": 0.02,
        "command_resampling_s": (10.0, 10.0),
        "seed": 42,
        "num_envs": 4096,
        "max_iterations": 1500,
        "num_steps_per_env": 24,
        "empirical_normalization": None,
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [512, 256, 128],
        "actor_class": "ActorCritic",
        "algorithm_class": "PPO",
        "learning_rate": 0.001,
        "gamma": 0.99,
        "lam": 0.95,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "entropy_coef": 0.01,
        "desired_kl": 0.01,
    }
    base.update(overrides)
    return base


@needs_september_run
class TestSeptemberReference:
    def test_obs_contract(self):
        ref = _mod.load_reference(SEPTEMBER_RUN)
        # 3 (lin vel) + 3 (ang vel) + 3 (gravity) + 3 (command) + 12*3 (joint pos/vel/actions) + 187
        # (1.6 m x 1.0 m height grid at 0.1 m) = 235
        assert ref["obs_dim"] == 235
        assert ref["obs_term_names"][2] == "projected_gravity"
        assert ref["obs_term_widths"]["height_scan"] == 187
        assert ref["obs_term_widths"]["velocity_commands"] == 3  # heading range is not a command dim

    def test_action_and_control_contract(self):
        ref = _mod.load_reference(SEPTEMBER_RUN)
        assert ref["action_dim"] == 12
        assert ref["action_scale"] == pytest.approx(0.25)
        assert ref["action_use_default_offset"] is True
        assert ref["decimation"] == 4
        assert ref["sim_dt"] == pytest.approx(0.005)
        assert ref["step_dt"] == pytest.approx(0.02)

    def test_command_contract(self):
        ref = _mod.load_reference(SEPTEMBER_RUN)
        assert ref["command_ranges"]["lin_vel_x"] == (-1.0, 1.0)
        assert ref["command_ranges"]["lin_vel_y"] == (-1.0, 1.0)
        assert ref["command_ranges"]["ang_vel_z"] == (-1.0, 1.0)
        assert ref["command_resampling_s"] == (10.0, 10.0)
        assert ref["standing_env_fraction"] == pytest.approx(0.02)

    def test_normalizer_contract(self):
        # No observation normalisation in the September recipe: an ONNX fallback policy takes raw obs.
        ref = _mod.load_reference(SEPTEMBER_RUN)
        assert ref["empirical_normalization"] is None
        assert ref["actor_obs_normalization"] is False

    def test_identical_live_contract_passes(self):
        ref = _mod.load_reference(SEPTEMBER_RUN)
        assert _mod.verdict(_mod.compare(ref, dict(ref))) == "CONTRACT OK"


class TestCompareCatches:
    def test_observation_reorder_is_a_mismatch(self):
        ref = make_reference()
        live = dict(ref, obs_term_names=list(reversed(ref["obs_term_names"])))
        rows = _mod.compare(ref, live)
        assert _mod.verdict(rows) == "CONTRACT MISMATCH"
        assert any("obs_term_names" in line for line in _mod.mismatch_summary(rows))

    def test_observation_dim_change_is_a_mismatch(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, obs_dim=236))
        assert _mod.verdict(rows) == "CONTRACT MISMATCH"

    def test_action_scale_change_is_a_mismatch(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, action_scale=0.5))
        assert _mod.verdict(rows) == "CONTRACT MISMATCH"
        assert "action_scale" in _mod.mismatch_summary(rows)[0]

    def test_control_rate_change_is_a_mismatch(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, decimation=2, step_dt=0.01))
        assert _mod.verdict(rows) == "CONTRACT MISMATCH"

    def test_command_range_change_is_caught_elementwise(self):
        ref = make_reference()
        live = dict(ref, command_ranges={"lin_vel_x": (-1.0, 1.5), "ang_vel_z": (-1.0, 1.0)})
        assert _mod.verdict(_mod.compare(ref, live)) == "CONTRACT MISMATCH"

    def test_float_noise_within_tolerance_is_not_a_mismatch(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, sim_dt=0.005 + 1e-12))
        assert _mod.verdict(rows) == "CONTRACT OK"

    def test_missing_live_value_is_incomplete_not_ok(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, action_scale=None))
        assert _mod.verdict(rows) == "INCOMPLETE"

    def test_run_metadata_never_gates(self):
        # num_envs / seed / iterations describe the run, not what the policy sees.
        ref = make_reference()
        live = dict(ref, num_envs=1024, seed=7, max_iterations=999)
        assert _mod.verdict(_mod.compare(ref, live)) == "CONTRACT OK"

    def test_unobservable_term_widths_do_not_crash_compare(self):
        ref = make_reference()
        live = dict(ref, obs_term_widths=None, obs_term_funcs=None)
        rows = _mod.compare(ref, live)
        assert _mod.verdict(rows) == "CONTRACT OK"  # informational rows only
        assert all(r["ok"] is None for r in rows if r["field"] in ("obs_term_widths", "obs_term_funcs"))


class TestFormatting:
    def test_table_marks_mismatches(self):
        ref = make_reference()
        rows = _mod.compare(ref, dict(ref, action_scale=0.5))
        table = _mod.format_table(rows)
        assert "MISMATCH" in table
        assert table.splitlines()[0].startswith("field")

    def test_mismatch_summary_is_empty_when_ok(self):
        ref = make_reference()
        assert _mod.mismatch_summary(_mod.compare(ref, dict(ref))) == []
