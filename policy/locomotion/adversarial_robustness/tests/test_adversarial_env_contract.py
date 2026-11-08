"""Local exercise of the attack/robustify wrappers against a fake vectorised env.

Why this exists
---------------
Every Isaac-side bug this phase cost 10-30 minutes of paid GPU time to find, and all of them were
*contract* bugs, not physics: a policy handed a Python module instead of a callable, observations
arriving as a mapping, `gymnasium` returning five values from `step`, a policy input that was not
grouped, a training loop that never saved. None of those needs a GPU to catch -- they need a stand-in
env that behaves like the real one at the interface.

So this file stubs the Isaac-only surface (`isaaclab`, `isaaclab_rl`) and drives the real wrapper code
with a fake env that returns a grouped observation, a five-value step and per-env state. It runs under
CPU torch, no Isaac, no GPU:

    python3 -m venv /tmp/cerberus-preflight
    /tmp/cerberus-preflight/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/cerberus-preflight/bin/pip install pytest
    /tmp/cerberus-preflight/bin/python -m pytest \
        policy/locomotion/adversarial_robustness/tests/test_adversarial_env_contract.py -q

Skipped automatically where torch is absent (the repo's default pytest environment).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _install_stub_isaac_modules():
    """Stand-ins so the wrapper modules import without Isaac Lab; returns the names added.

    Importing ``policy.locomotion`` runs its platform auto-discovery, which calls
    ``isaaclab_tasks.utils.import_packages``; a no-op is enough because the wrappers only touch Isaac
    inside their build functions. ``isaaclab_rl.rsl_rl`` is stubbed because ``attack_env`` imports the
    vec-env adapter inside ``build_attack_env`` and the tests only need the type to exist.
    """
    added = []
    for name in ("isaaclab", "isaaclab.app", "isaaclab_rl", "isaaclab_tasks", "isaaclab_tasks.utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            added.append(name)
    utils = sys.modules["isaaclab_tasks.utils"]
    utils.import_packages = lambda *args, **kwargs: None
    sys.modules["isaaclab_tasks"].utils = utils

    rsl = sys.modules.get("isaaclab_rl.rsl_rl")
    if rsl is None:
        rsl = types.ModuleType("isaaclab_rl.rsl_rl")
        sys.modules["isaaclab_rl.rsl_rl"] = rsl
        added.append("isaaclab_rl.rsl_rl")

        class RslRlVecEnvWrapper:
            """Placeholder."""

        rsl.RslRlVecEnvWrapper = RslRlVecEnvWrapper
    return added


# The stubs are removed again after the import: a fake ``isaaclab`` left in sys.modules for the rest of
# the session makes every test that asserts "this module does not import Isaac" fail.
_ADDED_STUBS = _install_stub_isaac_modules()

from policy.locomotion.adversarial_robustness import (
    attack_env,
    attack_math,
    robustify_env,
)

for _name in _ADDED_STUBS:
    sys.modules.pop(_name, None)

NUM_ENVS = 4
OBS_DIM = 12
ACT_DIM = 3


class FakeCommand:
    def __init__(self, num_envs):
        self.vel_command_b = torch.zeros((num_envs, 3))
        self.vel_command_b[:, 0] = 0.5  # a real command, so "unchanged" is observable


class FakeHooks:
    """Records what the wrappers ask for, and returns a grouped observation."""

    def __init__(self, num_envs=NUM_ENVS, obs_dim=OBS_DIM):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.command_term = FakeCommand(num_envs)
        self.forces = []
        self.commands = []

    def apply_force(self, force_xy):
        self.forces.append(force_xy.detach().clone())

    def apply_command(self, command_xy):
        self.commands.append(command_xy.detach().clone())

    def observations(self):
        return torch.zeros((self.num_envs, self.obs_dim))

    def read_state(self, lambdas=None):
        n = self.num_envs
        return {
            "projected_gravity_z": torch.zeros(n),
            "base_angular_xy": torch.zeros((n, 2)),
            "joint_torques": torch.zeros((n, ACT_DIM)),
            "joint_torque_limits": torch.ones((n, ACT_DIM)),
            "layer_inf_norms": lambdas,
        }


class FakeVecEnv:
    """Mimics RslRlVecEnvWrapper on the GA stack: grouped obs, five-value step, TensorDict-ish."""

    def __init__(self, num_envs=NUM_ENVS, obs_dim=OBS_DIM, act_dim=ACT_DIM, step_shape=5):
        self.num_envs = num_envs
        self.num_obs = obs_dim
        self.num_actions = act_dim
        self.max_episode_length = 10
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.step_shape = step_shape
        self.steps = 0

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return {"policy": torch.zeros((self.num_envs, self.num_obs))}

    def reset(self):
        return {"policy": torch.zeros((self.num_envs, self.num_obs))}, {}

    def step(self, actions):
        self.steps += 1
        self.episode_length_buf += 1
        obs = {"policy": torch.zeros((self.num_envs, self.num_obs))}
        reward = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_shape == 5:
            return obs, reward, terminated, truncated, {}
        dones = terminated | truncated
        return obs, reward, dones, {"time_outs": truncated}


class GroupedPolicy(torch.nn.Module):
    """Stands in for rsl-rl's MLPModel: it reads `obs[group]` and raises on a bare tensor."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.obs_groups = ["policy"]
        self.linear = torch.nn.Linear(obs_dim, act_dim)

    def forward(self, obs):
        if not isinstance(obs, dict):
            raise TypeError(f"policy expected a grouped observation, got {type(obs).__name__}")
        return self.linear(obs["policy"])


def make_bounds():
    return attack_math.PerturbationBounds(force_peak_n=100.0, command_peak_mps=1.0)


class TestAttackWrapperContract:
    def _wrapper(self, modality="combined", num_envs=NUM_ENVS, step_shape=5):
        hooks = FakeHooks(num_envs)
        env = FakeVecEnv(num_envs, step_shape=step_shape)
        policy = GroupedPolicy()
        return attack_env.AttackEnvWrapper(
            env=env, policy=policy, bounds=make_bounds(), hooks=hooks, modality=modality
        ), hooks, env

    def test_policy_receives_a_grouped_observation(self):
        # The bug this guards: the wrapper passed the raw tensor, rsl-rl's model reads obs[group].
        wrapper, _, _ = self._wrapper()
        _obs, reward, dones, _extras = wrapper.step(torch.zeros((NUM_ENVS, 4)))
        assert reward.shape == (NUM_ENVS,)
        assert dones.shape == (NUM_ENVS,)

    def test_five_value_step_is_accepted(self):
        wrapper, _, env = self._wrapper(step_shape=5)
        _obs, _reward, dones, extras = wrapper.step(torch.zeros((NUM_ENVS, 4)))
        assert env.steps == 1
        assert "time_outs" in extras
        assert dones.shape == (NUM_ENVS,)

    def test_four_value_step_is_accepted(self):
        wrapper, _, _ = self._wrapper(step_shape=4)
        _obs, _reward, dones, extras = wrapper.step(torch.zeros((NUM_ENVS, 4)))
        assert dones.shape == (NUM_ENVS,)
        assert "time_outs" in extras

    def test_force_and_command_are_applied_once_per_step(self):
        wrapper, hooks, _ = self._wrapper()
        wrapper.step(torch.zeros((NUM_ENVS, 4)))
        assert len(hooks.forces) == 1 and len(hooks.commands) == 1

    def test_force_is_rate_limited_to_a_tenth_of_range(self):
        wrapper, hooks, _ = self._wrapper()
        wrapper.step(torch.full((NUM_ENVS, 4), 10.0))  # asks far past the bounds
        force = hooks.forces[0]
        assert torch.all(force.abs() <= 0.1 * 100.0 + 1e-6), force

    def test_force_only_modality_leaves_the_command_alone(self):
        wrapper, hooks, _ = self._wrapper(modality="force_only")
        wrapper.step(torch.full((NUM_ENVS, 4), 10.0))
        assert torch.all(hooks.commands[0][:, 0] == 0.0)  # zeroed command channel, not 0.5

    def test_command_only_modality_applies_no_force(self):
        wrapper, hooks, _ = self._wrapper(modality="command_only")
        wrapper.step(torch.full((NUM_ENVS, 4), 10.0))
        assert torch.all(hooks.forces[0] == 0.0)

    def test_get_observations_delegates_to_the_env(self):
        wrapper, _, _ = self._wrapper()
        obs = wrapper.get_observations()
        assert isinstance(obs, dict) and "policy" in obs

    def test_policy_input_wrapper_raises_for_multi_group_models(self):
        wrapper, _, _ = self._wrapper()
        wrapper.policy.obs_groups = ["policy", "critic"]
        with pytest.raises(RuntimeError, match="several observation groups"):
            wrapper.policy_input(torch.zeros((NUM_ENVS, OBS_DIM)))


class TestRateLimitPerEnv:
    def test_channels_advance_independently(self):
        wrapper, hooks, _ = TestAttackWrapperContract()._wrapper()
        wrapper.step(torch.tensor([[100.0, 0.0, 0.0, 0.0]] * NUM_ENVS))
        first = hooks.forces[-1][:, 0].clone()
        # Ask a second time with the same raw value: every env should advance by the same capped step.
        wrapper.step(torch.tensor([[100.0, 0.0, 0.0, 0.0]] * NUM_ENVS))
        second = hooks.forces[-1][:, 0].clone()
        assert torch.all(second > first)
        assert torch.allclose(second - first, torch.full_like(first, 0.1 * 100.0))


class TestScaleReachesBeyondTheTrainedRange:
    def test_scale_above_one_actually_allows_more_force(self):
        # The regression this guards: clamping at the base peak made every scale > 1 identical to 1, so a
        # "no boundary up to 2x" verdict would have been reported without ever applying 2x.
        hooks = FakeHooks(NUM_ENVS)
        wrapper = attack_env.AttackEnvWrapper(
            env=FakeVecEnv(NUM_ENVS), policy=GroupedPolicy(), bounds=make_bounds(), hooks=hooks, modality="force_only"
        )
        wrapper.scale = 2.0
        for _ in range(30):  # ramp past the base peak (10 N per step at rate 0.1 of 100 N)
            wrapper.step(torch.full((NUM_ENVS, 4), 1000.0))
        assert torch.all(hooks.forces[-1].abs() > 100.0), hooks.forces[-1]
        assert torch.all(hooks.forces[-1].abs() <= 200.0 + 1e-6)


class TestRobustifyWrapper:
    def _wrapper(self, mix, num_envs=NUM_ENVS):
        hooks = FakeHooks(num_envs)
        env = FakeVecEnv(num_envs)
        adversary = GroupedPolicy(obs_dim=OBS_DIM, act_dim=4)
        # The adversary is called with the env's grouped observation too.
        adversary.obs_groups = ["policy"]
        wrapper = robustify_env.AdversarialRolloutWrapper(
            env, adversary, hooks, make_bounds(), mix_fraction=mix, seed=0
        )
        wrapper.reset()
        return wrapper, hooks, env

    def test_zero_mix_never_perturbs(self):
        wrapper, hooks, _ = self._wrapper(mix=0.0)
        wrapper.step(torch.zeros((NUM_ENVS, ACT_DIM)))
        assert hooks.forces == [] and hooks.commands == []

    def test_full_mix_perturbs_every_env(self):
        wrapper, hooks, _ = self._wrapper(mix=1.0)
        wrapper.step(torch.zeros((NUM_ENVS, ACT_DIM)))
        assert len(hooks.forces) == 1
        assert hooks.forces[0].shape == (NUM_ENVS, 2)

    def test_unattacked_envs_keep_their_command(self):
        wrapper, hooks, _ = self._wrapper(mix=1.0)
        wrapper._mask = torch.tensor([True, False, True, False])
        wrapper.step(torch.zeros((NUM_ENVS, ACT_DIM)))
        command = hooks.commands[0][:, 0]
        # Rows that were not attacked keep the env's own 0.5 m/s command.
        assert torch.all(command[~wrapper._mask] == 0.5)

    def test_mask_is_redrawn_only_at_episode_start(self):
        wrapper, _, env = self._wrapper(mix=0.5)
        wrapper._mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
        env.episode_length_buf += 1  # mid-episode: nothing starts
        mask = wrapper._draw_mask()
        assert not mask.any()

    def test_wrapper_exposes_the_runner_surface(self):
        wrapper, _, env = self._wrapper(mix=0.5)
        assert wrapper.num_envs == env.num_envs
        assert wrapper.num_obs == env.num_obs
        assert wrapper.num_actions == env.num_actions
        assert wrapper.max_episode_length == env.max_episode_length
        assert wrapper.episode_length_buf is env.episode_length_buf


class TestRunnerCfgFromManifest:
    """The gate and the fine-tuner rebuild the adversary from its manifest, never from a default.

    Lives here rather than with the pure gate-math tests because importing train_adversary pulls torch,
    which the repo's default pytest environment does not have.
    """

    def test_reads_runner_cfg(self, tmp_path):
        import json

        from policy.locomotion.adversarial_robustness.train_adversary import (
            runner_cfg_from_manifest,
        )

        run = tmp_path / "adversary"
        run.mkdir()
        (run / "train_manifest.json").write_text(json.dumps({"runner_cfg": {"class_name": "OnPolicyRunner"}}))
        assert runner_cfg_from_manifest(run)["class_name"] == "OnPolicyRunner"

    def test_missing_manifest_is_an_error(self, tmp_path):
        from policy.locomotion.adversarial_robustness.train_adversary import (
            runner_cfg_from_manifest,
        )

        with pytest.raises(FileNotFoundError, match="train_manifest.json"):
            runner_cfg_from_manifest(tmp_path / "nope")

    def test_manifest_without_runner_cfg_is_an_error(self, tmp_path):
        import json

        from policy.locomotion.adversarial_robustness.train_adversary import (
            runner_cfg_from_manifest,
        )

        run = tmp_path / "adversary"
        run.mkdir()
        (run / "train_manifest.json").write_text(json.dumps({"iterations": 1000}))
        with pytest.raises(KeyError, match="runner_cfg"):
            runner_cfg_from_manifest(run)
