"""Fine-tune a locomotion policy with a trained adversary in the loop.

The Phase 0.5 gate can find that a candidate is broken at a magnitude it was trained to survive. The
paper's remedy (Section IV-C) is not more random pushes but a fraction of rollouts *under the learned
attack*: attack -> fine-tune -> re-attack. This module supplies the missing piece -- an env wrapper that
lets a trained, frozen adversary act on a small share of envs while the locomotion policy trains.

It sits between the vectorised env and the rsl-rl runner, so the training loop, rewards, PPO settings
and network are untouched: only a fraction of the observed trajectories are attacked, exactly as the
paper mixes roughly 5% adversarial rollouts into the batch.

Design notes:
  * the attack mask is drawn per env at episode start (not per step), so an attacked rollout is
    attacked for its whole episode rather than flickering on and off;
  * non-attacked envs keep their own command -- the wrapper rewrites only the attacked rows;
  * the adversary never trains here: its parameters are frozen and its output is clamped to the same
    bounds the gate used, so fine-tuning targets the attack that actually broke the candidate.
"""

from __future__ import annotations

import torch

from policy.locomotion.adversarial_robustness import attack_math


class AdversarialRolloutWrapper:
    """Wrap an rsl-rl vec env and apply a frozen adversary to ``mix_fraction`` of the envs."""

    def __init__(
        self,
        env,
        adversary,
        hooks,
        bounds: attack_math.PerturbationBounds,
        *,
        mix_fraction: float = 0.05,
        attack_scale: float = 1.0,
        seed: int | None = None,
    ):
        if not 0.0 <= mix_fraction <= 1.0:
            raise ValueError(f"mix_fraction must be in [0, 1], got {mix_fraction}")
        self.env = env
        self.adversary = adversary
        self.hooks = hooks
        self.bounds = bounds
        self.mix_fraction = mix_fraction
        self.attack_scale = attack_scale
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)
        self._obs = None
        self._mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # -- passthrough surface the runner reads ---------------------------------------------------
    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def num_obs(self) -> int:
        return self.env.num_obs

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    @property
    def max_episode_length(self) -> int:
        return self.env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @property
    def device(self):
        return self.env.device

    @property
    def cfg(self):
        return getattr(self.env, "cfg", {})

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def get_observations(self):
        self._obs = self.env.get_observations()
        return self._obs

    def reset(self):
        self.get_observations()
        self._mask = self._draw_mask(reset_all=True)
        return self._obs

    # -- the wrapper's actual job ---------------------------------------------------------------
    def _draw_mask(self, reset_all: bool = False):
        """Attack mask for envs whose episode just started (all of them on a full reset)."""
        draws = torch.rand(self.num_envs, generator=self._generator)
        fresh = torch.as_tensor(draws < self.mix_fraction, device=self.device)
        if reset_all:
            return fresh
        episodes_started = torch.as_tensor(self.episode_length_buf == 0, device=self.device)
        return torch.where(episodes_started, fresh, self._mask)

    def _split_action(self, attack_action):
        """Clamp the adversary's raw 4-D output to the gate's bounds (no rate limit here: the gate
        already found the achievable magnitude, and fine-tuning should face exactly that)."""
        scaled = attack_action * self.attack_scale
        clamp = torch.tensor(
            [self.bounds.force_peak_n, self.bounds.force_peak_n,
             self.bounds.command_peak_mps, self.bounds.command_peak_mps],
            device=scaled.device,
        )
        scaled = torch.clamp(scaled, -clamp, clamp)
        return scaled[:, :2], scaled[:, 2:]

    def step(self, actions):
        self._mask = self._draw_mask()
        if self._mask.any() and self._obs is not None:
            with torch.no_grad():
                force, command = self._split_action(self.adversary(self._obs))
            force = force * self._mask.unsqueeze(1)
            self.hooks.apply_force(force)
            current = self.hooks.command_term.vel_command_b[:, :2].clone()
            current[self._mask] = command[self._mask]
            self.hooks.apply_command(current)
        outcome = self.env.step(actions)
        self._obs = outcome[0]
        return outcome


def make_adversarial_wrapper(env, *, adversary_dir: str, mix_fraction: float = 0.05,
                             attack_scale: float = 1.0, seed: int | None = None):
    """Factory used by ``train.py --adversarial_env``: load a frozen adversary and wrap the env.

    ``adversary_dir`` is the adversary's run directory (its manifest records the runner config), and
    the bounds come from the platform's push curriculum plus the task's own command ranges -- the same
    derivation the gate used, so the fine-tune faces exactly the attack that broke the candidate.
    """
    from policy.locomotion.adversarial_robustness import (
        attack_env,
        train_adversary,
    )
    from policy.locomotion.core.platforms.go2.platform import GO2

    adversary = train_adversary.load_adversary_policy(env, adversary_dir)
    hooks = attack_env.ManagerBasedHooks(env.unwrapped)
    command_ranges = env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges
    bounds = attack_math.PerturbationBounds.from_config(GO2, command_ranges)
    return AdversarialRolloutWrapper(
        env, adversary, hooks, bounds, mix_fraction=mix_fraction, attack_scale=attack_scale, seed=seed
    )
