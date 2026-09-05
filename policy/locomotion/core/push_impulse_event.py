"""Shared true-impulse push mechanism: apply a force for a short fixed duration via
Isaac Lab's `permanent_wrench_composer`, then explicitly clear it back to zero.

Isaac Lab's `permanent_wrench_composer` (unlike `instantaneous_wrench_composer`, which
resets every physics step -- verified directly from Isaac Lab source,
`Articulation.write_data_to_sim`/`reset`) only resets at episode reset. Isaac Lab's own
built-in `mdp.apply_external_force_torque` (used by both curriculum.py's training push
and, until this fix, push_recovery_event.py's eval push) sets a force via this composer
and never clears it -- so what both those callers called a "push" was actually a
continuous, unending force lasting from one interval trigger to the next (5-12s in
training; from push_time_s to episode end in eval), not the short impulse (0.15-0.25s)
both REFERENCES.md and isaaclab-go2-locomotion's own numbers assume. See
REFERENCES.md's "True impulse, not a sustained force" section for the full
investigation (a real 1500-iteration training run's near-100% base_contact failure
rate, and a 0% push-recovery-eval result at the exact peak trained magnitude, both
traced to this).

Both `push_disturbance_curriculum` (training, via `apply_push_impulse` wired in
push_disturbance_cfg.py) and `push_recovery_event.py`'s `apply_single_push` (eval) use
`schedule_push_clear` here, paired with a second EventTerm running
`clear_expired_push_impulses` at a degenerate (step_dt, step_dt) interval so it fires
every env.step() and zeroes any impulse whose duration has elapsed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Larger than any realistic common_step_counter value within a training run -- marks an
# env as having no push scheduled, so clear_expired_push_impulses leaves it alone until
# its first real push sets a real clear step.
_NO_PUSH_SCHEDULED = 2**62


def _clear_step_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "push_clear_step"):
        env.push_clear_step = torch.full(
            (env.num_envs,), _NO_PUSH_SCHEDULED, dtype=torch.int64, device=env.device
        )
    return env.push_clear_step


def schedule_push_clear(env: ManagerBasedEnv, env_ids: torch.Tensor, impulse_duration_s: float) -> None:
    """Records the step at which `env_ids`' just-applied push should clear to zero."""
    duration_steps = max(1, round(impulse_duration_s / env.step_dt))
    _clear_step_buffer(env)[env_ids] = env.common_step_counter + duration_steps


def apply_push_impulse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    impulse_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Same random sampling as `isaaclab.envs.mdp.apply_external_force_torque`, plus
    scheduling this impulse to self-clear after `impulse_duration_s` (paired with
    `clear_expired_push_impulses`) instead of persisting until the next trigger.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=asset_cfg.body_ids, env_ids=env_ids
    )
    schedule_push_clear(env, env_ids, impulse_duration_s)


def clear_expired_push_impulses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Zeroes any env's push force once its scheduled clear step has passed.

    Registered as its own EventTerm at a degenerate (step_dt, step_dt) interval so it
    fires every env.step() for (statistically, with thousands of envs on the same
    degenerate timer) every env, not just the sparse env_ids a push term's own trigger
    happens to include this call -- it sweeps env.common_step_counter (a single global
    step index, not per-env) against every env's own recorded clear step instead of
    relying on the env_ids Isaac Lab's interval bookkeeping passes in.
    """
    clear_step = _clear_step_buffer(env)
    expired = (clear_step <= env.common_step_counter) & (clear_step != _NO_PUSH_SCHEDULED)
    if not expired.any():
        return
    expired_ids = expired.nonzero(as_tuple=True)[0]
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    zeros = torch.zeros((len(expired_ids), num_bodies, 3), device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=zeros, torques=zeros, body_ids=asset_cfg.body_ids, env_ids=expired_ids
    )
    clear_step[expired_ids] = _NO_PUSH_SCHEDULED
