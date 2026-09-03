"""Deterministic, single-shot push event for the push-recovery eval sweep.

Unlike curriculum.py's training-time push_disturbance (random magnitude,
random interval, x/y/z symmetric -- broad robustness training), this applies
exactly ONE horizontal push per episode, at a precise simulation time, with a
per-env magnitude fixed by the eval harness (see push_recovery_eval.py) --
because the eval's job (plans/handoff_phase0.md step 4) is the classic
"push recovery" curve: a controlled shove at a known magnitude, not the
broader disturbance distribution training uses.

Direction is randomized per trial (uniform in the horizontal plane) so many
trials at the same magnitude sample different push angles, matching
Rudin/Hwangbo's own push-test methodology. z is left at 0 -- vertical
"payload" disturbance is a training-time robustness concern (see
REFERENCES.md's "push direction" section), not what a push-*recovery* test
means; the handoff itself frames this as "the base up to +-1 m/s in x/y."

Registered as an EventTerm with mode="interval" and a degenerate
interval_range_s=(T, T) (min==max), so it fires exactly once per episode at
simulation time T -- reusing Isaac Lab's own interval bookkeeping rather than
reimplementing step-counting. Relies on all envs resetting in lockstep at
eval start (true for push_recovery_eval.py's one-env-per-trial design), so
every env's interval elapses at the same step and every trial gets pushed at
the same point in its own episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_single_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    magnitudes: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    log_angles: bool = False,
) -> None:
    """One-shot horizontal push. `magnitudes` is a [num_envs] tensor of force
    magnitude in N, one fixed value per env (set once by push_recovery_eval.py
    before the run -- not sampled here). Direction is randomized per call.

    `log_angles`, if True, records the actual sampled angle (radians) into
    `env.push_angle_log` (a [num_envs] tensor, created here on first call) --
    NOT via a caller-supplied tensor passed through EventTermCfg params:
    Isaac Lab's manager deep-copies term params when the EventTerm is
    registered, so writes to a tensor passed that way never reflect back to
    the caller's own reference (silently -- no error, just always-zero
    reads). Writing directly onto the live `env` object instead is the
    pattern this codebase already uses for the same reason (see
    core/curriculum.py's `env.push_curriculum_k_c`).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    mags = magnitudes[env_ids]
    theta = torch.rand(len(env_ids), device=asset.device) * 2 * torch.pi
    if log_angles:
        if not hasattr(env, "push_angle_log"):
            env.push_angle_log = torch.zeros(env.num_envs, device=asset.device)
        env.push_angle_log[env_ids] = theta
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    forces[:, 0, 0] = mags * torch.cos(theta)
    forces[:, 0, 1] = mags * torch.sin(theta)
    torques = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)

    # same underlying call apply_external_force_torque uses -- forces are
    # only actually applied on the next asset.write_data_to_sim(), which the
    # environment's own step loop already calls.
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )
