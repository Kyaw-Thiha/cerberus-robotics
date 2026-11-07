"""Phase 0.5 attack environment: a frozen locomotion policy inside the loop, perturbed by an adversary.

Why a wrapper rather than a new Isaac Lab task
----------------------------------------------
The thing being *trained* here is not the locomotion policy -- that is frozen -- so this cannot be an
ordinary ``ManagerBasedRLEnvCfg``. Each step:

  1. the adversary emits a 4-D action: base push force (x, y) and a velocity-command override (x, y);
  2. the wrapper clamps it to the trained bounds, rate-limits it to 0.1x range per step, and applies it
     through two hooks;
  3. the **frozen** policy turns the (attacked) observation into joint targets;
  4. the environment steps, and the adversary's reward is computed from the resulting robot state.

Backend-agnostic by construction (an H2 requirement): the policy is *any* ``obs -> actions`` callable
and the perturbations go through ``ForceHook`` / ``CommandHook``. Nothing here imports the locomotion
training task's runner, so a future blind / map-based / world-model backend can be attacked through
this same wrapper unchanged. Only ``ManagerBasedHooks`` (the default, used for the Go2 ManagerBased
task) knows about Isaac Lab managers.

Pure logic (clamping, rate limiting, magnitudes, reward, gate) lives in ``attack_math`` and is tested
there without Isaac. This file is the thin, Isaac-touching layer.

POD-ONLY. Every Isaac API used here is confirmed against the repo's own push code where it exists
(``push_impulse_event.set_forces_and_torques``, ``command_manager.get_term("base_velocity").vel_command_b``);
the three marked ``POD-CHECK`` in :meth:`ManagerBasedHooks.read_state` are the ones the P3a smoke must
confirm before any adversary training time is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from policy.locomotion.adversarial_robustness import attack_math

# ------------------------------------------------------------------------------------------------
# Hooks: the only two places a perturbation enters the simulation
# ------------------------------------------------------------------------------------------------


class ManagerBasedHooks:
    """Default hooks for a ManagerBasedRLEnv Go2 task.

    ``apply_force`` writes the horizontal force on the root body through the same
    ``permanent_wrench_composer`` path the repo's own push events use (see
    ``policy/locomotion/core/push_impulse_event.py``); ``apply_command`` writes the command term's
    live buffer, the same object the eval script reads. Both are re-applied every step: the composer
    persists until reset and the command term resamples on its own timer, so a per-step write is what
    makes the adversary's position hold for exactly as long as the adversary holds it.
    """

    def __init__(self, env, root_body_name: str = "base", command_name: str = "base_velocity"):
        self.env = env.unwrapped
        self.command_name = command_name
        self.asset = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(root_body_name)[0] if root_body_name else slice(None)
        self.command_term = self.env.command_manager.get_term(command_name)
        self._num_bodies = self.asset.num_bodies

    def apply_force(self, force_xy) -> None:
        n = self.env.num_envs
        forces = torch.zeros((n, self._num_bodies, 3), device=self.env.device)
        forces[:, 0, 0] = force_xy[:, 0]
        forces[:, 0, 1] = force_xy[:, 1]
        torques = torch.zeros_like(forces)
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=self.body_ids, env_ids=torch.arange(n, device=self.env.device)
        )

    def apply_command(self, command_xy) -> None:
        self.command_term.vel_command_b[:, :2] = command_xy

    def observations(self):
        """Recompute observations after the command override (the command is an observation term)."""
        return self.env.observation_manager.compute()["policy"]

    def read_state(self, lambdas=None) -> dict:
        """Everything the adversary reward needs, read once per step.

        POD-CHECK 1: ``asset.data.projected_gravity_b`` -- the body-frame gravity vector the policy
        itself observes; z is -1 upright, +1 upside down.
        POD-CHECK 2: ``asset.data.applied_torque`` and ``asset.data.joint_effort_limits`` -- used for
        the paper's torque-limit term. If either is absent on this Lab line, fall back to zeros (the
        term then contributes nothing) rather than guessing limits.
        POD-CHECK 3: joint ordering -- both are indexed by joint, and the paper's term is a sum, so a
        permutation is harmless; this is noted only because other code in the stack does care.
        """
        data = self.asset.data
        projected_gravity = getattr(data, "projected_gravity_b", None)
        g_z = projected_gravity[:, 2] if projected_gravity is not None else torch.zeros(self.env.num_envs, device=self.env.device)
        base_ang_vel = getattr(data, "root_ang_vel_b", None)
        if base_ang_vel is None:
            base_ang_vel = torch.zeros((self.env.num_envs, 3), device=self.env.device)
        torques = getattr(data, "applied_torque", None)
        limits = getattr(data, "joint_effort_limits", None)
        return {
            "projected_gravity_z": g_z,
            "base_angular_xy": base_ang_vel[:, :2],
            "joint_torques": torques,
            "joint_torque_limits": limits,
            "layer_inf_norms": lambdas,
        }


# ------------------------------------------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------------------------------------------


@dataclass
class AttackEnvWrapper:
    """A frozen policy + an adversary channel over an existing env. ``step`` takes the attack action.

    ``policy`` MUST be frozen by the caller (``torch.no_grad`` is used here too, and
    :func:`assert_policy_frozen` verifies no weights changed -- the spec's "Done with" item).
    """

    env: object
    policy: object
    bounds: attack_math.PerturbationBounds
    hooks: object
    modality: str = "combined"
    reward_config: attack_math.AdversaryRewardConfig = field(default_factory=attack_math.AdversaryRewardConfig)
    layer_inf_norms_fn: object | None = None
    scale: float = 1.0
    """Multiplier on the adversary's raw output. The reduce-until-survives search sets this below 1
    instead of rebuilding the env per probe: the channel peaks stay at the trained bounds, so the
    magnitude reported by ``attack_math.normalised_magnitude`` remains in trained-range units."""
    _channels: frozenset = field(init=False)
    _force_state: list = field(init=False)
    _command_state: list = field(init=False)

    def __post_init__(self):
        if not (isinstance(self.scale, (int, float)) and self.scale > 0):
            raise ValueError(f"scale must be positive, got {self.scale!r}")
        self._channels = attack_math.channels_for(self.modality)
        # One 2-D state per env per channel: the rate cap is per-env, so a shared channel object
        # would let one env's history cap another's.
        self._force_state = [[0.0, 0.0] for _ in range(self.num_envs)]
        self._command_state = [[0.0, 0.0] for _ in range(self.num_envs)]

    # -- dimensions, for a runner that expects a VecEnv-shaped object ---------------------------
    @property
    def num_envs(self) -> int:
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def num_actions(self) -> int:
        return 4  # force x/y + command x/y (the paper's force + command pair)

    # -- the rest of the surface an rsl-rl runner reads off a VecEnv ------------------------------
    @property
    def num_obs(self) -> int:
        space = getattr(self.env.unwrapped, "single_observation_space", None)
        if space is not None and getattr(space, "shape", None):
            return int(space.shape[0])
        return int(self.hooks.observations().shape[1])

    @property
    def max_episode_length(self) -> int:
        return int(getattr(self.env.unwrapped, "max_episode_length", 0))

    @property
    def episode_length_buf(self):
        return getattr(self.env.unwrapped, "episode_length_buf", None)

    @property
    def cfg(self):
        """Minimal cfg object rsl-rl reads for logging/seed (``env.cfg["seed"]`` style)."""
        return {"seed": getattr(self, "seed", None), "num_envs": self.num_envs}

    def get_observations(self):
        """Current policy observations WITHOUT stepping (the runner calls this between iterations)."""
        return self.hooks.observations()

    def reset(self):
        obs = self.env.reset()
        self._force_state = [[0.0, 0.0] for _ in range(self.num_envs)]
        self._command_state = [[0.0, 0.0] for _ in range(self.num_envs)]
        self._last_obs = obs
        return obs

    def attack_action_to_channels(self, attack_action) -> tuple[torch.Tensor, torch.Tensor]:
        """Split, clamp and rate-limit the raw 4-D adversary output. Pure apart from tensor ops."""
        if attack_action.shape[-1] != 4:
            raise ValueError(f"expected a 4-D attack action, got {tuple(attack_action.shape)}")
        attack_action = attack_action * self.scale
        zero = torch.zeros_like(attack_action[:, :1])
        force_raw = attack_action[:, :2] if "force" in self._channels else torch.cat([zero, zero], dim=1)
        command_raw = attack_action[:, 2:] if "command" in self._channels else torch.cat([zero, zero], dim=1)
        self._force_state = self._step_channel(
            self._force_state, force_raw, self.bounds.force_peak_n, self.bounds.max_force_delta_per_step
        )
        self._command_state = self._step_channel(
            self._command_state, command_raw, self.bounds.command_peak_mps, self.bounds.max_command_delta_per_step
        )
        force = torch.as_tensor(self._force_state, dtype=torch.float32, device=attack_action.device)
        command = torch.as_tensor(self._command_state, dtype=torch.float32, device=attack_action.device)
        return force, command

    def _step_channel(self, states, raw, peak: float, max_delta: float) -> list:
        """Per-env rate limiting (pure: ``attack_math.rate_limited_step`` row by row)."""
        rows = raw.detach().cpu().tolist()
        if len(rows) != len(states):
            raise ValueError(f"channel state has {len(states)} envs, action has {len(rows)}")
        return [
            attack_math.rate_limited_step(prev, row, peak=peak, max_delta_per_step=max_delta)
            for prev, row in zip(states, rows, strict=True)
        ]

    def step(self, attack_action):
        """One adversary step -> (obs, reward, dones, extras), with the adversary's reward signal."""
        force, command = self.attack_action_to_channels(attack_action)
        self.hooks.apply_force(force)
        self.hooks.apply_command(command)
        obs = self.hooks.observations()
        self._last_obs = obs
        with torch.no_grad():
            actions = self.policy(obs)
            obs, _, dones, extras = self.env.step(actions)
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
        alive = ~(dones.bool() & ~time_outs.bool()) if time_outs is not None else ~dones.bool()
        state = self.hooks.read_state(self.layer_inf_norms_fn() if self.layer_inf_norms_fn else None)
        reward = self._reward(state, alive)
        extras = dict(extras or {})
        extras["attack"] = attack_math.modality_evidence(
            force.mean(dim=0).detach().cpu().tolist(),
            command.mean(dim=0).detach().cpu().tolist(),
            self.bounds,
        )
        return obs, reward, dones, extras

    def _reward(self, state, alive) -> torch.Tensor:
        """Adversary reward per env, via the paper's formula. Loops only over the joint axis."""
        n = self.env.unwrapped.num_envs
        out = torch.zeros(n, device=self.env.unwrapped.device)
        g_z = state["projected_gravity_z"].detach().cpu().tolist()
        ang = state["base_angular_xy"].detach().cpu().tolist()
        torques = state["joint_torques"]
        limits = state["joint_torque_limits"]
        alive_cpu = alive.detach().cpu().tolist()
        for i in range(n):
            row_t = torques[i].detach().cpu().tolist() if torques is not None else []
            row_l = limits[i].detach().cpu().tolist() if limits is not None else []
            if row_t and not row_l:
                row_l = []
                row_t = []
            out[i] = attack_math.adversary_reward(
                alive=bool(alive_cpu[i]),
                projected_gravity_z=g_z[i],
                base_angular_xy=ang[i],
                joint_torques=row_t,
                joint_torque_limits=row_l,
                layer_inf_norms=state.get("layer_inf_norms") or (),
                config=self.reward_config,
            )["reward"]
        return out


def assert_policy_frozen(policy, reference_state: dict | None = None) -> dict:
    """Snapshot/verify a policy's parameters are untouched. Used by the P3a smoke.

    Returns ``{name: tensor.clone()}`` the first time, and raises ``AssertionError`` if any parameter
    or buffer has changed or accumulated a gradient since that snapshot -- i.e. the frozen policy
    really was frozen, which the spec requires evidence for.
    """
    current = {name: p.detach().clone() for name, p in policy.named_parameters()}
    current.update({name: b.detach().clone() for name, b in policy.named_buffers()})
    if reference_state is None:
        return current
    if set(current) != set(reference_state):
        raise AssertionError("policy parameter set changed during adversary training")
    for name, tensor in current.items():
        if not torch.equal(tensor, reference_state[name]):
            raise AssertionError(f"frozen policy parameter {name!r} changed")
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            raise AssertionError(f"frozen policy parameter {name!r} accumulated a gradient")
    return reference_state


def build_attack_env(
    task: str,
    policy,
    bounds: attack_math.PerturbationBounds,
    *,
    num_envs: int = 64,
    modality: str = "combined",
    root_body_name: str = "base",
    play: bool = True,
    env_cfg_overrides: dict | None = None,
):
    """Create the Go2 env and wrap it. Call AFTER AppLauncher has started.

    ``play=True`` selects the Play task (randomisation off) so the attack measures the policy, not a
    randomised physics draw. The training pushes are disabled there too: the adversary is the only
    perturbation in the loop, which is what makes the gate's numbers attributable.
    """
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    import policy.locomotion
    from common.cerberus_compat.cfg_registry import load_env_cfg

    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = num_envs
    for key, value in (env_cfg_overrides or {}).items():
        setattr(env_cfg, key, value)
    env = gym.make(task, cfg=env_cfg)
    hooks = ManagerBasedHooks(env, root_body_name=root_body_name)
    return AttackEnvWrapper(env=env, policy=policy, bounds=bounds, hooks=hooks, modality=modality)
