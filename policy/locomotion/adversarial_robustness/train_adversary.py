#!/usr/bin/env python3
"""Train the Phase 0.5 adversary against a FROZEN locomotion candidate.

Standard rsl-rl PPO, but the environment's action is an attack (push force + command override) and
its reward is the paper's adversary objective (see ``attack_math.adversary_reward``). The locomotion
policy is a fixed component: it is loaded once, its parameters are snapshotted, and
``attack_env.assert_policy_frozen`` is used before and after training to prove they never moved.

The Lipschitz regularisation from the paper is the one part rsl-rl cannot express as a loss term (it
is a property of the policy weights, not of a transition), so it is applied as an env-side reward
term through the wrapper's ``layer_inf_norms_fn``. That is also why the adversary's actor is an
ordinary MLP from the rsl-rl cfg rather than a bespoke class: the wrapper needs to read the weights it
is penalising.

POD-ONLY. The P3a smoke (``--dry-run`` here plus a zero-perturbation rollout) must pass before spending
the 3-hour cap: it verifies the env's dimensions, that the frozen policy is untouched, and that a
zero attack reproduces the candidate's standard success rate within noise.

Usage:
    export TERM=xterm ISAACLAB_PATH=/workspace/isaaclab
    $ISAACLAB_PATH/isaaclab.sh -p policy/locomotion/adversarial_robustness/train_adversary.py \
        --task Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0 \
        --checkpoint policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt \
        --num_envs 256 --iterations 1000 --runner_cfg policy/locomotion/adversarial_robustness/adversary_cfg.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Paper: 3 hidden layers for the Miki-style (push-randomised) policy, 2 for the didactic case.
ADVERSARY_HIDDEN_DIMS = (256, 128, 64)
ATTACK_ACTION_DIM = 4  # force x/y + command x/y
DEFAULT_RUNNER_CFG = {
    "class_name": "OnPolicyRunner",
    "num_steps_per_env": 24,
    "max_iterations": 1000,
    "save_interval": 25,
    "empirical_normalization": None,
    "actor": {
        "class_name": "MLP",
        "hidden_dims": list(ADVERSARY_HIDDEN_DIMS),
        "activation": "elu",
        "obs_normalization": False,
    },
    "critic": {
        "class_name": "MLP",
        "hidden_dims": list(ADVERSARY_HIDDEN_DIMS),
        "activation": "elu",
        "obs_normalization": False,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.01,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0")
    parser.add_argument("--checkpoint", required=True, help="Frozen candidate .pt (GA-trained only).")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modality", default="combined", choices=("force_only", "command_only", "combined"))
    parser.add_argument("--out_dir", default=None, help="Default: policy/locomotion/checkpoints/adversary_<stamp>")
    parser.add_argument("--runner_cfg", default=None, help="JSON file overriding DEFAULT_RUNNER_CFG.")
    parser.add_argument("--dry_run", action="store_true", help="Build everything, step zero attack, exit.")
    parser.add_argument("--max_wall_clock_h", type=float, default=3.0, help="Hard cap; the trainer stops at it.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--enable_cameras",
        action="store_true",
        default=True,
        help="Keep cameras enabled: on the v6 images a launch with them disabled dies at startup.",
    )
    return parser.parse_args(argv)


def load_frozen_policy(task: str, checkpoint: str):
    """Load a GA checkpoint's actor as an ``obs -> actions`` callable (never trained here)."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    import policy.locomotion
    from common.cerberus_compat.cfg_registry import (
        load_agent_cfg,
        load_env_cfg,
    )
    from common.cerberus_compat.rsl_rl_compat import (
        check_checkpoint_format,
        make_runner,
    )

    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = 1  # the runner only needs an env to construct; the wrapper drives the real one
    agent_cfg = load_agent_cfg(task)
    env = gym.make(task, cfg=env_cfg)
    fmt = check_checkpoint_format(checkpoint)
    print(f"[adversary] frozen policy checkpoint format: {fmt}")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = make_runner(wrapped, agent_cfg, log_dir=None, device=getattr(agent_cfg, "device", "cuda:0"))
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    env.close()
    return policy


def layer_inf_norms(policy_nn):
    """Infinity norm of each linear layer's weights -- the term the paper's lambda penalises."""
    import torch

    norms = []
    for module in policy_nn.modules():
        if isinstance(module, torch.nn.Linear):
            norms.append(float(module.weight.detach().abs().max()))
    return norms


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841 -- keeps Kit alive for the whole run

    try:
        return _run(args)
    except Exception:  # noqa: BLE001 -- Kit exits 0 on close, so a traceback here is the only evidence
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # SimulationApp.close() exits 0 regardless of what happened (docs/runpod.md) -- never mask it.
        sys.stdout.flush()


def _run(args) -> int:
    import time

    from common.cerberus_compat.cfg_registry import load_env_cfg
    from common.cerberus_compat.rsl_rl_compat import make_runner
    from policy.locomotion.adversarial_robustness import attack_env, attack_math
    from policy.locomotion.core.platforms.go2.platform import GO2

    env_cfg = load_env_cfg(args.task)
    ranges = env_cfg.commands.base_velocity.ranges
    bounds = attack_math.PerturbationBounds.from_config(GO2, ranges)
    print(f"[adversary] bounds: force {bounds.force_peak_n} N, command {bounds.command_peak_mps:.3f} m/s, "
          f"rate {bounds.rate_limit_fraction} of range per step")

    policy = load_frozen_policy(args.task, args.checkpoint)
    frozen_snapshot = attack_env.assert_policy_frozen(policy)
    wrapper = attack_env.build_attack_env(
        args.task, policy, bounds, num_envs=args.num_envs, modality=args.modality, play=True
    )

    runner_cfg = json.loads(json.dumps(DEFAULT_RUNNER_CFG))
    runner_cfg["max_iterations"] = args.iterations
    if args.runner_cfg:
        runner_cfg.update(json.loads(Path(args.runner_cfg).read_text()))
    runner_cfg.setdefault("seed", args.seed)

    if args.dry_run:
        return _dry_run(wrapper, runner_cfg, frozen_snapshot)

    # The runner is constructed BEFORE the Lipschitz callback is attached: it owns the actor, and the
    # callback has to read those weights.
    runner = make_runner(wrapper, runner_cfg, log_dir=args.out_dir, device=wrapper.device)
    adversary_nn = runner.alg.get_policy()
    wrapper.layer_inf_norms_fn = lambda: layer_inf_norms(adversary_nn)

    start = time.time()
    cap_s = args.max_wall_clock_h * 3600.0
    stop_reason = "iterations complete"
    for iteration in range(args.iterations):
        runner.learn(num_learning_iterations=1)
        elapsed = time.time() - start
        if iteration % 10 == 0:
            print(f"[adversary] iter {iteration}/{args.iterations}  elapsed {elapsed / 60:.1f} min")
        if elapsed > cap_s:
            stop_reason = f"wall-clock cap {args.max_wall_clock_h} h reached at iteration {iteration}"
            print(f"[adversary] STOP: {stop_reason}")
            break

    attack_env.assert_policy_frozen(policy, frozen_snapshot)  # raises if the candidate moved at all
    print("[adversary] frozen locomotion policy verified unchanged")

    elapsed = time.time() - start
    manifest = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "modality": args.modality,
        "num_envs": args.num_envs,
        "iterations_requested": args.iterations,
        "iterations_completed": iteration + 1,
        "stop_reason": stop_reason,
        "wall_clock_s": round(elapsed, 1),
        "bounds": {
            "force_peak_n": bounds.force_peak_n,
            "command_peak_mps": bounds.command_peak_mps,
            "rate_limit_fraction": bounds.rate_limit_fraction,
        },
        "runner_cfg": runner_cfg,
        "out_dir": args.out_dir,
    }
    out_dir = Path(args.out_dir or f"policy/locomotion/checkpoints/adversary_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[adversary] wrote {out_dir / 'train_manifest.json'}")
    return 0


def _dry_run(wrapper, runner_cfg, frozen_snapshot) -> int:
    """P3a smoke: one zero-perturbation rollout + shape checks, no training, no time spent."""
    import torch

    from policy.locomotion.adversarial_robustness import attack_env

    obs = wrapper.reset()
    zero = torch.zeros((wrapper.num_envs, ATTACK_ACTION_DIM), device=wrapper.device)
    print(f"[adversary] dry run: obs dim {tuple(obs.shape)}, attack dim {ATTACK_ACTION_DIM}, "
          f"runner cfg class {runner_cfg['class_name']}")
    nan_seen = False
    for step in range(200):
        obs, reward, _dones, extras = wrapper.step(zero)
        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
            nan_seen = True
            print(f"[adversary] NON-FINITE value at step {step} -- FAIL")
            break
    attack_env.assert_policy_frozen(wrapper.policy, frozen_snapshot)
    attack = extras.get("attack", {})
    print(f"[adversary] dry run: zero perturbation -> combined magnitude {attack.get('combined')}, "
          f"force {attack.get('force_n')} N, command {attack.get('command_mps')} m/s")
    print("[adversary] dry run PASS" if not nan_seen else "[adversary] dry run FAIL")
    return 0 if not nan_seen else 1


if __name__ == "__main__":
    sys.exit(main())
