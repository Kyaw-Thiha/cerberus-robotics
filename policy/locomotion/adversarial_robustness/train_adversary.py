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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0")
    parser.add_argument("--checkpoint", required=True, help="Frozen candidate .pt (GA-trained only).")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modality", default="combined", choices=("force_only", "command_only", "combined"))
    parser.add_argument("--out_dir", default=None, help="Default: policy/locomotion/checkpoints/adversary_<stamp>")
    parser.add_argument("--runner_cfg", default=None, help="JSON file overriding the derived runner config.")
    parser.add_argument("--dry_run", action="store_true", help="Build everything, step zero attack, exit.")
    parser.add_argument("--max_wall_clock_h", type=float, default=3.0, help="Hard cap; the trainer stops at it.")
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=True)
    return parser.parse_args(argv)


def load_frozen_policy(task: str, checkpoint: str):
    """Load a GA checkpoint's actor as an ``obs -> actions`` callable (never trained here)."""
    import importlib

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    # Imported by name, not `import policy.locomotion`: "policy" is also this module's return value.
    importlib.import_module("policy.locomotion")
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


def load_adversary_policy(env, run_dir: str):
    """Rebuild a trained adversary from its run dir and return it as an inference policy.

    Uses the runner config stored in that run's manifest: the adversary's shape and observation groups
    have to match exactly what was trained, so they are read back rather than re-derived.
    """
    import json
    from pathlib import Path

    from common.cerberus_compat.rsl_rl_compat import (
        check_checkpoint_format,
        make_runner,
    )

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    runner_cfg = manifest["runner_cfg"]
    checkpoints = sorted(
        (p for p in run_dir.glob("model_*.pt") if p.stem[len("model_"):].isdigit()),
        key=lambda p: int(p.stem[len("model_"):]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no model_<n>.pt in {run_dir}")
    checkpoint = checkpoints[-1]
    check_checkpoint_format(str(checkpoint))
    runner = make_runner(env, runner_cfg, log_dir=None, device=env.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy


def _runner_cfg_from_task(task: str, iterations: int, seed: int) -> dict:
    """The task's PPO runner config as a dict, retargeted at the adversary's MLP and budget."""
    from common.cerberus_compat.cfg_registry import load_agent_cfg
    from common.cerberus_compat.rsl_rl_compat import migrate_agent_cfg

    agent_cfg = migrate_agent_cfg(load_agent_cfg(task))
    cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else json.loads(json.dumps(agent_cfg.__dict__))
    for model in ("actor", "critic"):
        block = cfg.get(model)
        if isinstance(block, dict) and "hidden_dims" in block or isinstance(block, dict):
            block["hidden_dims"] = list(ADVERSARY_HIDDEN_DIMS)
    cfg["max_iterations"] = int(iterations)
    cfg["seed"] = int(seed)
    cfg["save_interval"] = min(25, max(1, int(iterations)))
    cfg["experiment_name"] = "unitree_go2_adversary"
    return cfg


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

    # Start from the task's own runner config (migrated for the installed rsl-rl), then change only the
    # parts that make it an adversary: a smaller MLP, this iteration count, this seed. Hand-writing the
    # cfg dict would silently miss keys rsl-rl requires (obs_groups, logger, device, ...).
    runner_cfg = _runner_cfg_from_task(args.task, args.iterations, args.seed)
    if args.runner_cfg:
        runner_cfg.update(json.loads(Path(args.runner_cfg).read_text()))

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
    # Train in blocks, not one iteration at a time: rsl-rl only writes a checkpoint on its own
    # iteration counter, and calling learn(1) repeatedly keeps that counter at 0 -- the loop ran but
    # produced nothing to evaluate. Each block is followed by an explicit save, so an interrupted
    # run still leaves usable weights.
    save_every = 50
    completed = 0
    out_dir = Path(args.out_dir or f"policy/locomotion/checkpoints/adversary_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    while completed < args.iterations:
        block = min(save_every, args.iterations - completed)
        runner.learn(num_learning_iterations=block)
        completed += block
        elapsed = time.time() - start
        runner.save(str(out_dir / f"model_{completed}.pt"))
        print(f"[adversary] iter {completed}/{args.iterations}  elapsed {elapsed / 60:.1f} min  "
              f"saved model_{completed}.pt")
        if elapsed > cap_s:
            stop_reason = f"wall-clock cap {args.max_wall_clock_h} h reached at iteration {completed}"
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
        "iterations_completed": completed,
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
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[adversary] wrote {out_dir / 'train_manifest.json'}")
    return 0


def _dry_run(wrapper, runner_cfg, frozen_snapshot) -> int:
    """P3a smoke: one zero-perturbation rollout + shape checks, no training, no time spent."""
    import torch

    from policy.locomotion.adversarial_robustness import attack_env
    from policy.locomotion.eval.policy_io import select_policy_obs

    obs = wrapper.reset()
    zero = torch.zeros((wrapper.num_envs, ATTACK_ACTION_DIM), device=wrapper.device)
    print(f"[adversary] dry run: obs dim {tuple(select_policy_obs(obs).shape)}, "
          f"attack dim {ATTACK_ACTION_DIM}, runner cfg class {runner_cfg['class_name']}")
    nan_seen = False
    for step in range(200):
        obs, reward, _dones, extras = wrapper.step(zero)
        if not torch.isfinite(select_policy_obs(obs)).all() or not torch.isfinite(reward).all():
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
