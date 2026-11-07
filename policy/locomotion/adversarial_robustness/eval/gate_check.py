#!/usr/bin/env python3
"""Phase 0.5 gate evaluation: the minimum perturbation that breaks the candidate, per modality.

The gate is relative (``plans/handoff_phase0.5.md``): the minimum *combined* force+command magnitude
that induces a fall must exceed the push range the candidate was trained and evaluated against
(normalised to 1.0, see ``attack_math``). This script finds that minimum by reduce-until-survives --
bisecting the wrapper's ``scale`` -- and writes ``gate_verdict.json`` with the numbers, plus the same
search for force-only and command-only so the report can show whether the combination is what matters
(the source paper's Table IV prediction) rather than asserting it.

Output goes to ``policy/locomotion/adversarial_robustness/eval/results/<stamp>/gate_verdict.json`` by
default (the layout in ``docs/project_structure.md`` puts the module's eval artifacts under it).

POD-ONLY. Run the P3a smoke first: with zero perturbation this must reproduce the candidate's standard
success rate within noise, otherwise the wrapper is measuring itself, not the policy.

Usage:
    $ISAACLAB_PATH/isaaclab.sh -p policy/locomotion/adversarial_robustness/eval/gate_check.py \
        --checkpoint policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt \
        --adversary_dir policy/locomotion/checkpoints/adversary_<stamp> \
        --trials 100 --max_scale 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0")
    parser.add_argument("--checkpoint", required=True, help="Candidate .pt that is being gated.")
    parser.add_argument("--adversary_dir", required=True, help="Run dir with the trained adversary model_*.pt.")
    parser.add_argument("--modalities", default="combined,force_only,command_only")
    parser.add_argument("--trials", type=int, default=100, help="Trials (= envs) per probed scale.")
    parser.add_argument("--max_scale", type=float, default=2.0, help="Largest allowed multiplier probed.")
    parser.add_argument("--min_scale", type=float, default=0.05)
    parser.add_argument("--fall_threshold", type=float, default=0.5, help="Fall rate counted as 'induced a fall'.")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Bisection stop width on the scale.")
    parser.add_argument("--zero_attack_check", action="store_true", help="Also record the zero-perturbation fall rate.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args(argv)


def newest_checkpoint(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        suffix = path.stem[len("model_") :]
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no model_<n>.pt in {run_dir}")
    return max(candidates)[1]


def rollout_fall_rate(wrapper, adversary, trials: int, max_steps: int = 400) -> dict:
    """Run one episode per env under the adversary; return the fall rate.

    Falls are counted with the env's own termination signal (never a new criterion), and time-outs are
    excluded exactly as the push-recovery eval does -- otherwise every trial would count as a fall when
    its episode ends.
    """
    import torch

    obs = wrapper.reset()
    fell = torch.zeros(wrapper.num_envs, dtype=torch.bool, device=wrapper.device)
    finished = torch.zeros(wrapper.num_envs, dtype=torch.bool, device=wrapper.device)
    nan_seen = False
    for _ in range(max_steps):
        with torch.no_grad():
            action = adversary(obs)
        obs, _reward, dones, extras = wrapper.step(action)
        if not torch.isfinite(obs).all():
            nan_seen = True
            break
        time_outs = extras.get("time_outs")
        terminated = dones.bool() & (~time_outs.bool() if time_outs is not None else True)
        fell |= terminated & ~finished
        finished |= dones.bool()
        if bool(finished.all()):
            break
    n = int(wrapper.num_envs)
    return {"fall_rate": float(fell.sum()) / n, "trials": n, "nan": nan_seen}


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841
    try:
        return _run(args)
    except Exception:  # noqa: BLE001 -- never let SimulationApp.close()'s exit(0) hide a failure
        import traceback

        traceback.print_exc()
        return 1


def _run(args) -> int:

    from common.cerberus_compat.cfg_registry import load_env_cfg
    from policy.locomotion.adversarial_robustness import attack_env, attack_math
    from policy.locomotion.adversarial_robustness.train_adversary import (
        load_frozen_policy,
    )
    from policy.locomotion.core.platforms.go2.platform import GO2

    env_cfg = load_env_cfg(args.task)
    bounds = attack_math.PerturbationBounds.from_config(GO2, env_cfg.commands.base_velocity.ranges)
    policy = load_frozen_policy(args.task, args.checkpoint)
    frozen_snapshot = attack_env.assert_policy_frozen(policy)
    adversary_checkpoint = newest_checkpoint(Path(args.adversary_dir))

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    results = {}
    for modality in modalities:
        print(f"\n[gate] modality: {modality}")
        wrapper = attack_env.build_attack_env(
            args.task, policy, bounds, num_envs=args.trials, modality=modality, play=True
        )
        adversary = _load_adversary(wrapper, adversary_checkpoint, args)

        def evaluate(scale, _wrapper=wrapper, _adversary=adversary):
            _wrapper.scale = scale
            return rollout_fall_rate(_wrapper, _adversary, args.trials)

        search = attack_math.find_minimum_inducing(
            evaluate,
            max_scale=args.max_scale,
            min_scale=args.min_scale,
            fall_threshold=args.fall_threshold,
            tolerance=args.tolerance,
        )
        minimum = search.get("scale") if search.get("found") else None
        evidence = attack_math.modality_evidence(
            [bounds.force_peak_n * (minimum or 0.0), 0.0],
            [bounds.command_peak_mps * (minimum or 0.0), 0.0],
            bounds,
        )
        results[modality] = {
            "minimum_score": minimum,
            "fall_rate_at_minimum": search.get("fall_rate"),
            "search": search,
            "magnitude_at_minimum": evidence,
        }
        print(f"[gate] {modality}: minimum score {minimum!r} (fall rate {search.get('fall_rate')})")

        if args.zero_attack_check:
            wrapper.scale = 0.0
            results[modality]["zero_attack"] = rollout_fall_rate(wrapper, adversary, args.trials)
            print(f"[gate] {modality}: zero-perturbation fall rate {results[modality]['zero_attack']['fall_rate']:.1%}")
        wrapper.env.close()

    attack_env.assert_policy_frozen(policy, frozen_snapshot)
    summary = attack_math.summarise_modalities(results)
    combined = results.get("combined", {})
    decision = attack_math.gate_decision(combined.get("minimum_score"), trained_range_score=1.0)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "verdict": decision["verdict"],
        "reason": decision["reason"],
        "minimum_inducing_score": decision.get("minimum_inducing_score"),
        "trained_range_score": decision.get("trained_range_score"),
        "margin": decision.get("margin"),
        "candidate_checkpoint": args.checkpoint,
        "adversary_checkpoint": str(adversary_checkpoint),
        "task": args.task,
        "bounds": {
            "force_peak_n": bounds.force_peak_n,
            "command_peak_mps": bounds.command_peak_mps,
            "rate_limit_fraction": bounds.rate_limit_fraction,
            "force_peak_torque_nm": bounds.force_peak_torque_nm,
        },
        "search": {
            "trials": args.trials,
            "fall_threshold": args.fall_threshold,
            "max_scale": args.max_scale,
            "min_scale": args.min_scale,
            "tolerance": args.tolerance,
        },
        "modalities": results,
        "summary": summary,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_clock_s": None,
    }
    path = out_dir / "gate_verdict.json"
    path.write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    print(f"\n[gate] verdict: {decision['verdict']} -- {decision['reason']}")
    print(f"[gate] wrote {path}")
    return 0


def _load_adversary(wrapper, checkpoint: Path, args):
    """Load the trained adversary through a runner built on this wrapper (never trained here)."""
    import json as _json

    from common.cerberus_compat.rsl_rl_compat import (
        check_checkpoint_format,
        make_runner,
    )
    from policy.locomotion.adversarial_robustness import train_adversary

    manifest_path = Path(args.adversary_dir) / "train_manifest.json"
    runner_cfg = _json.loads(_json.dumps(train_adversary.DEFAULT_RUNNER_CFG))
    if manifest_path.is_file():
        stored = _json.loads(manifest_path.read_text()).get("runner_cfg")
        if stored:
            runner_cfg = stored
    check_checkpoint_format(str(checkpoint))
    runner = make_runner(wrapper, runner_cfg, log_dir=None, device=wrapper.device)
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=wrapper.device)


if __name__ == "__main__":
    sys.exit(main())
