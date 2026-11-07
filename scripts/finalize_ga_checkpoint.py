#!/usr/bin/env python3
"""Finalize a GA training run into a named candidate: copy the checkpoint, export ONNX, write the sidecar.

H2 P2's checkpoint contract is ``go2_locomotion_ga_v1_candidate.{pt,onnx,meta.json}`` in
``policy/locomotion/checkpoints/``. This script produces all three from a training run directory, and
it is deliberately fussy: it refuses to write over anything (the September checkpoints must stay
byte-identical for the parity comparison), and it refuses to emit a sidecar whose fields it could not
actually measure (wall clock is read from the run's own tensorboard event file, config hash from the
run's dumped ``params/``, physics from the resolved env cfg).

Pod-only for the ONNX half (it needs Isaac to rebuild the runner). The copy-without-clobber and
sidecar halves are plain python, and ``--skip-onnx`` exercises those anywhere.

Usage (on the pod, after the training run finished):

    export TERM=xterm ISAACLAB_PATH=/workspace/isaaclab
    $ISAACLAB_PATH/isaaclab.sh -p scripts/finalize_ga_checkpoint.py \
        --run-dir policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_10-00-00 \
        --name go2_locomotion_ga_v1_candidate \
        --final-reward 19.7 --terrain-stats '{"mean": 5.9, "max": 9, "frac_at_max": 0.147}'

Exit codes: 0 written, 1 failed, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CHECKPOINTS_DIR = _REPO_ROOT / "policy" / "locomotion" / "checkpoints"


def _load_pure_module(relative_path: str, name: str):
    """Load a module by file path, bypassing ``policy.locomotion``'s package import.

    ``policy/locomotion/__init__.py`` runs Isaac Lab's platform auto-discovery, which needs
    ``isaaclab_tasks``. These helpers are pure python and must stay usable without Isaac (CI, a
    workstation, ``--skip-onnx``), so they are loaded straight from their files.
    """
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their defining module through sys.modules
    spec.loader.exec_module(module)
    return module


def newest_checkpoint(run_dir: Path) -> Path:
    """Highest-iteration ``model_<n>.pt`` in ``run_dir`` (ignores ``model_best_*``).

    Determined by the number in the name, not mtime: a copied or touched file must not change which
    checkpoint is the final one.
    """
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        stem = path.stem
        if not stem.startswith("model_"):
            continue
        suffix = stem[len("model_") :]
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"no model_<n>.pt checkpoints in {run_dir}")
    return max(candidates)[1]


def find_event_file(run_dir: Path) -> Path | None:
    events = sorted(run_dir.glob("events.out.tfevents*"))
    return events[-1] if events else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run dir (model_*.pt, params/, events).")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: highest-iteration model_<n>.pt.")
    parser.add_argument("--name", required=True, help="Output stem, e.g. go2_locomotion_ga_v1_candidate.")
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0")
    parser.add_argument(
        "--final-reward",
        type=float,
        default=None,
        help="Final mean reward. Default: read from the run's tensorboard event file (Train/mean_reward).",
    )
    parser.add_argument(
        "--terrain-stats",
        default=None,
        help="JSON dict of final terrain-level stats. Default: read from the event file's Curriculum/* tags.",
    )
    parser.add_argument("--env-report", type=Path, default=_REPO_ROOT / "outputs" / "env_report.json")
    parser.add_argument("--physics", default=None, help="Resolved physics backend, e.g. isaacsim_physx.")
    parser.add_argument("--iterations", type=int, default=None, help="Default: read from params/agent.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Default: read from params/agent.yaml.")
    parser.add_argument("--num-envs", type=int, default=None, help="Default: read from params/env.yaml.")
    parser.add_argument("--skip-onnx", action="store_true", help="Copy + sidecar only (no Isaac needed).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=True)
    args = parser.parse_args(argv)

    # The ONNX export needs a live Kit app; launch it here, with the full launcher namespace, rather
    # than handing AppLauncher a hand-built Namespace (which leaves the physics manager unimportable:
    # "No module named 'omni.physics'").
    if not args.skip_onnx and not args.dry_run:
        from isaaclab.app import AppLauncher

        args.app_launcher = AppLauncher(args)
        args.simulation_app = args.app_launcher.app

    cm = _load_pure_module("policy/locomotion/core/checkpoint_meta.py", "cerberus_checkpoint_meta")
    contract_check = _load_pure_module("policy/locomotion/eval/contract_check.py", "cerberus_contract_check")

    run_dir = args.run_dir.resolve()
    checkpoint = (args.checkpoint or newest_checkpoint(run_dir)).resolve()
    if not checkpoint.is_file():
        print(f"[finalize] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    out_dir = args.out_dir.resolve()
    target_pt = out_dir / f"{args.name}.pt"
    target_onnx = out_dir / f"{args.name}.onnx"
    target_meta = out_dir / f"{args.name}.meta.json"

    try:
        cm.ensure_new_paths([target_pt, target_onnx, target_meta])
    except FileExistsError as e:
        print(f"[finalize] {e}", file=sys.stderr)
        return 1

    # --- attribution: everything the sidecar states must be measured, not assumed ------------------
    reference = contract_check.load_reference(run_dir)
    iterations = args.iterations if args.iterations is not None else reference["max_iterations"]
    seed = args.seed if args.seed is not None else reference["seed"]
    num_envs = args.num_envs if args.num_envs is not None else reference["num_envs"]

    events = find_event_file(run_dir)
    span = cm.tfevents_wall_time_span(events) if events else {"span_s": None}
    if span.get("span_s") is None:
        print(
            "[finalize] no tensorboard event file (or no wall_time in it): cannot state the run's wall clock. "
            "Refusing to invent one -- pass the run dir that actually contains events.out.tfevents*.",
            file=sys.stderr,
        )
        return 1

    env_report = None
    if args.env_report.is_file():
        env_report = json.loads(args.env_report.read_text())
        env_report = {"path": str(args.env_report), "resolved_physics": env_report.get("resolved_physics")}
    physics = args.physics or (env_report or {}).get("resolved_physics") or None
    if physics is None and not args.skip_onnx:
        print("[finalize] physics backend unknown: pass --physics (the sidecar must state it).", file=sys.stderr)
        return 1

    logged = cm.training_summary(events) if events else {}
    final_reward = args.final_reward if args.final_reward is not None else logged.get("final_reward")
    if final_reward is None:
        print(
            "[finalize] no --final-reward and no Train/mean_reward in the event file: refusing to invent the "
            "run's final reward.",
            file=sys.stderr,
        )
        return 1
    if args.terrain_stats:
        terrain_stats = json.loads(args.terrain_stats)
    else:
        terrain_stats = logged.get("terrain_levels")
        if terrain_stats is None:
            print("[finalize] WARNING: run logged no Curriculum/terrain_* tags; terrain stats will be null")
        elif "fall_rate" in logged and terrain_stats is not None:
            terrain_stats = {**terrain_stats, "final_fall_rate": logged["fall_rate"]}
    print(f"[finalize] final reward: {final_reward:.4f} (from event file: {args.final_reward is None})")

    print(f"[finalize] run dir     : {run_dir}")
    print(f"[finalize] checkpoint  : {checkpoint.name}")
    print(f"[finalize] wall clock  : {span['span_s'] / 3600:.2f} h ({span['events']} events)")
    print(f"[finalize] iterations  : {iterations}  seed: {seed}  num_envs: {num_envs}")
    print(f"[finalize] physics     : {physics}")
    print(f"[finalize] config hash : {cm.config_hash(run_dir / 'params')}")
    print(f"[finalize] outputs     : {target_pt.name}, {target_onnx.name}, {target_meta.name}")

    if args.dry_run:
        print("[finalize] --dry-run: nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy first: the source run keeps its own checkpoint, and the named candidate is a new file.
    shutil.copy2(checkpoint, target_pt)
    print(f"[finalize] copied -> {target_pt}")

    if not args.skip_onnx:
        try:
            _export_onnx(checkpoint, target_onnx, args.task)
        except Exception as e:  # noqa: BLE001 -- report, then leave the .pt + sidecar in place for a retry
            print(f"[finalize] ONNX export failed: {type(e).__name__}: {e}", file=sys.stderr)
            print("[finalize] checkpoint and sidecar are still valid; re-run only the export once fixed.", file=sys.stderr)
            return 1

    meta = cm.build_meta(
        checkpoint=str(target_pt),
        onnx=str(target_onnx) if target_onnx.exists() else None,
        git_hash_value=cm.git_hash(_REPO_ROOT),
        task_id=args.task,
        physics_backend=physics,
        iterations=int(iterations),
        seed=seed,
        num_envs=int(num_envs),
        wall_clock_s=span["span_s"],
        final_reward=final_reward,
        config_hash_value=cm.config_hash(run_dir / "params"),
        env_report=env_report,
        terrain_levels=terrain_stats,
        run_dir=str(run_dir),
        extra={"source_checkpoint": str(checkpoint), "event_file": str(events) if events else None},
    )
    cm.write_meta(target_meta, meta, checkpoint_paths=[target_pt])
    print(f"[finalize] wrote -> {target_meta}")
    return 0


def _export_onnx(checkpoint: Path, target_onnx: Path, task: str) -> None:  # pragma: no cover - pod only
    """Rebuild the runner around ``checkpoint`` and export its actor to ``target_onnx``."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import policy.locomotion  # noqa: F401
    from common.cerberus_compat.cfg_registry import (
        load_agent_cfg,
        load_env_cfg,
    )
    from common.cerberus_compat.rsl_rl_compat import (
        check_checkpoint_format,
        export_policy,
        get_policy_for_export,
        make_runner,
    )

    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = 1
    env = gym.make(task, cfg=env_cfg)
    agent_cfg = load_agent_cfg(task)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    fmt = check_checkpoint_format(str(checkpoint))
    print(f"[finalize] checkpoint format: {fmt}")
    runner = make_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    get_policy_for_export(runner)  # raises if the runner exposes no exportable actor
    # rsl-rl >= 4 exports directly; the JIT file is a by-product of the same call, kept beside the ONNX
    # so the exact deployed module can be inspected later.
    staging = target_onnx.parent / f"{target_onnx.stem}.export"
    staging.mkdir(exist_ok=True)
    export_policy(runner, str(staging), jit_filename=f"{target_onnx.stem}.torchscript.pt", onnx_filename=target_onnx.name)
    produced = staging / target_onnx.name
    if not produced.is_file():
        raise RuntimeError(f"export produced no ONNX at {produced}")
    shutil.move(str(produced), str(target_onnx))
    os.replace(staging / f"{target_onnx.stem}.torchscript.pt", target_onnx.with_suffix(".torchscript.pt"))
    staging.rmdir()
    print(f"[finalize] exported -> {target_onnx}")
    env.close()


if __name__ == "__main__":
    sys.exit(main())
