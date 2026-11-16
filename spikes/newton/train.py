"""Retrain the Cerberus Go2 rough task on Newton from scratch: stock recipe, 4096 envs, 1500 iterations.

Same MDP as the PhysX retrain (``UnitreeGo2RoughCerberusEnvCfg``: push disturbances, the push curriculum,
the stair scaling), same rsl_rl PPO preset, seed 42; only the physics backend and the height-scan sensor are
Newton's. Trained in chunks so a stall can be detected early and so per-chunk timings are recorded; clips are
recorded afterwards from the checkpoints saved at each chunk boundary (rendering inside the training loop
would dominate the run time -- measured: 265 s for 300 rendered steps vs 1.9 s per 24-step training
iteration).

    python -m spikes.newton.train --out <dir> --num_envs 4096 --iterations 1500 --seed 42
"""

from __future__ import annotations

import math
import shutil
import sys
import time

from spikes.newton import common, tasks

CANDIDATE = "go2_newton_locomotion_v1_candidate"


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--stall_after", type=int, default=300,
                        help="abort if the run has not produced a finite reward by this many iterations")
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()
    report: dict = {"stage": "train", "args": vars(args), "versions": common.software_versions()}

    cfg, info = tasks.build_cfg(variant="train", num_envs=args.num_envs, joint_ordering="none",
                               seed=args.seed, episode_length_s=20.0)
    report["cfg_info"] = info
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym
    import torch

    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except ImportError:
        from isaaclab_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg

    from common.cerberus_compat.rsl_rl_compat import make_runner

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_ID, cfg=cfg)
        timer.mark("env_built")
        agent_cfg = UnitreeGo2RoughPPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.max_iterations = args.iterations
        agent_cfg.save_interval = args.save_interval
        agent_cfg.experiment_name = CANDIDATE
        run_dir = out / "train" / CANDIDATE
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        timer.mark("vec_wrapped")

        torch.cuda.reset_peak_memory_stats()
        runner = make_runner(vec, agent_cfg, log_dir=str(run_dir), device=str(cfg.sim.device))
        timer.mark("runner_ready")
        report["num_envs"] = int(args.num_envs)
        report["learning_rate"] = getattr(agent_cfg.algorithm, "learning_rate", None)
        report["num_steps_per_env"] = getattr(agent_cfg, "num_steps_per_env", None)

        chunks = []
        done = 0
        stalled = False
        first = True
        while done < args.iterations:
            n = min(args.chunk, args.iterations - done)
            t0 = time.time()
            runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
            first = False
            done += n
            wall = time.time() - t0
            latest = _latest_checkpoint(run_dir)
            finite, bad_keys = _finite(latest)
            reward, terrain = _curve(run_dir)
            chunks.append({"up_to_iteration": done, "iterations": n, "wall_s": round(wall, 1),
                           "s_per_iteration": round(wall / n, 3), "checkpoint": str(latest) if latest else None,
                           "params_finite": finite, "nan_keys": bad_keys[:5], "reward": reward})
            timer.mark(f"chunk_{done}")
            print(f"[train] {done}/{args.iterations} iters, {wall / n:.2f} s/iter, finite={finite}, "
                  f"reward={reward}, terrain={terrain}", flush=True)
            report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if not finite or (reward is not None and math.isnan(reward)):
                stalled = True
                print("[train] STALLED: non-finite parameters or reward; stopping", flush=True)
                break
        timer.mark("learn")
        report["chunks"] = chunks
        report["iterations_done"] = done
        report["stalled"] = stalled
        report["train_wall_s"] = round(sum(c["wall_s"] for c in chunks), 1)
        report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        report["final_reward"], report["terrain_levels"] = _curve(run_dir)

        if done >= args.stall_after and not stalled:
            latest = _latest_checkpoint(run_dir)
            candidate = out / f"{CANDIDATE}.pt"
            shutil.copyfile(latest, candidate)
            onnx = out / f"{CANDIDATE}.onnx"
            try:
                runner.export_policy_to_onnx(path=str(out), filename=onnx.name)
                exported = True
            except Exception as exc:  # noqa: BLE001 - recorded, the checkpoint still stands
                exported = f"{type(exc).__name__}: {exc}"
            timer.mark("export")
            report["export"] = {"checkpoint": str(candidate), "onnx": str(onnx) if exported is True else None,
                                "onnx_error": None if exported is True else exported}
            report["meta"] = _meta(run_dir, candidate, out, args, report)
        env.close()

    report["timings"] = timer.as_dict()
    common.write_json(out / "train.json", report)
    print(f"[train] done={report['iterations_done']} stalled={report['stalled']} "
          f"wall={report['train_wall_s']}s peak_vram={report['peak_vram_mib']}MiB", flush=True)
    return 0


def _latest_checkpoint(run_dir):
    from pathlib import Path

    files = sorted(Path(run_dir).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return files[-1] if files else None


def _finite(path) -> tuple[bool, list[str]]:
    if path is None:
        return False, ["no checkpoint"]
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict") or {}
    bad = [k for k, v in state.items() if hasattr(v, "isfinite") and not bool(torch.isfinite(v).all())]
    return (not bad), bad


def _curve(run_dir) -> tuple[float | None, dict | None]:
    """Final mean reward and terrain-level stats from the run's tensorboard events."""
    from pathlib import Path

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:  # noqa: BLE001
        return None, None
    events = sorted(Path(run_dir).glob("events.out.tfevents*"))
    if not events:
        return None, None
    acc = EventAccumulator(str(events[-1]))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    reward = None
    for tag in ("Train/mean_reward", "Loss/mean_reward", "Train/mean_reward/time"):
        if tag in tags:
            reward = float(acc.Scalars(tag)[-1].value)
            break
    terrain = {}
    for tag in tags:
        if "terrain" in tag.lower():
            terrain[tag] = float(acc.Scalars(tag)[-1].value)
    return reward, (terrain or None)


def _meta(run_dir, candidate, out, args, report) -> dict:
    """The sidecar the repo's checkpoint tooling expects, plus the Newton-specific fields."""
    import hashlib
    import json

    from policy.locomotion.core.checkpoint_meta import build_meta, git_hash

    solver = dict(report.get("cfg_info", {}))
    config_blob = json.dumps(solver, sort_keys=True).encode()
    meta = build_meta(
        checkpoint=str(candidate),
        task_id=tasks.NEWTON_TASK_ID,
        physics_backend={"preset": "newton_mjwarp", "solver": "MJWarpSolverCfg",
                         "stock_values": _solver_values(), "num_substeps": solver.get("num_substeps")},
        iterations=int(report["iterations_done"]),
        seed=int(args.seed),
        num_envs=int(args.num_envs),
        wall_clock_s=float(report["train_wall_s"]),
        final_reward=float(report["final_reward"] or 0.0),
        config_hash_value=hashlib.sha256(config_blob).hexdigest()[:16],
        git_hash_value=git_hash(str(common.CODE_ROOT)),
        onnx=str(out / f"{CANDIDATE}.onnx"),
        terrain_levels=report.get("terrain_levels"),
        run_dir=str(run_dir),
        extra={"versions": report["versions"], "peak_vram_mib": report.get("peak_vram_mib"),
               "s_per_iteration": round(report["train_wall_s"] / max(report["iterations_done"], 1), 3),
               "phase": "newton retrain, phase 5", "recipe": "stock rsl_rl PPO preset, seed 42"},
    )
    (out / f"{CANDIDATE}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[train] wrote {out / (CANDIDATE + '.meta.json')}", flush=True)
    return meta


def _solver_values() -> dict:
    """The stock preset's contact-solver values, as recorded for the report."""
    from policy.locomotion.core.platforms.go2.env_cfg import UnitreeGo2RoughCerberusEnvCfg_PLAY

    newton = UnitreeGo2RoughCerberusEnvCfg_PLAY().sim.physics.newton_mjwarp
    solver = newton.solver_cfg
    shape = getattr(newton, "default_shape_cfg", None)
    return {"njmax": getattr(solver, "njmax", None), "nconmax": getattr(solver, "nconmax", None),
            "cone": getattr(solver, "cone", None), "impratio": getattr(solver, "impratio", None),
            "integrator": getattr(solver, "integrator", None),
            "num_substeps": getattr(newton, "num_substeps", None),
            "shape": {"margin": getattr(shape, "margin", None), "ke": getattr(shape, "ke", None),
                      "kd": getattr(shape, "kd", None)} if shape is not None else None}


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
