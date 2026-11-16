"""Turn a saved Newton retrain checkpoint into the candidate artifacts: .pt, .onnx, meta JSON.

Used when a training run is cut short: the checkpoints are already in R2, and the candidate name, the ONNX
export and the sidecar are produced from whichever checkpoint is being promoted (here ``model_1250.pt``,
iteration 1312 of the run's own counter).

    python -m spikes.newton.export_candidate --checkpoint <model_N.pt> --run-dir <tfevents dir> \
        --out <dir> --iterations 1250
"""

from __future__ import annotations

import shutil
import sys

from spikes.newton import common, tasks
from spikes.newton.train import CANDIDATE, _curve

#: the run's own log said the last logged iteration was 1312 of 1495, from a 1500-iteration target
ITERS_COMPLETED_NOTE = "checkpoint model_1250.pt; the run's own counter had reached iteration 1312"


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None, help="the training run dir, for the reward/terrain curves")
    parser.add_argument("--iterations", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    report: dict = {"stage": "export_candidate", "args": vars(args), "versions": common.software_versions()}

    cfg, info = tasks.build_cfg(num_envs=max(args.num_envs, 16), joint_ordering="none", seed=args.seed,
                                episode_length_s=22.0)
    report["cfg_info"] = info
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym

    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except ImportError:
        from isaaclab_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg

    from common.cerberus_compat.rsl_rl_compat import make_runner

    with launch_simulation(cfg, args):
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        agent_cfg = UnitreeGo2RoughPPORunnerCfg()
        agent_cfg.seed = args.seed
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = make_runner(vec, agent_cfg, log_dir=None, device=str(cfg.sim.device))
        runner.load(args.checkpoint)
        candidate = out / f"{CANDIDATE}.pt"
        shutil.copyfile(args.checkpoint, candidate)
        onnx_name = f"{CANDIDATE}.onnx"
        onnx_error = None
        try:
            runner.export_policy_to_onnx(path=str(out), filename=onnx_name)
        except Exception as exc:  # noqa: BLE001 - recorded, the checkpoint still stands
            onnx_error = f"{type(exc).__name__}: {exc}"
        env.close()

        reward, terrain = _curve(args.run_dir) if args.run_dir else (None, None)
        meta = _meta(out, candidate, args, reward, terrain, info)
        report.update({"checkpoint": str(candidate), "onnx": str(out / onnx_name) if not onnx_error else None,
                       "onnx_error": onnx_error, "final_reward": reward, "terrain_levels": terrain, "meta": meta})
        common.write_json(out / "export.json", report)
        print(f"[export] {candidate} onnx={'ok' if not onnx_error else onnx_error} "
              f"reward={reward} terrain={terrain}", flush=True)
    return 0


def _meta(out, candidate, args, reward, terrain, info) -> dict:
    import hashlib
    import json as _json

    from policy.locomotion.core.checkpoint_meta import build_meta, git_hash

    from spikes.newton.train import _solver_values

    meta = build_meta(
        checkpoint=str(candidate),
        task_id=tasks.NEWTON_TASK_ID,
        physics_backend={"preset": "newton_mjwarp", "solver": "MJWarpSolverCfg",
                         "stock_values": _solver_values(),
                         "num_substeps": info.get("num_substeps")},
        iterations=int(args.iterations),
        seed=int(args.seed),
        num_envs=4096,
        wall_clock_s=2532.0,  # 42:12 of training to the cut, from the run's own log
        final_reward=float(reward if reward is not None else 0.0),
        config_hash_value=hashlib.sha256(_json.dumps(info, sort_keys=True).encode()).hexdigest()[:16],
        git_hash_value=git_hash(str(common.CODE_ROOT)),
        onnx=str(out / f"{CANDIDATE}.onnx"),
        terrain_levels=terrain,
        run_dir=str(args.run_dir),
        extra={"phase": "newton retrain, phase 5 (run cut at iteration 1312 of 1495)",
               "note": ITERS_COMPLETED_NOTE, "s_per_iteration": 1.95,
               "recipe": "stock rsl_rl PPO preset, 4096 envs, seed 42, push events + curriculum + stair scaling"},
    )
    (out / f"{CANDIDATE}.meta.json").write_text(_json.dumps(meta, indent=2) + "\n")
    print(f"[export] wrote {out / (CANDIDATE + '.meta.json')}", flush=True)
    return meta


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
