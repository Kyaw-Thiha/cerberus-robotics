"""50-iteration retrain probe at 4096 envs: seconds per iteration and peak VRAM.

Same task, recipe and runner the real retrain uses (Cerberus Go2 rough cfg, stock rsl_rl PPO preset), just
stopped after 50 iterations. The number this produces pins the "what would a Newton retrain cost" estimate
that otherwise has to be extrapolated from 128-env timings.

    python -m spikes.newton.retrain_probe --out data/reports/newton/retrain_probe --iterations 50
"""

from __future__ import annotations

import sys
import time

from spikes.newton import common, tasks


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()
    report: dict = {"stage": "retrain_probe", "args": vars(args), "versions": common.software_versions()}

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering="none", seed=args.seed,
                                episode_length_s=20.0)
    report["cfg_info"] = info
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        import torch
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        except ImportError:  # older layout
            from isaaclab_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
        from common.cerberus_compat.rsl_rl_compat import make_runner

        agent_cfg = UnitreeGo2RoughPPORunnerCfg()
        agent_cfg.max_iterations = args.iterations
        agent_cfg.experiment_name = "newton_retrain_probe"
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        timer.mark("vec_wrapped")
        report["num_envs"] = int(args.num_envs)
        report["decimation"] = cfg.decimation
        report["steps_per_iteration"] = int(agent_cfg.num_steps_per_env)
        runner = make_runner(vec, agent_cfg, log_dir=str(out / "train"), device=str(cfg.sim.device))
        timer.mark("runner_ready")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        wall = time.time() - t0
        timer.mark("learn")
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20

        report.update({
            "iterations": args.iterations,
            "wall_s": round(wall, 2),
            "s_per_iteration": round(wall / args.iterations, 3),
            "peak_vram_allocated_mib": round(peak_alloc, 1),
            "peak_vram_reserved_mib": round(peak_reserved, 1),
            "env_build_s": timer.marks[1][1] if len(timer.marks) > 1 else None,
            "timings": timer.as_dict(),
            "extrapolation": _extrapolate(wall / args.iterations, args.iterations),
        })
        common.write_json(out / "retrain_probe.json", report)
        print(f"[probe] {args.num_envs} envs, {args.iterations} iters: {report['s_per_iteration']} s/iter, "
              f"peak VRAM {peak_alloc:.0f} MiB", flush=True)
        print(f"[probe] env build {report['env_build_s']}s, learn {wall:.1f}s", flush=True)
        env.close()
    return 0


def _extrapolate(s_per_iter: float, measured_iters: int) -> dict:
    """A 1500-iteration retrain at the measured rate, with the first-iteration warm-up excluded."""
    full = 1500
    return {
        "measured_s_per_iteration": round(s_per_iter, 3),
        "retrain_1500_iterations_h": round(s_per_iter * full / 3600, 2),
        "retrain_1500_iterations_usd_at_0_89_per_h": round(s_per_iter * full / 3600 * 0.89, 2),
        "retrain_1500_iterations_usd_at_0_27_per_h": round(s_per_iter * full / 3600 * 0.27, 2),
        "note": "wall time / iterations; the first iterations include CUDA-graph capture and rollout warm-up, "
                "so this over-estimates the steady state slightly",
    }


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
