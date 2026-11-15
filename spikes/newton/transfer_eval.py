"""Stage 4: run the PhysX-trained policy on Newton and measure forward tracking + falls per terrain level.

The policy is evaluated exactly as it was trained (no tuning, no normaliser): identical observation
term order, action term and command term, only the physics backend differs. ``--joint_ordering`` selects
the documented transfer flag: ``none`` (backend order), ``physx``/``mjwarp`` (emulated convention), or
``contract`` (the explicit 12-name list the checkpoint was trained on).

    python -m spikes.newton.transfer_eval --policy <ckpt.pt|ckpt.onnx> --out data/reports/newton/transfer \
        --levels 0,1,2,3,4,5 --envs_per_level 64 --steps 1000 --vx 0.8
"""

from __future__ import annotations

import csv
import sys

import torch

from spikes.newton import common, tasks
from spikes.newton.policy_loader import load_policy


def _reset(env):
    result = env.reset()
    return (result[0] if isinstance(result, tuple) else result)


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--policy", required=True, help="checkpoint (.pt) or ONNX export")
    parser.add_argument("--joint_ordering", default="none", choices=["none", "physx", "mjwarp", "contract"])
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--envs_per_level", type=int, default=64)
    parser.add_argument("--vx", type=float, default=0.8)
    args = parser.parse_args(sys.argv[1:])

    out = common.out_dir(args.out)
    levels = [int(x) for x in args.levels.split(",")]
    num_envs = len(levels) * args.envs_per_level
    timer = common.Timer()
    report = {"stage": "transfer", "args": vars(args), "versions": common.software_versions()}

    cfg, info = tasks.build_cfg(num_envs=num_envs, joint_ordering=args.joint_ordering,
                                forward_command=args.vx, seed=args.seed,
                                episode_length_s=args.steps * 0.02 + 2.0)
    report["cfg_info"] = info
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym
    from policy.locomotion.core.terrain_pinning import pin_terrain, sub_terrain_type_for_column

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        terrain = env.unwrapped.scene.terrain
        num_cols = int(terrain.terrain_origins.shape[1])
        levels_per_env = [lvl for lvl in levels for _ in range(args.envs_per_level)]
        types_per_env = [i % num_cols for i in range(num_envs)]
        pin_terrain(env, levels_per_env, types_per_env)
        type_names = [sub_terrain_type_for_column(cfg.scene.terrain.terrain_generator, c, num_cols)
                      for c in types_per_env]
        report["terrain"] = {"num_rows": int(terrain.terrain_origins.shape[0]), "num_cols": num_cols,
                             "levels": levels, "envs_per_level": args.envs_per_level}
        report["sim"] = common.sim_info(env)
        report["joints"] = common.joint_layout(env)

        policy, policy_info = load_policy(args.policy, "cpu")
        report["policy"] = policy_info
        print(f"[transfer] policy: {policy_info['source']}", flush=True)

        obs = _reset(env)
        obs_vec = common.obs_vector(obs)
        if obs_vec.shape[-1] != 235:
            report["error"] = f"policy expects the 235-D contract, env gave {obs_vec.shape[-1]}"
            common.write_json(out / "transfer.json", report)
            return 1

        robot = env.unwrapped.scene["robot"]
        n = args.num_envs = int(num_envs)
        alive = torch.ones(n, dtype=torch.bool)
        first_fall = torch.full((n,), -1, dtype=torch.long)
        fall_steps = torch.zeros(n)
        err_vx = torch.zeros(n)
        err_vy = torch.zeros(n)
        err_wz = torch.zeros(n)
        alive_steps = torch.zeros(n)
        nan_seen = False

        dt = float(report["sim"].get("step_dt") or 0.02)
        t0 = __import__("time").time()
        with torch.inference_mode():
            for step in range(args.steps):
                action = policy(obs_vec)
                result = env.step(action)
                obs = result[0]
                obs_vec = common.obs_vector(obs)
                base_contact = common.term_bool(env, "base_contact")
                just_fell = torch.zeros(n, dtype=torch.bool)
                if base_contact is not None:
                    just_fell = base_contact.bool() & alive
                    first_fall[just_fell] = step
                    fall_steps += just_fell.float()
                    alive &= ~base_contact.bool()
                vel_b = robot.data.root_lin_vel_b
                ang_b = robot.data.root_ang_vel_b
                track = torch.stack([(vel_b[:, 0] - args.vx).abs(), vel_b[:, 1].abs(), ang_b[:, 2].abs()], dim=1)
                err_vx += track[:, 0] * alive.float()
                err_vy += track[:, 1] * alive.float()
                err_wz += track[:, 2] * alive.float()
                alive_steps += alive.float()
                if bool(torch.isnan(obs_vec).any()):
                    nan_seen = True
                if step % 100 == 0:
                    print(f"[transfer] step {step}/{args.steps} alive={int(alive.sum())}/{n}", flush=True)
        wall = __import__("time").time() - t0
        timer.mark("rollout")

        denom = alive_steps.clamp(min=1.0)
        rows = []
        for level in levels:
            idx = [i for i, lvl in enumerate(levels_per_env) if lvl == level]
            fell = alive[idx].logical_not() | (fall_steps[idx] > 0)
            rows.append({
                "terrain_level": level,
                "envs": len(idx),
                "falls": int(fell.sum()),
                "fall_rate": round(float(fell.float().mean()), 4),
                "mean_abs_vx_error_mps": round(float((err_vx[idx] / denom[idx]).mean()), 4),
                "mean_abs_vy_mps": round(float((err_vy[idx] / denom[idx]).mean()), 4),
                "mean_abs_wz_radps": round(float((err_wz[idx] / denom[idx]).mean()), 4),
                "mean_alive_steps": round(float(alive_steps[idx].mean()), 1),
                "median_steps_to_first_fall": (None if int((first_fall[idx] >= 0).sum()) == 0
                                               else int(first_fall[idx][first_fall[idx] >= 0].median())),
            })
        report["rollout"] = {
            "steps": args.steps, "num_envs": n, "seconds": round(wall, 2),
            "steps_per_s": round(args.steps / max(wall, 1e-9), 1),
            "nan_seen": nan_seen,
            "total_falls": int((fall_steps > 0).sum()),
            "fall_rate": round(float((fall_steps > 0).float().mean()), 4),
            "mean_abs_vx_error_mps": round(float((err_vx / denom).mean()), 4),
            "mean_abs_vy_mps": round(float((err_vy / denom).mean()), 4),
            "mean_abs_wz_radps": round(float((err_wz / denom).mean()), 4),
            "median_steps_to_first_fall": (None if int((first_fall >= 0).sum()) == 0
                                           else int(first_fall[first_fall >= 0].median())),
            "envs_still_alive": int(alive.sum()),
        }
        report["per_level"] = rows
        report["sub_terrain_by_column"] = {str(c): name for c, name in zip(range(num_cols), [
            sub_terrain_type_for_column(cfg.scene.terrain.terrain_generator, c, num_cols) for c in range(num_cols)])}

        with (out / "per_level.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        common.write_json(out / "transfer.json", report)
        common.write_json(out / "timings.json", timer.as_dict())
        for row in rows:
            print("[transfer] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        env.close()
    print("[transfer] done", flush=True)
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
