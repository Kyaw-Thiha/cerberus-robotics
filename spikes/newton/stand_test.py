"""Does the closed loop survive a contact-solver change? Zero-command stand, policy in the loop.

Used as the 5-minute pre-check before the retrain: the same PhysX-trained policy, the same task, only
``num_substeps`` on the stock Newton preset differs (2 = stock, 4 = the one lever the preset names).

    python -m spikes.newton.stand_test --policy <ckpt.pt> --out <dir> --num_substeps 2
"""

from __future__ import annotations

import sys

import torch

from spikes.newton import common, tasks
from spikes.newton.policy_loader import load_policy


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--num_substeps", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering="none", forward_command=0.0,
                                seed=args.seed, episode_length_s=args.steps * 0.02 + 2.0,
                                num_substeps=args.num_substeps)
    report = {"stage": "stand_test", "args": vars(args), "cfg_info": info,
              "versions": common.software_versions()}
    tasks.register()
    policy, policy_info = load_policy(args.policy, "cpu")
    report["policy"] = policy_info

    from isaaclab.app import launch_simulation
    import gymnasium as gym

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        robot = env.unwrapped.scene["robot"]
        result = env.reset()
        obs = common.obs_vector(result[0] if isinstance(result, tuple) else result)
        contacts = 0.0
        heights, gz = [], []
        with torch.inference_mode():
            for _ in range(args.steps):
                result = env.step(policy(obs))
                obs = common.obs_vector(result[0])
                flag = common.term_bool(env, "base_contact")
                contacts += float(flag.float().sum()) if flag is not None else 0.0
                heights.append(float(robot.data.root_pos_w[:, 2].mean()))
                gz.append(float(robot.data.projected_gravity_b[:, 2].mean()))
        timer.mark("stand")
        report.update({
            "num_substeps": args.num_substeps,
            "steps": args.steps,
            "envs": args.num_envs,
            "base_contact_steps_total": contacts,
            "stayed_up": contacts == 0.0 and gz[-1] < -0.9,
            "base_z_start": round(heights[0], 4),
            "base_z_end": round(heights[-1], 4),
            "base_z_min": round(min(heights), 4),
            "gz_start": round(gz[0], 4),
            "gz_end": round(gz[-1], 4),
            "gz_min": round(min(gz), 4),
            "timings": timer.as_dict(),
        })
        common.write_json(out / "stand_test.json", report)
        print(f"[stand] substeps={args.num_substeps} envs={args.num_envs} steps={args.steps} "
              f"contact={contacts} base {report['base_z_start']}->{report['base_z_end']} "
              f"gz {report['gz_start']}->{report['gz_end']} stayed_up={report['stayed_up']}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
