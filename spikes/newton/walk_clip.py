"""A 20 s walking clip of a trained policy on a pinned rough-terrain level, with stills.

The Newton GL visualizer has no camera tracking at this tag (``NewtonVisualizerCfg`` carries only a static
``eye``/``lookat``), so the camera is aimed at the pinned terrain patch: a first, unrendered environment is
created only to learn the patch origin (``scene.env_origins[0]`` after ``pin_terrain``), and the clip is then
recorded from a second environment whose camera sits 2.5 m from that origin.

    python -m spikes.newton.walk_clip --policy <ckpt.pt> --level 3 --out <dir> --steps 1000
"""

from __future__ import annotations

import sys

import torch

from spikes.newton import common, tasks
from spikes.newton.policy_loader import load_policy

NO_VIZ = {"visualizer": []}


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--type", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=1000, help="1000 steps at 50 Hz = 20 s")
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()
    report: dict = {"stage": "walk_clip", "args": vars(args), "versions": common.software_versions()}

    from isaaclab.app import launch_simulation
    import gymnasium as gym
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg
    from policy.locomotion.core.terrain_pinning import pin_terrain

    tasks.register()
    policy, policy_info = load_policy(args.policy, "cpu")
    report["policy"] = policy_info

    cfg, info = tasks.build_cfg(num_envs=1, joint_ordering="none", forward_command=args.vx, seed=args.seed,
                                episode_length_s=args.steps * 0.02 + 5.0)
    report["cfg_info"] = info

    # pass 1: where is the pinned patch? (no visualizer)
    with launch_simulation(cfg, NO_VIZ):
        probe = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        pin_terrain(probe, [args.level], [args.type])
        origin = probe.unwrapped.scene.env_origins[0].tolist()
        timer.mark("probe_env")
        report["patch_origin"] = [round(v, 2) for v in origin]
        probe.close()

    ox, oy, _oz = origin
    eye = (ox + 2.2, oy - 2.6, 1.4)
    lookat = (ox, oy, 0.35)
    cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=eye, lookat=lookat)
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl", output_dir=str(out / "videos"),
                                            video_length=args.steps, video_interval=0)]
    report["camera"] = {"eye": [round(v, 2) for v in eye], "lookat": [round(v, 2) for v in lookat]}
    report["terrain"] = {"level": args.level, "type": args.type}

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        pin_terrain(env, [args.level], [args.type])
        timer.mark("env_built")
        robot = env.unwrapped.scene["robot"]
        result = env.reset()
        obs = common.obs_vector(result[0] if isinstance(result, tuple) else result)
        contacts = 0.0
        heights, gz, vx = [], [], []
        with torch.inference_mode():
            for step in range(args.steps):
                result = env.step(policy(obs))
                obs = common.obs_vector(result[0])
                flag = common.term_bool(env, "base_contact")
                contacts += float(flag.float().sum()) if flag is not None else 0.0
                heights.append(float(robot.data.root_pos_w[:, 2]))
                gz.append(float(robot.data.projected_gravity_b[:, 2]))
                vx.append(float(robot.data.root_lin_vel_b[:, 0]))
                if step and step % 250 == 0:
                    print(f"[walk] step {step}: base_z={heights[-1]:.3f} gz={gz[-1]:.3f} vx={vx[-1]:.2f} "
                          f"contact={contacts}", flush=True)
        timer.mark("rollout")
        report.update({
            "steps": args.steps,
            "seconds": round(args.steps * 0.02, 1),
            "base_contact_steps": contacts,
            "base_z_start": round(heights[0], 3), "base_z_end": round(heights[-1], 3),
            "gz_min": round(min(gz), 3), "gz_end": round(gz[-1], 3),
            "mean_forward_speed": round(sum(vx) / len(vx), 3),
            "commanded_forward_speed": args.vx,
            "videos": [p.name for p in sorted((out / "videos").glob("*.mp4"))],
            "timings": timer.as_dict(),
        })
        common.write_json(out / "walk_clip.json", report)
        print(f"[walk] level={args.level} contact={contacts} mean_vx={report['mean_forward_speed']} "
              f"gz_min={report['gz_min']} clips={report['videos']}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
