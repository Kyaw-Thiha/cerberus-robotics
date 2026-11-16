"""Rest-pose test: hold the default joint targets for N steps and log where each joint settles.

No policy is involved. The action term is `default_joint_pos + scale * raw_action`, so a zero action commands
the asset's default joint angles; the settled joint positions are the steady state of the actuator path under
gravity. A large, consistent per-joint offset from the commanded defaults is the signature of a drive that is
stiffer or biased relative to the intended PD (isaac-sim/IsaacLab#5806: the Newton backend re-enabling joint
target stiffness/damping for explicit actuators).

    python -m spikes.newton.rest_pose --out data/reports/newton/rest_pose --use_newton_actuators true
"""

from __future__ import annotations

import sys

import torch

from spikes.newton import common, contract, tasks


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--use_newton_actuators", default="true", choices=["true", "false"])
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()

    cfg, info = tasks.build_cfg(num_envs=args.num_envs, joint_ordering="none", seed=args.seed,
                                episode_length_s=args.steps * 0.02 + 5.0,
                                use_newton_actuators=args.use_newton_actuators == "true")
    report = {"stage": "rest_pose", "args": vars(args), "cfg_info": info,
              "versions": common.software_versions()}
    tasks.register()

    from isaaclab.app import launch_simulation
    import gymnasium as gym

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        robot = env.unwrapped.scene["robot"]
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs_vec = common.obs_vector(obs)

        names = list(robot.joint_names)
        default_asset = robot.data.default_joint_pos[0].tolist()
        contract_defaults = _contract_defaults()
        action = torch.zeros((args.num_envs, int(common.action_layout(env).get("total_action_dim") or 12)),
                             device=obs_vec.device)

        heights = []
        t0 = __import__("time").time()
        for _ in range(args.steps):
            env.step(action)
            heights.append(float(robot.data.root_pos_w[:, 2].mean()))
        wall = __import__("time").time() - t0
        timer.mark("hold")

        settled = robot.data.joint_pos.mean(dim=0)
        spread = robot.data.joint_pos.std(dim=0)
        velocity = robot.data.joint_vel.abs().mean(dim=0)
        target = torch.tensor(default_asset, device=settled.device, dtype=settled.dtype)

        rows = []
        for i, name in enumerate(names):
            commanded = contract_defaults.get(name)
            rows.append({
                "joint": name,
                "commanded_default_rad": round(commanded, 4) if commanded is not None else None,
                "asset_default_rad": round(float(default_asset[i]), 4),
                "settled_rad": round(float(settled[i]), 4),
                "offset_from_asset_default_rad": round(float(settled[i] - target[i]), 4),
                "offset_from_asset_default_deg": round(float((settled[i] - target[i]) * 180 / 3.141592653589793), 2),
                "offset_from_contract_rad": (round(float(settled[i] - commanded), 4) if commanded is not None else None),
                "offset_from_contract_deg": (round(float((settled[i] - commanded) * 180 / 3.141592653589793), 2)
                                             if commanded is not None else None),
                "env_std_rad": round(float(spread[i]), 5),
                "abs_joint_vel_radps": round(float(velocity[i]), 5),
            })

        offsets = [abs(row["offset_from_contract_rad"]) for row in rows if row["offset_from_contract_rad"] is not None]
        report.update({
            "sim": common.sim_info(env),
            "joints": rows,
            "summary": {
                "max_abs_offset_rad": round(max(offsets), 4),
                "max_abs_offset_deg": round(max(offsets) * 180 / 3.141592653589793, 2),
                "mean_abs_offset_rad": round(sum(offsets) / len(offsets), 4),
                "mean_abs_offset_deg": round(sum(offsets) / len(offsets) * 180 / 3.141592653589793, 2),
                "joints_over_5deg": [row["joint"] for row in rows
                                     if row["offset_from_contract_deg"] is not None
                                     and abs(row["offset_from_contract_deg"]) > 5.0],
                "base_height_start_m": round(heights[0], 4),
                "base_height_end_m": round(heights[-1], 4),
                "base_height_min_m": round(min(heights), 4),
                "steps": args.steps,
                "seconds": round(wall, 2),
            },
            "default_vs_contract_max_abs_rad": round(max(
                abs(default_asset[i] - contract_defaults[n]) for i, n in enumerate(names)
                if n in contract_defaults), 6),
        })
        common.write_json(out / "rest_pose.json", report)
        common.write_json(out / "timings.json", timer.as_dict())
        print("[rest] joint                commanded  settled  offset_deg  std", flush=True)
        for row in rows:
            print(f"[rest] {row['joint']:<20} {row['commanded_default_rad']:>9} {row['settled_rad']:>8} "
                  f"{row['offset_from_contract_deg']:>10} {row['env_std_rad']:>8}", flush=True)
        print(f"[rest] summary {report['summary']}", flush=True)
        env.close()
    return 0


def _contract_defaults() -> dict:
    data = contract._load()  # noqa: SLF001 - the contract reader is this package's own
    return {name: float(value) for name, value in data["joints"]["default_joint_pos_rad"].items()}


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
