"""What a failing transfer rollout looks like, how "fall" is actually measured, and the policy standing still.

Four things in one process, in this order:

1. a 10 s Newton GL clip of the forward-command rollout with the PhysX-trained policy (5 envs visible),
   plus base height and projected-gravity z logged for those 5 envs at every step;
2. the fall definition: `transfer_eval` counts a fall when the ``base_contact`` termination term fires, i.e.
   the torso touches something -- not a height or an orientation threshold. The same per-step log says
   whether that flag ever fired with the robot still upright;
3. a zero-command stand with the policy in the loop for 300 steps: does anything fall?
4. the 235-D observation split into its eight terms, with each term's dim, scale, values and the largest
   difference against the reference (the contract's values, the recorded PhysX standing pose, or the
   analytic value at rest).

    python -m spikes.newton.rollout_clip --policy <ckpt.pt> --out data/reports/newton/rollout_clip
"""

from __future__ import annotations

import json
import sys

import torch

from spikes.newton import common, contract, tasks
from spikes.newton.policy_loader import load_policy

TERM_DIMS = [3, 3, 3, 3, 12, 12, 12, 187]
TERM_NAMES = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
              "joint_pos", "joint_vel", "actions", "height_scan"]
#: PhysX reference for a standing Go2, measured on the GA stack (docs/h2_report.md §5, contract file)
PHYSX_STANDING_GRAVITY = [-0.090, -0.129, -0.988]


def main() -> int:
    parser = common.build_parser(__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--vx", type=float, default=0.8)
    parser.add_argument("--clip_steps", type=int, default=500, help="500 steps at 50 Hz = 10 s")
    parser.add_argument("--stand_steps", type=int, default=300)
    parser.add_argument("--log_envs", type=int, default=5)
    parser.add_argument("--clip_only", action="store_true", help="record the clip and stop (no stand/term diff)")
    parser.add_argument("--plane", action="store_true", help="flat ground for the clip (readable camera framing)")
    parser.add_argument("--camera", default="", help="eye_x,eye_y,eye_z,lookat_x,lookat_y,lookat_z")
    args = parser.parse_args(sys.argv[1:])
    out = common.out_dir(args.out)
    timer = common.Timer()
    report: dict = {"stage": "rollout_clip", "args": vars(args), "versions": common.software_versions()}

    tasks.register()
    from isaaclab.app import launch_simulation
    import gymnasium as gym
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    policy, policy_info = load_policy(args.policy, "cpu")
    report["policy"] = policy_info

    # ---- 1+2: the clip, and what "fall" means ------------------------------------------------------
    cfg, info = tasks.build_cfg(num_envs=max(args.log_envs, 5), joint_ordering="none",
                                forward_command=args.vx, seed=args.seed,
                                episode_length_s=args.clip_steps * 0.02 + 2.0)
    if args.plane:
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainImporterCfg

        cfg.curriculum.terrain_levels = None
        cfg.curriculum.terrain_level_max = None
        cfg.curriculum.terrain_level_frac_at_max = None
        cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="plane",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0),
            debug_vis=False)
        report.setdefault("clip", {})["terrain"] = "plane"
    if args.camera:
        from isaaclab.visualizers import VisualizerCfg

        eye = tuple(float(v) for v in args.camera.split(",")[:3])
        lookat = tuple(float(v) for v in args.camera.split(",")[3:6])
        cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=eye, lookat=lookat)
        report.setdefault("clip", {})["camera"] = {"eye": eye, "lookat": lookat}
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:newton_gl",
                                            output_dir=str(out / "videos"),
                                            video_length=args.clip_steps, video_interval=0)]
    report["clip"] = {"cfg_info": info, "recorder": {"source": "visualizer:newton_gl",
                                                    "video_length": args.clip_steps}}

    with launch_simulation(cfg, args):
        timer.mark("simulation_ready")
        env = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=cfg)
        timer.mark("env_built")
        robot = env.unwrapped.scene["robot"]
        obs = _reset(env)
        obs_vec = common.obs_vector(obs)
        track = {key: [] for key in ("base_z", "gz", "flag")}
        first_fall = {}
        n_log = min(args.log_envs, args.num_envs)
        with torch.inference_mode():
            for step in range(args.clip_steps):
                action = policy(obs_vec)
                result = env.step(action)
                obs_vec = common.obs_vector(result[0])
                flag = common.term_bool(env, "base_contact")
                flag = flag.bool() if flag is not None else torch.zeros(args.num_envs, dtype=torch.bool)
                base_z = robot.data.root_pos_w[:, 2]
                gz = robot.data.projected_gravity_b[:, 2]
                track["base_z"].append(base_z[:n_log].tolist())
                track["gz"].append(gz[:n_log].tolist())
                track["flag"].append(flag[:n_log].tolist())
                for i in range(n_log):
                    if flag[i] and i not in first_fall:
                        first_fall[i] = {"step": step, "base_z": round(float(base_z[i]), 4),
                                         "gz": round(float(gz[i]), 4),
                                         "upright": bool(float(gz[i]) < -0.9)}
        timer.mark("clip_rollout")
        report["fall_definition"] = _fall_definition(track, first_fall, args.clip_steps)
        report["clip"]["videos"] = [p.name for p in sorted((out / "videos").glob("*.mp4"))]
        env.close()

            # ---- 3: zero-command stand with the policy in the loop ------------------------------------
        stand_cfg, stand_info = tasks.build_cfg(num_envs=8, joint_ordering="none", forward_command=0.0,
                                                seed=args.seed, episode_length_s=args.stand_steps * 0.02 + 2.0)
        report["stand"] = {"cfg_info": stand_info}
    if args.clip_only:
        report["timings"] = timer.as_dict()
        common.write_json(out / "rollout_clip.json", report)
        print("[clip] clip_only: wrote the clip and the per-step log, skipping stand/term diff", flush=True)
        return 0
    with launch_simulation(stand_cfg, args):
        env2 = gym.make(tasks.NEWTON_TASK_PLAY_ID, cfg=stand_cfg)
        robot2 = env2.unwrapped.scene["robot"]
        obs2 = _reset(env2)
        obs2_vec = common.obs_vector(obs2)
        report["observation"] = _term_diff(env2, obs2_vec, robot2, stage="reset")
        flags = 0.0
        heights, gz_series = [], []
        with torch.inference_mode():
            for _ in range(args.stand_steps):
                result = env2.step(policy(obs2_vec))
                obs2_vec = common.obs_vector(result[0])
                flag = common.term_bool(env2, "base_contact")
                flags += float(flag.float().sum()) if flag is not None else 0.0
                heights.append(float(robot2.data.root_pos_w[:, 2].mean()))
                gz_series.append(float(robot2.data.projected_gravity_b[:, 2].mean()))
        timer.mark("stand")
        report["stand"].update({
            "steps": args.stand_steps,
            "base_contact_steps_total": flags,
            "envs": 8,
            "base_z_start": round(heights[0], 4),
            "base_z_end": round(heights[-1], 4),
            "base_z_min": round(min(heights), 4),
            "base_z_max": round(max(heights), 4),
            "gz_start": round(gz_series[0], 4),
            "gz_end": round(gz_series[-1], 4),
            "observation_settled": _term_diff(env2, obs2_vec, robot2, stage="settled", heights=heights),
        })
        env2.close()

    report["timings"] = timer.as_dict()
    common.write_json(out / "rollout_clip.json", report)
    print("[clip] fall definition:", json.dumps(report["fall_definition"])[:600], flush=True)
    print("[clip] stand:", json.dumps(report["stand"])[:400], flush=True)
    return 0


def _reset(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _fall_definition(track, first_fall: dict, steps: int) -> dict:
    flag_steps = sum(1 for row in track["flag"] for value in row if value)
    upright_falls = [i for i, row in first_fall.items() if row["upright"]]
    min_z = min(min(row) for row in track["base_z"])
    return {
        "definition": "a fall is the base_contact termination term firing: the torso (body name 'base') "
                      "touching terrain, not a height or orientation threshold",
        "steps_logged": steps,
        "logged_envs": len(track["base_z"][0]) if track["base_z"] else 0,
        "flag_fires": flag_steps,
        "first_fall": first_fall,
        "first_fall_steps": {str(i): row["step"] for i, row in first_fall.items()},
        "first_fall_base_z": {str(i): row["base_z"] for i, row in first_fall.items()},
        "first_fall_gz": {str(i): row["gz"] for i, row in first_fall.items()},
        "envs_flagging_while_upright": upright_falls,
        "min_base_z_m": round(min_z, 4),
        "verdict": ("the flag only fired on non-upright robots"
                    if not upright_falls else
                    f"{len(upright_falls)} env(s) flagged while still upright (gz < -0.9): the flag can fire "
                    "without the robot having fallen"),
    }


def _term_diff(env, obs_vec, robot, *, stage: str, heights: list | None = None) -> dict:
    """Split the 235-D observation into its terms and compare each with its reference value."""
    terms = {}
    start = 0
    scales = _term_scales(env)
    for name, dim in zip(TERM_NAMES, TERM_DIMS, strict=True):
        block = obs_vec[:, start:start + dim]
        start += dim
        entry = {"dim": dim, "scale": scales.get(name),
                 "mean": round(float(block.mean()), 5), "min": round(float(block.min()), 5),
                 "max": round(float(block.max()), 5)}
        entry.update(_reference_diff(name, block, robot, heights))
        terms[name] = entry
    return {"stage": stage, "total_dim": int(obs_vec.shape[-1]), "terms": terms}


def _term_scales(env) -> dict:
    out = {}
    group = getattr(getattr(env.unwrapped.cfg, "observations", None), "policy", None)
    for name in TERM_NAMES:
        term = getattr(group, name, None)
        if term is not None:
            out[name] = getattr(term, "scale", None)
    return out


def _reference_diff(name: str, block, robot, heights: list | None) -> dict:
    import math

    if name == "projected_gravity":
        ref = PHYSX_STANDING_GRAVITY if heights is not None else [0.0, 0.0, -1.0]
        values = block.mean(dim=0)
        diffs = [round(float(values[i]) - ref[i], 4) for i in range(3)]
        return {"reference": ref, "reference_source": "docs/h2_report.md §5 standing Go2 on PhysX"
                if heights is not None else "level base at spawn", "values": [round(float(v), 4) for v in values],
                "per_component_diff": diffs, "max_abs_diff": max(abs(d) for d in diffs)}
    if name == "joint_pos":
        ref = 0.0  # the term is relative to the default pose
        return {"reference": ref, "reference_source": "joint_pos is relative to default_joint_pos",
                "max_abs_diff": round(float(block.abs().max()), 5),
                "default_vs_contract": _default_vs_contract(robot)}
    if name in ("base_lin_vel", "base_ang_vel", "joint_vel", "actions"):
        return {"reference": 0.0, "reference_source": "at rest / no action issued yet" if name != "actions"
                else "the first observation is taken before any action",
                "max_abs_diff": round(float(block.abs().max()), 5)}
    if name == "velocity_commands":
        ref = 0.0 if float(block.abs().max()) < 1e-6 else None
        return {"reference": ref, "reference_source": "zero-command stand",
                "values": [round(float(v), 4) for v in block.mean(dim=0)],
                "max_abs_diff": None if ref is None else round(float(block.abs().max()), 5)}
    if name == "height_scan":
        base_z = float(robot.data.root_pos_w[:, 2].mean())
        expected = base_z - 0.5
        diff = float((block - expected).abs().max())
        return {"reference": round(expected, 4), "reference_source": "base_z - 0 (plane) - 0.5",
                "max_abs_diff": round(diff, 5),
                "cells_over_1cm": int((((block - expected).abs() > 1e-2).sum(dim=1) > 0).sum())}
    return {"note": f"no reference defined for {name}", "max_abs_diff": math.nan}


def _default_vs_contract(robot) -> dict:
    names = list(robot.joint_names)
    contract_map = contract._load()["joints"]["default_joint_pos_rad"]  # noqa: SLF001
    asset = robot.data.default_joint_pos[0].tolist()
    diffs = {n: round(float(asset[i]) - float(contract_map[n]), 5) for i, n in enumerate(names) if n in contract_map}
    return {"max_abs_diff": max(abs(v) for v in diffs.values()), "per_joint": diffs}


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
