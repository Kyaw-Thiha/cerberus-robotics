# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Push-recovery eval sweep (plans/handoff_phase0.md step 4). Loads a trained
# checkpoint, runs it against a magnitude x trials_per_cell grid of one-shot
# horizontal pushes (see push_recovery_event.py), and writes a per-cell
# success-rate CSV that plot_push_recovery.py turns into the curve.
#
# One env per trial (not one env replayed many times) -- num_envs =
# len(magnitudes) * len(terrain_levels or [None]) * trials_per_cell, each
# pinned to one (magnitude, terrain_level) cell for its whole (single)
# episode. See the design discussion in this session: this matches how
# Rudin/Hwangbo's own published push-tests work (many independent trials
# under fixed conditions), and is much simpler to grade correctly than
# per-env retry bookkeeping.
#
# Terrain difficulty is a second axis, orthogonal to magnitude (--terrain_levels,
# a Rough task only -- Flat has no difficulty grid). Each env is pinned to a
# specific TerrainImporter difficulty row directly, bypassing the adaptive
# terrain_levels curriculum used during training/play: right after gym.make
# (before RslRlVecEnvWrapper, which resets on construction) we overwrite
# terrain.terrain_levels/.terrain_types and recompute terrain.env_origins --
# InteractiveScene.env_origins is a live property reading straight off the
# TerrainImporter, so this takes effect at that first reset. See
# TerrainImporter.update_env_origins in Isaac Lab's own source for the same
# indexing pattern (terrain_origins[level, type]).

"""Run the push-recovery eval sweep against a trained checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(os.environ["ISAACLAB_PATH"], "scripts", "reinforcement_learning", "rsl_rl"))

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Push-recovery eval sweep for a trained Cerberus Go2 checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the eval env from (its Play variant's randomization-off settings apply).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--magnitudes",
    type=str,
    default="0,40,80,120,160,200",
    help="Comma-separated push force magnitudes in N.",
)
parser.add_argument("--trials_per_cell", type=int, default=100, help="Independent trials per grid cell.")
parser.add_argument(
    "--terrain_levels",
    type=str,
    default=None,
    help="Comma-separated terrain difficulty row indices (0=easiest) to stratify against, orthogonal to "
    "--magnitudes. Requires a Rough task with a generator terrain (its TerrainImporter.terrain_origins grid). "
    "Omit for a Flat, magnitude-only sweep.",
)
parser.add_argument("--push_time_s", type=float, default=3.0, help="Simulation time the push fires at.")
parser.add_argument("--window_s", type=float, default=3.0, help="Post-push window to judge recovery over.")
parser.add_argument("--sustain_s", type=float, default=1.0, help="Tail of the window tracking error must stay low for.")
parser.add_argument("--roll_pitch_bound_deg", type=float, default=30.0, help="Max |roll|/|pitch| at window end.")
parser.add_argument(
    "--tracking_error_threshold", type=float, default=0.3, help="Max base_velocity/error_vel_xy (m/s) to count as recovered."
)
parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: eval/results/<timestamp>.csv).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from policy.locomotion.eval.push_recovery_event import apply_single_push
from policy.locomotion.core.script_utils import checkpoints_root
from policy.locomotion.core.terrain_pinning import pin_terrain


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    magnitudes = [float(m) for m in args_cli.magnitudes.split(",")]
    trials = args_cli.trials_per_cell
    terrain_levels_list = [int(x) for x in args_cli.terrain_levels.split(",")] if args_cli.terrain_levels else None

    # grid cells in (magnitude, terrain_level) order, each repeated `trials` times --
    # env i's cell is cells[i // trials]. terrain_level is None for a Flat (magnitude-only) sweep.
    if terrain_levels_list is None:
        cells = [(m, None) for m in magnitudes]
    else:
        cells = [(m, lvl) for m in magnitudes for lvl in terrain_levels_list]
    num_envs = len(cells) * trials

    magnitudes_tensor = torch.tensor([m for m, _ in cells], device=args_cli.device or "cuda:0").repeat_interleave(
        trials
    )

    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = args_cli.push_time_s + args_cli.window_s + 1.0

    env_cfg.events.push_recovery_probe = EventTerm(
        func=apply_single_push,
        mode="interval",
        interval_range_s=(args_cli.push_time_s, args_cli.push_time_s),
        params={
            "magnitudes": magnitudes_tensor,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "log_angles": True,
        },
    )

    log_root_path = checkpoints_root(os.path.dirname(os.path.dirname(__file__)), agent_cfg)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)

    if terrain_levels_list is not None:
        terrain = env.unwrapped.scene.terrain
        if terrain is None or terrain.terrain_origins is None:
            raise RuntimeError(
                f"--terrain_levels was given but task '{args_cli.task}' has no terrain-difficulty grid "
                "(no TerrainImporter.terrain_origins) -- pass a Rough task instead."
            )
        num_cols = terrain.terrain_origins.shape[1]
        # pin each env's difficulty row directly, bypassing the adaptive placement --
        # spread across terrain_type columns round-robin for variety within a level, same as
        # TerrainImporter's own initial placement does.
        levels_per_env = [lvl for _, lvl in cells for _ in range(trials)]
        types_per_env = [i % num_cols for i in range(num_envs)]
        pin_terrain(env, levels_per_env, types_per_env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    asset = raw_env.scene["robot"]
    dt = raw_env.step_dt
    device = raw_env.device

    push_step = round(args_cli.push_time_s / dt)
    window_end_step = round((args_cli.push_time_s + args_cli.window_s) / dt)
    sustain_start_step = round((args_cli.push_time_s + args_cli.window_s - args_cli.sustain_s) / dt)
    total_steps = round(env_cfg.episode_length_s / dt)

    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    roll_pitch_ok_at_end = torch.zeros(num_envs, dtype=torch.bool, device=device)
    tracking_ok_sustained = torch.ones(num_envs, dtype=torch.bool, device=device)
    roll_pitch_bound_rad = math.radians(args_cli.roll_pitch_bound_deg)

    # continuous values, not just pass/fail -- kept alongside the booleans so
    # per-trial results can show *how* a trial failed/passed, not just that it did
    roll_final = torch.zeros(num_envs, device=device)
    pitch_final = torch.zeros(num_envs, device=device)
    tracking_error_max_in_sustain = torch.zeros(num_envs, device=device)

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(total_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            newly_done = dones.bool() & ~finished
            fell |= newly_done
            finished |= dones.bool()

            if step >= sustain_start_step and step < window_end_step:
                command = raw_env.command_manager.get_command("base_velocity")
                actual_xy = asset.data.root_lin_vel_b[:, :2]
                error_xy = torch.norm(command[:, :2] - actual_xy, dim=1)
                tracking_error_max_in_sustain = torch.maximum(tracking_error_max_in_sustain, error_xy)
                tracking_ok_sustained &= (error_xy < args_cli.tracking_error_threshold) | finished

            if step == window_end_step:
                roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
                roll_final, pitch_final = roll, pitch
                roll_pitch_ok_at_end = (roll.abs() < roll_pitch_bound_rad) & (pitch.abs() < roll_pitch_bound_rad)

    recovered = ~fell & roll_pitch_ok_at_end & tracking_ok_sustained

    # move everything to cpu once, up front -- avoids a device round-trip per field per trial
    fell_cpu = fell.cpu()
    roll_pitch_ok_cpu = roll_pitch_ok_at_end.cpu()
    tracking_ok_cpu = tracking_ok_sustained.cpu()
    recovered_cpu = recovered.cpu()
    roll_final_cpu = roll_final.cpu()
    pitch_final_cpu = pitch_final.cpu()
    tracking_error_max_cpu = tracking_error_max_in_sustain.cpu()
    # written by apply_single_push directly onto raw_env (not via EventTermCfg params --
    # see push_recovery_event.py's docstring for why that doesn't work)
    angle_log_cpu = raw_env.push_angle_log.cpu()

    # cells were laid out as contiguous `trials`-sized blocks, in `cells` order -- cell c's
    # envs are [c * trials, (c + 1) * trials)
    summary_results = []
    trial_results = []
    for c, (mag, lvl) in enumerate(cells):
        lo, hi = c * trials, (c + 1) * trials
        block = recovered_cpu[lo:hi]
        n = int(block.numel())
        successes = int(block.sum())
        terrain_label = lvl if lvl is not None else "flat"
        summary_results.append(
            {
                "magnitude_n": mag,
                "terrain_level": terrain_label,
                "trials": n,
                "successes": successes,
                "success_rate": successes / n,
            }
        )
        print(f"[RESULT] magnitude={mag}N terrain_level={terrain_label}  {successes}/{n} recovered  ({successes / n:.1%})")

        for trial_idx, env_idx in enumerate(range(lo, hi)):
            trial_results.append(
                {
                    "magnitude_n": mag,
                    "terrain_level": terrain_label,
                    "trial_index": trial_idx,
                    "push_angle_rad": float(angle_log_cpu[env_idx]),
                    "fell": bool(fell_cpu[env_idx]),
                    "roll_final_rad": float(roll_final_cpu[env_idx]),
                    "pitch_final_rad": float(pitch_final_cpu[env_idx]),
                    "roll_pitch_ok_at_end": bool(roll_pitch_ok_cpu[env_idx]),
                    "tracking_error_max_in_sustain": float(tracking_error_max_cpu[env_idx]),
                    "tracking_ok_sustained": bool(tracking_ok_cpu[env_idx]),
                    "recovered": bool(recovered_cpu[env_idx]),
                }
            )

    output_path = args_cli.output
    if output_path is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, f"push_recovery_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    trials_output_path = output_path.rsplit(".", 1)[0] + "_trials.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["magnitude_n", "terrain_level", "trials", "successes", "success_rate"])
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"[INFO] Wrote summary results to: {output_path}")

    with open(trials_output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "magnitude_n",
                "terrain_level",
                "trial_index",
                "push_angle_rad",
                "fell",
                "roll_final_rad",
                "pitch_final_rad",
                "roll_pitch_ok_at_end",
                "tracking_error_max_in_sustain",
                "tracking_ok_sustained",
                "recovered",
            ],
        )
        writer.writeheader()
        writer.writerows(trial_results)
    print(f"[INFO] Wrote per-trial results to: {trials_output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
