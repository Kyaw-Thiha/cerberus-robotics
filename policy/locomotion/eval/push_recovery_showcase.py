# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Small, separate video companion to push_recovery_eval.py's statistical
# sweep -- NOT part of that sweep, deliberately: the statistical sweep needs
# many trials per cell (default 100) to get a reliable success rate, and
# forcing rendering on for that many envs would slow the numerically
# important run for no benefit. This creates just enough envs to cover a
# handful of representative (magnitude, terrain_level[, sub_terrain_type])
# cells -- one clip per condition, one Isaac Sim launch (core/video_capture.py's
# camera-switching trick, same as play_terrain_showcase.py), so it stays fast
# even with combined magnitude x terrain-level x type coverage.
#
# Run once, against the SAME final checkpoint push_recovery_eval.py's
# statistical sweep evaluates -- not per training checkpoint like
# play_terrain_showcase.py's periodic reviews (see the design discussion this
# implements: push-recovery capability either emerged or didn't by the final
# checkpoint, unlike walking quality which is worth watching evolve
# gradually).
#
# Does NOT use an EventTerm for the push (unlike push_recovery_eval.py) --
# see core/video_capture.py's module docstring for why: an interval-mode
# EventTerm fires based on each env's own elapsed episode time, which drifts
# out of sync with when this loop actually starts recording that env. Fires
# the push manually via on_window_step instead, at a precise local step
# within each condition's own recording window.

"""Push-recovery video showcase: a handful of representative clips."""

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
from policy.cli_common import check_gpu_driver_for_rendering  # isort: skip

SUB_TERRAIN_TYPES = [
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "boxes",
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
]

parser = argparse.ArgumentParser(description="Push-recovery video showcase for a trained checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the showcase env from (must have a terrain-difficulty grid -- Rough only).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--magnitudes", type=str, default="0,80,160", help="Comma-separated push magnitudes (N) for the base grid."
)
parser.add_argument(
    "--terrain_levels", type=str, default="0,4,8", help="Comma-separated terrain difficulty rows for the base grid."
)
parser.add_argument(
    "--fixed-type",
    type=str,
    default="boxes",
    choices=SUB_TERRAIN_TYPES,
    help="Sub-terrain type used for base-grid cells not in --full-coverage-cells.",
)
parser.add_argument(
    "--full-coverage-cells",
    type=str,
    default="160:8,160:4,80:4,80:8",
    help="Comma-separated magnitude:level pairs (from --magnitudes/--terrain_levels) to show all 6 "
    "sub-terrain types at, instead of just --fixed-type.",
)
parser.add_argument("--push_time_s", type=float, default=3.0, help="Simulation time into each clip the push fires at.")
parser.add_argument("--window_s", type=float, default=3.0, help="Post-push time each clip covers.")
parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Where to write clips (default: <run_dir>/eval/push_recovery_showcase/ next to the loaded checkpoint).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

args_cli.enable_cameras = True  # video is the whole point of this script, not an opt-in flag
check_gpu_driver_for_rendering()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from policy.locomotion.eval.push_recovery_event import apply_single_push
from policy.locomotion.core.push_disturbance_cfg import IMPULSE_DURATION_S
from policy.locomotion.core.push_impulse_event import clear_expired_push_impulses
from policy.locomotion.core.script_utils import checkpoints_root
from policy.locomotion.core.terrain_pinning import pin_terrain, sub_terrain_column_for_type
from policy.locomotion.core.video_capture import record_condition_clips


def _build_conditions(
    magnitudes: list[float], levels: list[int], full_coverage_cells: set[tuple[float, int]], fixed_type: str
) -> list[tuple[float, int, str]]:
    """One (magnitude, level, type_name) tuple per clip -- all 6 types at
    each full-coverage cell, just `fixed_type` everywhere else."""
    conditions = []
    for mag in magnitudes:
        for level in levels:
            types = SUB_TERRAIN_TYPES if (mag, level) in full_coverage_cells else [fixed_type]
            conditions.extend((mag, level, type_name) for type_name in types)
    return conditions


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    magnitudes = [float(m) for m in args_cli.magnitudes.split(",")]
    levels = [int(x) for x in args_cli.terrain_levels.split(",")]
    full_coverage_cells = set()
    if args_cli.full_coverage_cells:
        for pair in args_cli.full_coverage_cells.split(","):
            mag_str, lvl_str = pair.split(":")
            full_coverage_cells.add((float(mag_str), int(lvl_str)))
    conditions = _build_conditions(magnitudes, levels, full_coverage_cells, args_cli.fixed_type)

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError(
            f"Task '{args_cli.task}' has no terrain generator (no difficulty grid) -- pass a Rough task instead."
        )
    num_cols = terrain_generator.num_cols
    type_columns = {
        type_name: sub_terrain_column_for_type(terrain_generator, type_name, num_cols) for type_name in SUB_TERRAIN_TYPES
    }

    num_envs = len(conditions)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = args_cli.push_time_s + args_cli.window_s + 1.0

    # apply_single_push (fired manually below via on_window_step, not an EventTerm
    # trigger) sets its force via the same permanent_wrench_composer that only resets
    # at episode reset -- so without this clear pass, the showcase's push would persist
    # for the rest of each short clip instead of the true short impulse the trained
    # policy actually experienced. See push_impulse_event.py / REFERENCES.md.
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.events.push_showcase_clear = EventTerm(
        func=clear_expired_push_impulses,
        mode="interval",
        interval_range_s=(step_dt, step_dt),
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )

    log_root_path = checkpoints_root(os.path.dirname(os.path.dirname(__file__)), agent_cfg)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    levels_per_env = [level for _, level, _ in conditions]
    types_per_env = [type_columns[type_name] for _, _, type_name in conditions]
    pin_terrain(env, levels_per_env, types_per_env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    dt = raw_env.step_dt
    push_local_step = round(args_cli.push_time_s / dt)
    steps_per_clip = round((args_cli.push_time_s + args_cli.window_s) / dt)

    magnitudes_tensor = torch.tensor([mag for mag, _, _ in conditions], device=raw_env.device)
    asset_cfg = SceneEntityCfg("robot", body_names="base")
    asset_cfg.resolve(raw_env.scene)

    def fire_push_at_window_start(env_idx: int, local_step: int) -> None:
        if local_step == push_local_step:
            apply_single_push(
                raw_env,
                torch.tensor([env_idx], device=raw_env.device),
                magnitudes_tensor,
                IMPULSE_DURATION_S,
                asset_cfg,
            )

    output_dir = (
        Path(args_cli.output_dir) if args_cli.output_dir else Path(resume_path).parent / "eval" / "push_recovery_showcase"
    )
    condition_names = [f"mag{int(mag)}_level{level}_{type_name}" for mag, level, type_name in conditions]
    record_condition_clips(
        env, policy, condition_names, steps_per_clip, output_dir, fps=args_cli.fps, on_window_step=fire_push_at_window_start
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
