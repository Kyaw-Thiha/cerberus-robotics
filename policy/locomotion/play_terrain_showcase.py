# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Structured Rough-terrain video showcase for a trained checkpoint -- one
# Isaac Sim launch, one clip per representative (difficulty level, sub-terrain
# type) condition, camera switched between envs mid-rollout (see
# core/video_capture.py) instead of paying a fresh app-launch per clip. Built
# for the periodic checkpoint-review cadence discussed for the Rough training
# run (policy_locomotion_status.md), and reusable for any other Rough
# checkpoint someone wants a structured look at.
#
# Default coverage: fixed sub-terrain type at the non-"full coverage" levels,
# all 6 sub-terrain types shown at the "full coverage" levels -- spans the
# full easy-to-hard range with full type coverage at a couple of
# representative difficulty points, without the combinatorial cost of every
# (level, type) pair (10 levels x 6 types = 60 clips).
#
# Plain walking, no forced pushes -- reuses the Play task (which already
# disables push_disturbance) since this is a qualitative "does the policy
# look sane on real terrain" check, not the push-recovery eval
# (policy/locomotion/eval/push_recovery_eval.py covers that, separately).

"""Structured Rough-terrain video showcase for a trained checkpoint."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
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

parser = argparse.ArgumentParser(description="Structured Rough-terrain video showcase for a trained checkpoint.")
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
    "--levels", type=str, default="0,2,4,6,8", help="Comma-separated terrain difficulty rows to show (0=easiest)."
)
parser.add_argument(
    "--full-coverage-levels",
    type=str,
    default="4,8",
    help="Subset of --levels to show all 6 sub-terrain types at, instead of just --fixed-type.",
)
parser.add_argument(
    "--fixed-type",
    type=str,
    default="boxes",
    choices=SUB_TERRAIN_TYPES,
    help="Sub-terrain type used for --levels entries not in --full-coverage-levels.",
)
parser.add_argument("--steps-per-clip", type=int, default=200, help="Simulation steps recorded per clip.")
parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Where to write clips (default: <run_dir>/terrain_showcase/ next to the loaded checkpoint).",
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

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from policy.locomotion.core.script_utils import checkpoints_root
from policy.locomotion.core.terrain_pinning import pin_terrain, sub_terrain_column_for_type
from policy.locomotion.core.video_capture import record_condition_clips


def _build_conditions(levels: list[int], full_coverage_levels: set[int], fixed_type: str) -> list[tuple[int, str]]:
    """One (level, type_name) pair per clip -- all 6 types at each
    full-coverage level, just `fixed_type` everywhere else."""
    conditions = []
    for level in levels:
        types = SUB_TERRAIN_TYPES if level in full_coverage_levels else [fixed_type]
        conditions.extend((level, type_name) for type_name in types)
    return conditions


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    levels = [int(x) for x in args_cli.levels.split(",")]
    full_coverage_levels = {int(x) for x in args_cli.full_coverage_levels.split(",")} if args_cli.full_coverage_levels else set()
    conditions = _build_conditions(levels, full_coverage_levels, args_cli.fixed_type)

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError(
            f"Task '{args_cli.task}' has no terrain generator (no difficulty grid) -- pass a Rough task instead."
        )
    num_cols = terrain_generator.num_cols
    type_columns = {
        type_name: sub_terrain_column_for_type(terrain_generator, type_name, num_cols) for type_name in SUB_TERRAIN_TYPES
    }

    env_cfg.scene.num_envs = len(conditions)

    log_root_path = checkpoints_root(os.path.dirname(__file__), agent_cfg)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    levels_per_env = [level for level, _ in conditions]
    types_per_env = [type_columns[type_name] for _, type_name in conditions]
    pin_terrain(env, levels_per_env, types_per_env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else Path(resume_path).parent / "terrain_showcase"
    condition_names = [f"level{level}_{type_name}" for level, type_name in conditions]
    record_condition_clips(env, policy, condition_names, args_cli.steps_per_clip, output_dir, fps=args_cli.fps)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
