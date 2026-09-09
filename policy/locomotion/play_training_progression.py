# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training-progression video: one clip per periodic checkpoint from a single
# training run, same terrain condition held fixed across every clip, so what
# changes between clips is purely the policy's own gait quality over training
# -- not a different checkpoint invocation's random terrain draw. Deliberately
# NOT train.py's --video flag: that renders during training itself (slows the
# actual run for no benefit -- see RUN_001.md's "watch gait evolve across
# checkpoints" note, discussed but never executed) and would show only the
# single checkpoint active at each recorded step, not a comparable montage
# across the whole run. This instead runs once, after training, loading each
# periodic checkpoint in turn against the same env.
#
# One Isaac Sim launch, one persistent single-env rollout, reloaded policy
# weights between clips (core/video_capture.py's record_condition_clips
# doesn't fit here -- it switches CAMERA between many envs under one fixed
# policy; this needs the opposite: one env, many policies).

"""Training-progression video: one clip per periodic checkpoint, same terrain held fixed."""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(os.environ["ISAACLAB_PATH"], "scripts", "reinforcement_learning", "rsl_rl"))

import cli_args  # isort: skip
from policy.cli_common import check_gpu_driver_for_rendering  # isort: skip

parser = argparse.ArgumentParser(description="Training-progression video: one clip per periodic checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the env from.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Training run directory containing periodic model_<iter>.pt checkpoints (e.g. "
    "policy/locomotion/checkpoints/unitree_go2_rough/<timestamp>).",
)
parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=150,
    help="Show every checkpoint whose iteration number is a multiple of this (must itself be a multiple of "
    "the run's save_interval, 50 by default). The final checkpoint is always included even if it doesn't "
    "land on this interval.",
)
parser.add_argument(
    "--terrain-level", type=int, default=2, help="Fixed terrain difficulty row held constant across every clip."
)
parser.add_argument(
    "--sub-terrain-type",
    type=str,
    default="random_rough",
    choices=["pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"],
    help="Fixed sub-terrain type held constant across every clip -- deliberately not boxes (this checkpoint's "
    "weakest type, see RUN_001.md) or a stair type (asymmetric spawn-plateau dynamics, see RUN_001.md), since "
    "the point here is showing gait-quality progression, not terrain-difficulty struggle.",
)
parser.add_argument("--steps-per-clip", type=int, default=200, help="Simulation steps recorded per clip.")
parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Where to write clips (default: <run-dir>/training_progression/).",
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

import imageio
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from policy.locomotion.core.terrain_pinning import pin_terrain

CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def _select_checkpoints(run_dir: Path, interval: int) -> list[tuple[int, Path]]:
    """Every periodic checkpoint whose iteration is a multiple of `interval`,
    plus the single highest-iteration checkpoint in the run dir regardless of
    whether it lands on that interval -- always show the final trained state."""
    found: dict[int, Path] = {}
    for path_str in glob.glob(str(run_dir / "model_*.pt")):
        m = CHECKPOINT_RE.match(Path(path_str).name)
        if m:
            found[int(m.group(1))] = Path(path_str)
    if not found:
        raise RuntimeError(f"No model_<iter>.pt checkpoints found under {run_dir}")
    final_iter = max(found)
    selected_iters = sorted({it for it in found if it % interval == 0} | {final_iter})
    return [(it, found[it]) for it in selected_iters]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    run_dir = Path(args_cli.run_dir)
    checkpoints = _select_checkpoints(run_dir, args_cli.checkpoint_interval)
    print(f"[INFO] Recording {len(checkpoints)} checkpoints: {[it for it, _ in checkpoints]}")

    env_cfg.scene.num_envs = 1

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    pin_terrain(env, [args_cli.terrain_level], [_column_for_fixed_type(env_cfg, args_cli.sub_terrain_type)])
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    raw_env = env.unwrapped
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else run_dir / "training_progression"
    output_dir.mkdir(parents=True, exist_ok=True)

    # One persistent inference_mode context for the WHOLE loop, not re-entered per
    # clip -- a tensor touched inside torch.inference_mode() (e.g. env.step()'s
    # internal state writes) stays permanently marked as an "inference tensor"
    # even after that block exits, and a later in-place write to it from OUTSIDE
    # inference_mode (e.g. the next clip's raw_env.reset(), which happened between
    # per-clip `with` blocks in an earlier version of this script) raises
    # `RuntimeError: Inplace update to inference tensor outside InferenceMode is
    # not allowed`. Found for real on a pod run: clip 1 (model_0.pt) wrote fine,
    # clip 2's reset() crashed immediately. Fixed by keeping every reset AND step
    # for every checkpoint inside the same inference_mode scope from the start.
    with torch.inference_mode():
        for iteration, ckpt_path in checkpoints:
            print(f"[INFO] Loading checkpoint: {ckpt_path}")
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(ckpt_path))
            policy = runner.get_inference_policy(device=raw_env.device)

            raw_env.reset()
            obs = env.get_observations()

            out_path = output_dir / f"iter_{iteration:04d}.mp4"
            writer = imageio.get_writer(str(out_path), fps=args_cli.fps)
            for _ in range(args_cli.steps_per_clip):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                frame = raw_env.render()
                if frame is not None:
                    writer.append_data(frame)
            writer.close()
            print(f"[INFO] Wrote {out_path}")

    env.close()


def _column_for_fixed_type(env_cfg, type_name: str) -> int:
    from policy.locomotion.core.terrain_pinning import sub_terrain_column_for_type

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise RuntimeError(f"Task '{args_cli.task}' has no terrain generator -- pass a Rough task instead.")
    return sub_terrain_column_for_type(terrain_generator, type_name, terrain_generator.num_cols)


if __name__ == "__main__":
    main()
    simulation_app.close()
