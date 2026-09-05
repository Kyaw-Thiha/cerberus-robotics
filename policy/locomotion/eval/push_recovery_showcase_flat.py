# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Flat-terrain counterpart to push_recovery_showcase.py -- same one-launch,
# camera-switching video capture (core/video_capture.py), but magnitude is
# the only condition axis: Flat has no terrain-difficulty grid to vary
# alongside it (push_recovery_showcase.py hard-requires one -- see its own
# `terrain_generator is None` check). Nothing else differs: same manual
# apply_single_push firing via on_window_step (not an EventTerm -- see
# video_capture.py's module docstring for why), same impulse-clear EventTerm
# pairing (push_impulse_event.py / REFERENCES.md's "True impulse, not a
# sustained force").

"""Push-recovery video showcase for a Flat-terrain checkpoint: one clip per
magnitude."""

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

parser = argparse.ArgumentParser(description="Push-recovery video showcase for a Flat-terrain checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0",
    help="Base task to build the showcase env from (Flat only -- see push_recovery_showcase.py for Rough).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--magnitudes", type=str, default="0,80,120,160,200", help="Comma-separated push magnitudes (N), one clip each."
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
from policy.locomotion.core.video_capture import record_condition_clips


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    magnitudes = [float(m) for m in args_cli.magnitudes.split(",")]

    num_envs = len(magnitudes)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = args_cli.push_time_s + args_cli.window_s + 1.0

    # See push_recovery_showcase.py's identical block: apply_single_push is fired
    # manually below (not an EventTerm trigger), so without this clear pass its
    # force would persist for the rest of each clip instead of a true short impulse.
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
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    dt = raw_env.step_dt
    push_local_step = round(args_cli.push_time_s / dt)
    steps_per_clip = round((args_cli.push_time_s + args_cli.window_s) / dt)

    magnitudes_tensor = torch.tensor(magnitudes, device=raw_env.device)
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
    condition_names = [f"mag{int(mag)}" for mag in magnitudes]
    record_condition_clips(
        env, policy, condition_names, steps_per_clip, output_dir, fps=args_cli.fps, on_window_step=fire_push_at_window_start
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
