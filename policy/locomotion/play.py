# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Thin wrapper around Isaac Lab's canonical
# scripts/reinforcement_learning/rsl_rl/play.py (verified against Isaac Lab
# v2.3.2's own copy). On top of the unmodified inference/export logic:
#   - registers our Cerberus Go2 tasks and defaults --task to the flat Play variant
#   - --config <name>: same preset support as train.py (see policy/config_presets.py)
#   - log-dir setup and video-wrapping factored into script_utils.py for readability
# (no wandb default here -- play.py only runs inference, log_dir=None, nothing
# to log to a project.)

"""Play a trained Cerberus Go2 locomotion checkpoint with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# make `policy.config_presets` and `common.cerberus_compat` importable regardless of cwd -- needed
# before the imports right below.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# local imports
from common.cerberus_compat.rsl_rl_compat import (  # isort: skip
    check_checkpoint_format,
    export_policy,
    get_policy_for_export,
    hydra_task_config,
    load_cli_args,
    make_runner,
)
from policy.cli_common import (  # isort: skip
    add_common_args,
    add_launcher_args,
    add_platform_args,
    apply_config_preset,
    prepare_rendering,
)

# Isaac Lab's RSL-RL cli_args: scripts/reinforcement_learning/rsl_rl/cli_args.py on Lab 2.3.2 (its directory
# is put on sys.path), isaaclab_rl.entrypoints.backends.cli_args_rsl_rl on Lab 3.0.
cli_args = load_cli_args()

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained Cerberus locomotion checkpoint with RSL-RL.")
add_common_args(
    parser,
    task_help="Raw gym task id override -- bypasses --platform/--terrain entirely.",
    video_help="Record videos during play.",
)
add_platform_args(parser)
parser.set_defaults(terrain="flat")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
prepare_rendering(args_cli)

hydra_args = apply_config_preset(args_cli, hydra_args)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from common.cerberus_compat.isaaclab_tasks import install_legacy_task_path_aliases

install_legacy_task_path_aliases()
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils import get_checkpoint_path
from policy.cli_common import resolve_task_id
from policy.locomotion.core.script_utils import checkpoints_root, configure_video_recorders, maybe_record_video

# now that policy.locomotion has registered every platform's gym tasks, resolve
# --platform/--terrain to a concrete task id (unless --task was passed explicitly)
args_cli.task = resolve_task_id(args_cli, play=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # per docs/project_structure.md: policy/locomotion/checkpoints/
    log_root_path = checkpoints_root(os.path.dirname(__file__), agent_cfg)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    native_video = configure_video_recorders(env_cfg, args_cli, log_dir, phase="play", checkpoint_path=resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video and not native_video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = maybe_record_video(env, args_cli, log_dir, phase="play", native=native_video)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name in ("OnPolicyRunner", "DistillationRunner"):
        # runner class picked from agent_cfg.class_name (OnPolicyRunner / DistillationRunner)
        runner = make_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # clear error if the checkpoint is a rsl-rl 3.x (model_state_dict) file on rsl-rl 5.x or vice versa
    check_checkpoint_format(resume_path)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # rsl-rl 3.x: runner.alg.policy / alg.actor_critic; rsl-rl 5.x: the actor model (runner.alg.get_policy())
    policy_nn = get_policy_for_export(runner)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy(runner, export_model_dir, jit_filename="policy.pt", onnx_filename="policy.onnx")

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
