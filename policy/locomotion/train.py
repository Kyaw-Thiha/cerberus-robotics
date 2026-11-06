# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Thin wrapper around Isaac Lab's canonical scripts/reinforcement_learning/rsl_rl/train.py
# (verified against Isaac Lab v2.3.2's own copy) -- per the "don't
# hand-tune beyond what's established" instruction, the PPO/OnPolicyRunner
# training logic itself is unmodified. On top of that unmodified core:
#   - registers our Cerberus Go2 tasks (policy.locomotion) and defaults --task
#     to our flat task -- the task registration Isaac Lab's own extension
#     template expects external tasks to hook in with (see the "PLACEHOLDER:
#     Extension template" comment in Isaac Lab's own train.py)
#   - --config <name>: loads a named preset from policy/locomotion/core/configs/
#     (see policy/config_presets.py)
#   - wandb is the default logger, project "cerberus-robotics" (override with
#     --logger tensorboard for a quick local run with no network access)
#   - swaps in BestCheckpointOnPolicyRunner (best_checkpoint_runner.py) for
#     the stock OnPolicyRunner -- same training behavior, additionally saves
#     model_best_<easy|medium|hard>.pt
#   - log-dir setup and video-wrapping, identical in behavior to Isaac Lab's
#     own script, are factored into script_utils.py for readability

"""Train the Cerberus Go2 locomotion policy with RSL-RL."""

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
    hydra_task_config,
    load_cli_args,
    make_runner,
)
from policy.cli_common import (  # isort: skip
    add_common_args,
    add_launcher_args,
    add_platform_args,
    apply_config_preset,
    local_rank,
    prepare_rendering,
)

# Isaac Lab's RSL-RL cli_args: scripts/reinforcement_learning/rsl_rl/cli_args.py on Lab 2.3.2 (its directory
# is put on sys.path), isaaclab_rl.entrypoints.backends.cli_args_rsl_rl on Lab 3.0.
cli_args = load_cli_args()

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the Cerberus locomotion policy with RSL-RL.")
add_common_args(
    parser,
    task_help="Raw gym task id override -- bypasses --platform/--terrain entirely.",
    video_help="Record videos during training.",
)
add_platform_args(parser)
parser.set_defaults(terrain="flat")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# wandb by default (project "cerberus-robotics"), not Isaac Lab's stock None
# (which falls back to tensorboard-only) -- override with --logger tensorboard
# for a quick local check that doesn't need network access.
parser.set_defaults(logger="wandb", log_project_name="cerberus-robotics")
# append AppLauncher cli args (AppLauncher.add_app_launcher_args, plus a --headless compat flag on Lab 3.0)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video; fails fast if the host's GPU driver is known
# to crash Isaac Sim's RTX renderer, rather than paying the full boot only to hit the same crash
prepare_rendering(args_cli)

# preset-provided overrides go first so explicit CLI overrides (already in
# hydra_args) win for any key both specify.
hydra_args = apply_config_preset(args_cli, hydra_args)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
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
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from common.cerberus_compat.isaaclab_tasks import install_legacy_task_path_aliases

install_legacy_task_path_aliases()
import policy.locomotion  # noqa: F401  -- registers the Cerberus Go2 tasks
from isaaclab_tasks.utils import get_checkpoint_path
from policy.cli_common import resolve_task_id
from policy.locomotion.core.best_checkpoint_runner import BestCheckpointOnPolicyRunner
from policy.locomotion.core.script_utils import (
    checkpoints_root,
    configure_video_recorders,
    maybe_record_video,
    new_run_log_dir,
)

# now that policy.locomotion has registered every platform's gym tasks, resolve
# --platform/--terrain to a concrete task id (unless --task was passed explicitly)
args_cli.task = resolve_task_id(args_cli, play=False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{local_rank(app_launcher)}"
        agent_cfg.device = f"cuda:{local_rank(app_launcher)}"
        seed = agent_cfg.seed + local_rank(app_launcher)
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # per docs/project_structure.md: policy/locomotion/checkpoints/, not
    # Isaac Lab's default logs/rsl_rl/ -- keeps this project's checkpoint
    # location consistent with the other frozen-artifact directories
    log_root_path = checkpoints_root(os.path.dirname(__file__), agent_cfg)
    log_dir = new_run_log_dir(log_root_path, agent_cfg)

    env_cfg.log_dir = log_dir

    native_video = configure_video_recorders(env_cfg, args_cli, log_dir, phase="train")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video and not native_video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = maybe_record_video(env, args_cli, log_dir, phase="train", native=native_video)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        # BestCheckpointOnPolicyRunner instead of stock OnPolicyRunner: also
        # writes model_best_<easy|medium|hard>.pt -- see best_checkpoint_runner.py.
        runner = make_runner(
            env, agent_cfg, log_dir=log_dir, device=agent_cfg.device, runner_cls=BestCheckpointOnPolicyRunner
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = make_runner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # clear error if the checkpoint is a rsl-rl 3.x (model_state_dict) file on rsl-rl 5.x or vice versa
        check_checkpoint_format(resume_path)
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
