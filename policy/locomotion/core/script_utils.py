"""Small helpers shared by train.py and play.py, factored out for readability.

Safe to import only after AppLauncher/Isaac Sim has initialized -- these are
called from inside each script's main(), never before. (Unlike
policy/config_presets.py, which has to run before AppLauncher and therefore
can't live under this package -- see that file's docstring.)
"""

from __future__ import annotations

import os
from datetime import datetime


def checkpoints_root(script_dir: str, agent_cfg) -> str:
    """policy/locomotion/checkpoints/<experiment_name>, per docs/project_structure.md
    (Isaac Lab's own scripts default to logs/rsl_rl/<experiment_name> instead).
    """
    return os.path.abspath(os.path.join(script_dir, "checkpoints", agent_cfg.experiment_name))


def new_run_log_dir(log_root_path: str, agent_cfg) -> str:
    """A fresh timestamped run directory under log_root_path, matching Isaac
    Lab's own train.py naming convention ({timestamp}[_{run_name}]).
    """
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    return os.path.join(log_root_path, log_dir)


def maybe_record_video(env, args_cli, log_dir: str, phase: str):
    """Wraps env in gym.wrappers.RecordVideo if --video was passed.

    `phase` is "train" (periodic clips, one every --video_interval steps) or
    "play" (a single clip from the start) -- matches Isaac Lab's own
    train.py/play.py video behavior exactly, just factored out.
    """
    if not args_cli.video:
        return env

    import gymnasium as gym
    from isaaclab.utils.dict import print_dict

    if phase == "train":
        step_trigger = lambda step: step % args_cli.video_interval == 0  # noqa: E731
    elif phase == "play":
        step_trigger = lambda step: step == 0  # noqa: E731
    else:
        raise ValueError(f"Unknown phase: {phase!r}, expected 'train' or 'play'")

    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", phase),
        "step_trigger": step_trigger,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print(f"[INFO] Recording videos during {phase}.")
    print_dict(video_kwargs, nesting=4)
    return gym.wrappers.RecordVideo(env, **video_kwargs)
