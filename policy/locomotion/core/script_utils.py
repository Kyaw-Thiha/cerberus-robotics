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


def configure_video_recorders(env_cfg, args_cli, log_dir: str, phase: str, checkpoint_path: str | None = None) -> bool:
    """Configure Isaac Lab 3.0's own video recorder on `env_cfg` when --video was passed.

    Lab 3.0 GA no longer renders `render_mode="rgb_array"` frames (`env.render()` returns None, so gym's
    RecordVideo wrote nothing and `play.py --video` still exited 0). Recording is configured on
    `env_cfg.video_recorders` instead (a headless Kit visualizer is injected when none exists).

    Call it after the env cfg is resolved and BEFORE `gym.make`. Returns True when the Lab-native recorder is
    configured (the caller must then skip RecordVideo and render_mode), False on Lab 2.x (no such API) or when
    --video was not passed.
    """
    if not args_cli.video:
        return False
    try:
        from isaaclab_rl.entrypoints.common import (
            apply_video_recording,
            pre_launch_video_config,
        )
    except ImportError:
        return False
    pre_launch_video_config(env_cfg, args_cli)
    apply_video_recording(env_cfg, log_dir, args_cli, subdir=phase, checkpoint_path=checkpoint_path)
    return True


def ensure_capture_visualizer(env_cfg) -> None:
    """Make sure a Kit visualizer exists so `capture_viewport_frame` has something to read (Lab 3.0).

    For scripts that write their own frames (the showcases) rather than using `--video`. Lab 2.x: no-op.
    """
    try:
        from isaaclab_rl.entrypoints.common import pre_launch_video_config
    except ImportError:
        return
    import argparse

    pre_launch_video_config(env_cfg, argparse.Namespace(video=True))


def point_viewport_at_env(raw_env, env_index: int) -> None:
    """Aim the viewport camera at env `env_index`.

    Lab 2.x: `raw_env.viewport_camera_controller.set_view_env_index`. Lab 3.0 removed that controller; the camera
    tracks an env through the Kit visualizer (`origin_type="env"`, `origin_env_index`, then `reapply_origin()`).
    """
    controller = getattr(raw_env, "viewport_camera_controller", None)
    if controller is not None:
        controller.set_view_env_index(env_index)
        return
    kit = [v for v in getattr(raw_env.sim, "visualizers", []) if hasattr(v, "reapply_origin")]
    if not kit:
        raise RuntimeError("no Kit visualizer is active; call ensure_capture_visualizer(env_cfg) before gym.make")
    for viz in kit:
        viz.cfg.origin_type = "env"
        viz.cfg.origin_env_index = env_index
        viz.reapply_origin()


def capture_viewport_frame(raw_env):
    """One RGB frame (H, W, 3 uint8 ndarray) of the viewport; raises if the stack cannot produce one.

    Lab 2.x: `env.render()` with render_mode="rgb_array". Lab 3.0: the first active visualizer that can
    capture (`env.sim.visualizers[i].render_rgb_array()`). Never returns None: a missing frame is an error,
    because writing empty clips and exiting 0 is exactly the failure this replaces.
    """
    visualizers = getattr(getattr(raw_env, "sim", None), "visualizers", None)
    if visualizers is None:
        frame = raw_env.render()
    else:
        capture = [v for v in visualizers if hasattr(v, "render_rgb_array")]
        if not capture:
            raise RuntimeError("no capture-capable visualizer is active; call ensure_capture_visualizer(env_cfg) before gym.make")
        frame = capture[0].render_rgb_array()
    if frame is None:
        raise RuntimeError("viewport capture returned no frame (rendering is not producing images)")
    return frame


def maybe_record_video(env, args_cli, log_dir: str, phase: str, native: bool = False):
    """Wraps env in gym.wrappers.RecordVideo if --video was passed.

    `phase` is "train" (periodic clips, one every --video_interval steps) or
    "play" (a single clip from the start) -- matches Isaac Lab's own
    train.py/play.py video behavior exactly, just factored out.
    """
    if not args_cli.video or native:
        return env

    import gymnasium as gym
    from isaaclab.utils.dict import print_dict

    if phase == "train":
        step_trigger = lambda step: step % args_cli.video_interval == 0
    elif phase == "play":
        step_trigger = lambda step: step == 0
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
