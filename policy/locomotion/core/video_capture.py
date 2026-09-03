"""Multi-condition video capture from a single Isaac Sim launch.

Isaac Lab's viewport camera watches exactly one env for a run's whole
lifetime -- there's no per-env Camera sensor here (perception/exteroception
is Phase 1's job, not Phase 0's, see REFERENCES.md). Getting one clip per
"condition" (e.g. one terrain difficulty level) the straightforward way --
one script invocation per clip -- means paying Isaac Sim's app-launch
overhead (~20-30s, measured repeatedly against this project's own image)
once per clip. This instead steps every condition's env together in ONE
continuous rollout and switches which env the camera is pointed at (and
which file frames are being written to) every `steps_per_clip` steps --
one launch, N clips, launch overhead paid once instead of N times.

Not RecordVideo-based: gym.wrappers.RecordVideo only supports one fixed
target env for the wrapper's entire lifetime, which doesn't fit "switch
targets mid-run" -- this writes frames directly via imageio instead.

Callers are responsible for pinning each env to the condition it should show
(e.g. via core.terrain_pinning.pin_terrain) before calling record_condition_clips.

If a condition needs a scripted event partway through its own clip (e.g. a
push-recovery showcase firing a push at a specific moment) DO NOT use Isaac
Lab's interval-mode EventTerm for it -- that fires based on each env's own
elapsed time since *its own* last reset, not relative to when this loop
starts watching it. Since env i's window doesn't start until step
i * steps_per_clip into the shared rollout, an interval-mode event anchored
to episode-elapsed time would already have fired long before that env's
window arrives, for every env after the first -- showing only the aftermath,
never the event itself. Use `on_window_step` instead, which fires relative to
each condition's own window (local_step 0 is that env's first recorded step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio
import torch


def record_condition_clips(
    env,
    policy,
    condition_names: list[str],
    steps_per_clip: int,
    output_dir: Path,
    fps: int = 30,
    on_window_step: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Steps `env` for len(condition_names) * steps_per_clip steps total, one
    clip per condition. condition_names[i] names env i's condition (already
    pinned by the caller) and becomes that clip's filename.

    Every env keeps stepping for the whole rollout regardless of whose window
    it currently is -- an env not yet being recorded may fall/reset in the
    meantime, same as it would in any other rollout. That's expected: each
    clip shows *a* representative episode under that env's pinned condition,
    not necessarily one starting fresh at step 0.

    `on_window_step(env_idx, local_step)`, if given, is called after every
    env.step() call, with local_step counting from 0 at the start of env_idx's
    own window -- see the module docstring for why this exists instead of an
    EventTerm.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_env = env.unwrapped
    camera = raw_env.viewport_camera_controller
    if camera is None:
        raise RuntimeError(
            "No viewport camera controller available -- run with rendering enabled (--enable_cameras)."
        )
    if raw_env.num_envs != len(condition_names):
        raise ValueError(
            f"condition_names has {len(condition_names)} entries but env has {raw_env.num_envs} envs -- "
            "one condition per env is required."
        )

    written_paths: list[Path] = []
    obs = env.get_observations()
    with torch.inference_mode():
        for env_idx, name in enumerate(condition_names):
            camera.set_view_env_index(env_idx)
            out_path = output_dir / f"{name}.mp4"
            writer = imageio.get_writer(str(out_path), fps=fps)
            for local_step in range(steps_per_clip):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if on_window_step is not None:
                    on_window_step(env_idx, local_step)
                frame = raw_env.render()
                if frame is not None:
                    writer.append_data(frame)
            writer.close()
            written_paths.append(out_path)
            print(f"[INFO] Wrote {out_path}")

    return written_paths
