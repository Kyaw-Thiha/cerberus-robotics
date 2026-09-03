"""Shared argparse CLI args for train.py/play.py -- the ~7 flags and the
--config preset-loading logic that were duplicated near-verbatim between the
two scripts.

Deliberately NOT under policy.locomotion, same reason as policy/config_presets.py:
this has to be importable before AppLauncher/Isaac Sim initializes (both scripts'
parser-building runs before that), and policy.locomotion's own __init__.py imports
Isaac Lab modules that aren't safe to touch that early.
"""

from __future__ import annotations

import argparse
import subprocess

from policy.config_presets import load_preset_overrides

# R590 branch (595.x) crashes Isaac Sim 5.1.0's RTX renderer during any
# rendering/video capture -- segfault in librtx.scenedb.plugin.so at Hydra
# engine creation, reproduced and confirmed by an Isaac Sim maintainer across
# multiple GPU models on both Windows and Linux:
# https://github.com/isaac-sim/IsaacSim/discussions/648
# Isaac Sim 5.1.0's validated Linux driver is 580.65.06. This is a host-level
# GPU driver -- on a rented pod, whichever physical machine gets allocated
# brings its own driver, so this can't be fixed in this image; the only
# mitigation is detecting it fast (before paying Isaac Sim's own multi-minute
# boot only to hit the same crash) and re-rolling the pod.
INCOMPATIBLE_DRIVER_BRANCHES = ("595",)


def check_gpu_driver_for_rendering() -> None:
    """Fails fast with a clear message if the host's GPU driver is on a
    branch known to crash Isaac Sim's RTX renderer. Call this before
    AppLauncher(args_cli) in any script that renders (--video, or a
    video-capture script like play_terrain_showcase.py where rendering isn't
    optional) -- not needed for pure headless-physics runs, which never
    create a Hydra render engine and so never hit this crash path regardless
    of driver.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[WARNING] Could not check GPU driver version ({e}) -- proceeding without the check.")
        return

    driver_version = result.stdout.strip()
    branch = driver_version.split(".")[0]
    if branch in INCOMPATIBLE_DRIVER_BRANCHES:
        raise SystemExit(
            f"GPU driver {driver_version} is on the R590 branch, which crashes Isaac Sim 5.1.0's RTX "
            "renderer during rendering/video capture (segfault in librtx.scenedb.plugin.so -- see "
            "https://github.com/isaac-sim/IsaacSim/discussions/648). Isaac Sim 5.1.0's validated Linux "
            "driver is 580.65.06. This is a host-level driver, not fixable in this image -- terminate "
            "this pod and provision a new one; a different host draw may have a compatible driver."
        )
    print(f"[INFO] GPU driver {driver_version} OK for rendering.")


def add_common_args(parser: argparse.ArgumentParser, default_task: str, task_help: str, video_help: str) -> None:
    """Adds the CLI args identical across train.py/play.py: --video,
    --video_length, --num_envs, --task, --agent, --seed, --config. Each
    script adds its own additional args after calling this.
    """
    parser.add_argument("--video", action="store_true", default=False, help=video_help)
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=default_task, help=task_help)
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Name of a preset under policy/locomotion/core/configs/ (e.g. 'smoke_test') -- its "
            "key: value pairs are applied as Hydra overrides, same as typing them on the "
            "CLI. Explicit CLI overrides still win over the preset for the same key."
        ),
    )


def apply_config_preset(args_cli: argparse.Namespace, hydra_args: list[str]) -> list[str]:
    """Prepends --config's preset overrides to hydra_args, if --config was given --
    explicit CLI overrides (already in hydra_args) still win for any key both
    specify, since Hydra takes the last value for a repeated key.
    """
    if args_cli.config is not None:
        return load_preset_overrides(args_cli.config) + hydra_args
    return hydra_args
