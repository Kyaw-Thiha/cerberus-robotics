#!/usr/bin/env python3
"""Smallest possible Isaac Sim/Lab headless launch: ground-plane Go2, 1 env, a few steps.

Run by ``scripts/pod_launch_probe.py`` under ``isaaclab.sh -p``, never on its own. It downloads nothing: the flat
Cerberus task uses a generated ground plane, not a scene USD and not a checkpoint. Its only job is to prove that Kit
starts and the task steps on this host before a driver script spends time on downloads or a long job.

Prints ``PROBE_OK`` on success; any crash is classified by the caller from the process exit code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TASK = "Isaac-Velocity-Flat-Unitree-Go2-Cerberus-Play-v0"
STEPS = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import importlib

        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401

        importlib.import_module("policy.locomotion")
        from common.cerberus_compat.cfg_registry import load_env_cfg

        cfg = load_env_cfg(TASK)
        cfg.scene.num_envs = 1
        env = gym.make(TASK, cfg=cfg)
        raw = env.unwrapped
        env.reset()
        actions = torch.zeros((1, raw.action_manager.total_action_dim), device=raw.device)
        for _ in range(STEPS):
            env.step(actions)
        env.close()
        print(f"PROBE_OK task={TASK} envs=1 steps={STEPS}")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
