"""Loads a named YAML preset from policy/locomotion/core/configs/ and converts it
into Hydra override argv tokens ("key.path=value" strings) -- so presets stay
in small, git-trackable, easily-edited files instead of long CLI invocations,
while reusing Hydra's own override syntax (already the config engine
train.py/play.py run on via Isaac Lab's hydra_task_config) rather than
inventing a new format.

Deliberately NOT under policy.locomotion: this module has to be importable
before AppLauncher/Isaac Sim initializes (it injects overrides into sys.argv
before Hydra parses them), and policy.locomotion's own __init__.py imports
Isaac Lab modules that aren't safe to touch that early.
"""

from __future__ import annotations

import os

import yaml


def load_preset_overrides(name: str) -> list[str]:
    """Reads policy/locomotion/core/configs/<name>.yaml and returns Hydra override
    strings, e.g. ["env.scene.num_envs=32", "agent.max_iterations=3"].
    """
    path = os.path.join(os.path.dirname(__file__), "locomotion", "core", "configs", f"{name}.yaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return [f"{key}={value}" for key, value in data.items()]
