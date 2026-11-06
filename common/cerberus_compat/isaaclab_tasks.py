"""Compatibility aliases for Isaac Lab task package paths moved in the 3.0 line."""

from __future__ import annotations

import importlib
import sys
import types


_LEGACY_MODULES = {
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg": (
        "isaaclab_tasks.core.velocity.config.go2.flat_env_cfg"
    ),
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg": (
        "isaaclab_tasks.core.velocity.config.go2.rough_env_cfg"
    ),
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg": (
        "isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg"
    ),
}


def install_legacy_task_path_aliases() -> None:
    """Make Lab 2.x ``manager_based`` task imports resolve on Lab 3.0.

    Cerberus' Go2 task config imports were written against Lab 2.x's package tree,
    while Lab 3.0 moved the same upstream configs under ``isaaclab_tasks.core``.
    Install only the exact legacy modules Cerberus uses; Isaac Lab itself remains
    unmodified.
    """
    for legacy_name, current_name in _LEGACY_MODULES.items():
        parts = legacy_name.split(".")
        for end in range(2, len(parts)):
            package_name = ".".join(parts[:end])
            if package_name in sys.modules:
                continue
            parent_name, child_name = package_name.rsplit(".", 1)
            parent = importlib.import_module(parent_name)
            package = types.ModuleType(package_name)
            package.__package__ = package_name
            package.__path__ = []
            sys.modules[package_name] = package
            setattr(parent, child_name, package)
        current = importlib.import_module(current_name)
        sys.modules[legacy_name] = current
        parent_name, child_name = legacy_name.rsplit(".", 1)
        setattr(importlib.import_module(parent_name), child_name, current)
