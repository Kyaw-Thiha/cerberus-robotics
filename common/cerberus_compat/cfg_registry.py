"""Resolve a registered gym task's config entry point into a config object.

Isaac Lab registers each task with *strings* like
``"policy.locomotion.core.platforms.go2.env_cfg:UnitreeGo2RoughCerberusEnvCfg_PLAY"`` under
``gym.spec(task).kwargs["env_cfg_entry_point"]``. Calling that value does not work (it is a string),
and the helper that does the lookup (``load_cfg_from_registry``) has moved between Lab versions.
This module resolves the string itself -- ``module:attr``, instantiated with no arguments -- which is
the same convention the platform registry already uses, so scripts that build an env outside
``hydra_task_config`` (the stack probe, the checkpoint exporter, the adversary trainer) agree on how.

Import-safe without Isaac: only ``importlib`` and ``gymnasium`` are touched, and ``gymnasium`` is
imported lazily inside :func:`entry_point_for`.
"""

from __future__ import annotations

import importlib


def resolve_entry_point(entry_point: str):
    """``"pkg.module:ClassName"`` -> the instantiated object."""
    if not isinstance(entry_point, str) or ":" not in entry_point:
        raise ValueError(f"not a 'module:attr' entry point: {entry_point!r}")
    module_name, attr_path = entry_point.split(":", 1)
    target = importlib.import_module(module_name)
    for part in attr_path.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"entry point {entry_point!r} resolved to a non-callable {type(target).__name__}")
    return target()


def entry_point_for(task: str, key: str = "env_cfg_entry_point") -> str:
    """The registered entry-point string for ``task`` under ``key`` (``gym.spec`` kwargs)."""
    import gymnasium as gym

    spec = gym.spec(task)
    try:
        return spec.kwargs[key]
    except (AttributeError, KeyError) as e:
        raise KeyError(
            f"task {task!r} has no {key!r} in its registered kwargs "
            f"(started with: {sorted(getattr(spec, 'kwargs', {}) or {})})"
        ) from e


def load_cfg(task: str, key: str = "env_cfg_entry_point"):
    """Instantiate the config registered for ``task`` under ``key``.

    Accepts a registered value that is already a class/instance (some tasks register the object
    directly) as well as the usual ``module:attr`` string.
    """
    registered = entry_point_for(task, key)
    if callable(registered) or not isinstance(registered, str):
        return registered() if callable(registered) else registered
    return resolve_entry_point(registered)


def load_env_cfg(task: str):
    """The task's environment config, with the task's own Play/Train variant resolved already."""
    return load_cfg(task, "env_cfg_entry_point")


def load_agent_cfg(task: str):
    """The task's RL runner config (rsl-rl entry point)."""
    for key in ("rsl_rl_cfg_entry_point", "agent_cfg_entry_point"):
        try:
            return load_cfg(task, key)
        except KeyError:
            continue
    raise KeyError(f"task {task!r} registers no agent/rsl_rl config entry point")
