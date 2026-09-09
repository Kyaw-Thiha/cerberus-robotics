"""Per-robot-platform contract. A new robot platform is a new PlatformCfg
instance (see core/platforms/go2/platform.py for the reference example) plus
a new env_cfg module -- nothing else in the training/eval/CLI stack should
need touching to add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class PlatformCfg:
    """Everything platform-specific the training/eval/CLI stack needs from a
    robot."""

    name: str  # e.g. "go2" -- used for --platform, checkpoint prefix, registry key

    # gym task ids
    flat_task_id: str
    flat_play_task_id: str
    rough_task_id: str
    rough_play_task_id: str

    # lazy "module:ClassName" strings (Isaac Lab's env_cfg_entry_point convention) --
    # the _PLAY variant's class name is derived by appending "_PLAY", not stored
    # separately.
    flat_env_cfg_entry_point: str
    rough_env_cfg_entry_point: str

    # actual imported class objects (Isaac Lab's rsl_rl_cfg_entry_point convention --
    # NOT a string, unlike the env_cfg entry points above)
    flat_agent_cfg: Any
    rough_agent_cfg: Any

    # push-disturbance curriculum, mass/geometry-tuned per robot -- see
    # policy/locomotion/REFERENCES.md for how Go2's values were derived; a new
    # platform must re-derive its own rather than reuse Go2's.
    push_peak_force_n: float
    push_peak_torque_nm: float
    root_body_name: str = "base"

    # terrain rescale (rough only), relative to Isaac Lab's stock ANYmal-scale defaults
    stairs_step_height_scale: float = 1.0

    # must match this platform's agent cfg's num_steps_per_env (curriculum cadence --
    # see core/curriculum.py's NUM_STEPS_PER_ENV comment)
    num_steps_per_env: int = 24

    checkpoint_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.checkpoint_prefix:
            object.__setattr__(self, "checkpoint_prefix", f"{self.name}_locomotion")


_PLATFORMS: dict[str, PlatformCfg] = {}


def register_platform(cfg: PlatformCfg) -> None:
    """Records `cfg` in the platform registry and registers its four gym
    environments (flat/rough x train/play). Raises if `cfg.name` is already
    registered -- call this exactly once per platform, typically at import
    time of that platform's `platform.py` module.
    """
    if cfg.name in _PLATFORMS:
        raise ValueError(f"platform '{cfg.name}' already registered")
    _PLATFORMS[cfg.name] = cfg

    for task_id, env_cfg_entry_point, agent_cfg, is_play in (
        (cfg.flat_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, False),
        (cfg.flat_play_task_id, cfg.flat_env_cfg_entry_point, cfg.flat_agent_cfg, True),
        (cfg.rough_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, False),
        (cfg.rough_play_task_id, cfg.rough_env_cfg_entry_point, cfg.rough_agent_cfg, True),
    ):
        resolved_entry_point = f"{env_cfg_entry_point}_PLAY" if is_play else env_cfg_entry_point
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": resolved_entry_point,
                "rsl_rl_cfg_entry_point": agent_cfg,
            },
        )


def get_platform(name: str) -> PlatformCfg:
    try:
        return _PLATFORMS[name]
    except KeyError:
        raise ValueError(f"unknown platform '{name}', available: {sorted(_PLATFORMS)}") from None


def available_platforms() -> list[str]:
    return sorted(_PLATFORMS)
