"""Event terms for the arm embodiment: hold the arm at a per-episode stowed offset, add a payload mass.

Pure logic where it can be (the offset sampler is tested without Isaac); the event functions themselves need the
running articulation and are only imported after the app launches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def sample_arm_offset(rand: Any, size: tuple[int, ...], max_offset_rad: float) -> Any:
    """Uniform offset in ``[-max_offset_rad, +max_offset_rad]``; ``rand`` is a torch generator-like callable."""
    return (rand(size) * 2.0 - 1.0) * max_offset_rad


def wrap_to_limit(value: float, low: float, high: float) -> float:
    """Clamp into the joint limit (no wrap-around; arm joints do not cross +-pi here)."""
    return max(low, min(high, value))


def arm_offset_bounds(stowed: Sequence[float], max_offset_rad: float,
                      limits: Sequence[tuple[float, float]] | None = None) -> tuple[list[float], list[float]]:
    """Low/high target for the stowed pose +- offset, clamped to the limits when given (pure)."""
    lows, highs = [], []
    for index, value in enumerate(stowed):
        low = value - max_offset_rad
        high = value + max_offset_rad
        if limits is not None:
            jl, jh = limits[index]
            low, high = wrap_to_limit(low, jl, jh), wrap_to_limit(high, jl, jh)
        lows.append(low)
        highs.append(high)
    return lows, highs


def _env_id_tensor(env_ids: Any, asset: Any):
    """Isaac Lab hands events ``None``, a slice (``env_ids[0:3]``) or a tensor; normalise to a tensor of indices."""
    import torch

    if env_ids is None:
        return torch.arange(asset.num_instances, device=asset.device)
    if isinstance(env_ids, slice):
        return torch.arange(asset.num_instances, device=asset.device)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids
    return torch.tensor(list(env_ids), device=asset.device)


def _as_tensor(values: Sequence[float], device, dtype):
    import torch

    return torch.tensor(list(values), device=device, dtype=dtype)


def randomize_arm_hold(env: Any, env_ids: Any, asset_cfg: Any, max_offset_rad: float,
                       stowed: Sequence[float]) -> None:
    """Reset the arm's position target to stowed + a per-env random offset, and put the joints there.

    The arm is not in the action vector, so this target persists: the implicit actuators hold it, and the legs
    policy sees a slightly different arm pose every episode.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    env_ids = _env_id_tensor(env_ids, asset)
    count = len(env_ids)
    base = asset.data.default_joint_pos.new_tensor(list(stowed))[ids]
    offsets = sample_arm_offset(lambda size: torch.rand(size, device=asset.device),
                                (count, len(ids)), max_offset_rad)
    limits = asset.data.joint_pos_limits[env_ids][:, ids]  # (count, n, 2)
    target = (base.unsqueeze(0) + offsets).clamp(min=limits[..., 0], max=limits[..., 1])
    asset.set_joint_position_target(target, joint_ids=ids, env_ids=env_ids)
    asset.write_joint_state_to_sim(target, torch.zeros_like(target), joint_ids=ids, env_ids=env_ids)
    asset.set_joint_position_target(target, joint_ids=ids, env_ids=env_ids)


def randomize_payload_mass(env: Any, env_ids: Any, asset_cfg: Any,
                           mass_range_kg: Sequence[float]) -> None:
    """Add a uniform payload between ``mass_range_kg`` to the given bodies (startup event)."""
    import torch

    asset = env.scene[asset_cfg.name]
    env_ids = _env_id_tensor(env_ids, asset)
    low, high = float(mass_range_kg[0]), float(mass_range_kg[1])
    count = len(env_ids)
    masses = torch.rand((count, len(asset_cfg.body_ids)), device=asset.device) * (high - low) + low
    asset.set_masses(masses, body_ids=asset_cfg.body_ids, env_ids=env_ids)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
