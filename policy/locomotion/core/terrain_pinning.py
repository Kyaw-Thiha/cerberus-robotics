"""Pins each env's terrain-difficulty row / sub-terrain-type column directly,
bypassing Isaac Lab's adaptive terrain_levels curriculum placement. Shared by
push_recovery_eval.py (the --terrain_levels axis in the push-recovery grid)
and play_terrain_showcase.py (structured Rough terrain review clips) -- both
need the exact same TerrainImporter.terrain_levels/.terrain_types/.env_origins
override, verified against Isaac Lab v2.3.2's own TerrainImporter.update_env_origins
for the correct indexing pattern (terrain_origins[level, type]).

Call pin_terrain after gym.make() but before RslRlVecEnvWrapper (which resets
on construction) or any other reset -- InteractiveScene.env_origins is a live
property reading straight off the TerrainImporter, so the override takes
effect at that first reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainGeneratorCfg


def sub_terrain_column_for_type(terrain_generator_cfg: TerrainGeneratorCfg, type_name: str, num_cols: int) -> int:
    """Returns the first column index generated as `type_name`, replicating
    TerrainGenerator._generate_curriculum_terrains' own column-to-subterrain
    assignment (proportional, by cumulative sub_terrains proportion) -- so
    this always matches whatever the actual terrain cfg's proportions are,
    rather than hardcoding column ranges that would silently drift if the
    upstream cfg's sub_terrains dict changes.

    Note: at a small enough num_cols, a low-proportion type may not get any
    column at all (see REFERENCES.md's note on the reduced 5-col Play grid
    structurally dropping one of the 6 Go2 rough sub-terrain types) -- this
    raises ValueError in that case rather than silently picking the wrong type.
    """
    names = list(terrain_generator_cfg.sub_terrains.keys())
    if type_name not in names:
        raise ValueError(f"Unknown sub-terrain type {type_name!r}; available: {names}")
    type_index = names.index(type_name)

    proportions = np.array([cfg.proportion for cfg in terrain_generator_cfg.sub_terrains.values()])
    proportions = proportions / proportions.sum()
    cumulative = np.cumsum(proportions)

    for col in range(num_cols):
        col_type_index = int(np.min(np.where(col / num_cols + 0.001 < cumulative)[0]))
        if col_type_index == type_index:
            return col
    raise ValueError(f"Sub-terrain type {type_name!r} has no column at num_cols={num_cols} (proportion too small).")


def pin_terrain(env, levels_per_env: list[int], types_per_env: list[int]) -> None:
    """Overwrites terrain.terrain_levels/.terrain_types and recomputes
    env_origins so each env spawns at its assigned (level, type) cell,
    instead of the adaptive curriculum's own random/performance-based
    placement.
    """
    terrain = env.unwrapped.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError(
            "pin_terrain requires a generator terrain (TerrainImporter.terrain_origins) -- "
            "this task has none (e.g. a Flat task)."
        )
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if max(levels_per_env) >= num_rows:
        raise ValueError(f"terrain level {max(levels_per_env)} exceeds available rows (0..{num_rows - 1}).")
    if max(types_per_env) >= num_cols:
        raise ValueError(f"terrain type column {max(types_per_env)} exceeds available columns (0..{num_cols - 1}).")

    device = terrain.device
    terrain.terrain_levels[:] = torch.tensor(levels_per_env, device=device, dtype=torch.long)
    terrain.terrain_types[:] = torch.tensor(types_per_env, device=device, dtype=torch.long)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
