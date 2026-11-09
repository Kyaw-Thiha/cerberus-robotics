"""Seeded start-goal sampling on the inflated grid: free-space pairs at a target geodesic distance.

Pure Python. "Geodesic" means the A* path length, not the straight-line distance, so a pair separated by a wall is
rejected rather than counted as a short hop. Rejections are counted by cause -- that count is a finding about the
scene (a house where most random pairs are unreachable is a different place from one where they are not).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from planning.global_planner.core.astar import PlanResult, astar
from planning.occupancy import OccupancyGrid


@dataclass(frozen=True)
class StartGoalPair:
    start: tuple[float, float]
    goal: tuple[float, float]
    shortest_path_m: float


@dataclass
class SamplingStats:
    requested: int = 0
    attempts: int = 0
    rejected_unreachable: int = 0
    rejected_distance: int = 0
    rejected_too_close: int = 0
    free_cells: int = 0
    pairs: list[StartGoalPair] = field(default_factory=list)


def sample_pairs(
    grid: OccupancyGrid,
    *,
    n: int,
    seed: int,
    min_distance_m: float = 3.0,
    max_distance_m: float = 10.0,
    min_separation_m: float = 0.5,
    attempts_per_pair: int = 40,
) -> SamplingStats:
    """Up to ``n`` reachable pairs whose A* path length lies in ``[min_distance_m, max_distance_m]``.

    Deterministic for a given ``(grid, seed, n)``: candidates come from a private ``random.Random(seed)`` and are
    accepted in draw order. Stops early (with a smaller list) rather than looping forever in a tiny scene.
    """
    import numpy as np

    from planning.global_planner.core.astar import traversable_mask

    free = traversable_mask(grid)
    ys, xs = np.nonzero(free)
    stats = SamplingStats(requested=n, free_cells=len(ys))
    if len(ys) < 2:
        return stats

    rng = random.Random(seed)
    while len(stats.pairs) < n and stats.attempts < n * attempts_per_pair:
        stats.attempts += 1
        a = int(rng.randrange(len(ys)))
        b = int(rng.randrange(len(ys)))
        if a == b:
            stats.rejected_too_close += 1
            continue
        start = grid.centre_of(int(ys[a]), int(xs[a]))
        goal = grid.centre_of(int(ys[b]), int(xs[b]))
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < min_separation_m:
            stats.rejected_too_close += 1
            continue
        plan: PlanResult = astar(grid, start, goal, shortcut=True)
        if not plan.success:
            stats.rejected_unreachable += 1
            continue
        if not (min_distance_m <= plan.length_m <= max_distance_m):
            stats.rejected_distance += 1
            continue
        stats.pairs.append(StartGoalPair(start=start, goal=goal, shortest_path_m=plan.length_m))
    return stats
