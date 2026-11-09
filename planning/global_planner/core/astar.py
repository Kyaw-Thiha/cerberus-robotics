"""A* on the inflated occupancy grid: the real global planner behind the H4 navigation eval.

Pure Python (stdlib + numpy). 8-connected, octile heuristic, no corner cutting, optional greedy shortcutting by
line of sight. The cost is Euclidean distance in CELLS; :class:`PlanResult.length_m` is the world-frame length of
the returned (possibly shortcut) path, which is what SPL uses as the shortest-path reference.

Blocked cells are ``BLOCKED`` plus, by default, everything inside the robot-radius inflation. ``UNKNOWN`` counts as
blocked unless ``allow_unknown=True`` -- planning through unmapped space is a choice, never a default.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from planning.occupancy import BLOCKED, FREE, OccupancyGrid

SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
)


@dataclass
class PlanResult:
    success: bool
    path: list[tuple[float, float]] = field(default_factory=list)
    cells: list[tuple[int, int]] = field(default_factory=list)
    length_m: float = 0.0
    expanded: int = 0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.success


def traversable_mask(grid: OccupancyGrid, *, inflated: bool = True, allow_unknown: bool = False):
    """Boolean (ny, nx): cells the planner may enter."""

    ok = grid.grid == FREE
    if inflated and grid.inflated is not None:
        ok &= ~grid.inflated
    if allow_unknown:
        ok |= grid.grid != BLOCKED
    return ok


def astar(
    grid: OccupancyGrid,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    inflated: bool = True,
    allow_unknown: bool = False,
    shortcut: bool = True,
) -> PlanResult:
    """Shortest 8-connected path from ``start`` to ``goal`` (world metres) on ``grid``."""
    import numpy as np

    free = traversable_mask(grid, inflated=inflated, allow_unknown=allow_unknown)
    ny, nx = grid.shape
    start_cell = grid.cell_of(*start)
    goal_cell = grid.cell_of(*goal)
    for name, (iy, ix) in (("start", start_cell), ("goal", goal_cell)):
        if not (0 <= iy < ny and 0 <= ix < nx):
            return PlanResult(False, reason=f"{name}_outside_grid")
        if not free[iy, ix]:
            return PlanResult(False, reason=f"{name}_blocked")

    def index(iy: int, ix: int) -> int:
        return iy * nx + ix

    def heur(iy: int, ix: int) -> float:
        dy, dx = abs(iy - goal_cell[0]), abs(ix - goal_cell[1])
        return max(dy, dx) + (SQRT2 - 1.0) * min(dy, dx)

    start_i, goal_i = index(*start_cell), index(*goal_cell)
    g = {start_i: 0.0}
    came: dict[int, int] = {}
    open_heap = [(heur(*start_cell), 0.0, start_i)]
    closed = np.zeros(ny * nx, dtype=bool)
    expanded = 0

    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if closed[current]:
            continue
        closed[current] = True
        expanded += 1
        if current == goal_i:
            break
        cy, cx = divmod(current, nx)
        for dy, dx, step in _NEIGHBOURS:
            y, x = cy + dy, cx + dx
            if not (0 <= y < ny and 0 <= x < nx) or not free[y, x]:
                continue
            if dy and dx and not (free[cy + dy, cx] and free[cy, cx + dx]):
                continue  # no corner cutting through a blocked diagonal
            n = index(y, x)
            if closed[n]:
                continue
            tentative = cost + step
            if tentative < g.get(n, math.inf):
                g[n] = tentative
                came[n] = current
                heapq.heappush(open_heap, (tentative + heur(y, x), tentative, n))

    if goal_i not in g or not closed[goal_i]:
        return PlanResult(False, expanded=expanded, reason="unreachable")

    cells = [divmod(goal_i, nx)]
    while cells[-1] != divmod(start_i, nx):
        cells.append(divmod(came[index(*cells[-1])], nx))
    cells.reverse()

    if shortcut:
        cells = shortcut_cells(free, cells)

    path = [grid.centre_of(iy, ix) for iy, ix in cells]
    path[0] = (float(start[0]), float(start[1]))
    path[-1] = (float(goal[0]), float(goal[1]))
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(path))
    return PlanResult(True, path=path, cells=cells, length_m=length, expanded=expanded)


def line_of_sight(free, a: tuple[int, int], b: tuple[int, int], step: float = 0.25) -> bool:
    """True when every cell crossed by the segment ``a``-``b`` is traversable (sampled at ``step`` cells)."""
    import numpy as np

    ny, nx = free.shape
    (ay, ax), (by, bx) = a, b
    n = int(max(abs(by - ay), abs(bx - ax)) / step) + 1
    for t in np.linspace(0.0, 1.0, n):
        y = round(ay + (by - ay) * t)
        x = round(ax + (bx - ax) * t)
        if not (0 <= y < ny and 0 <= x < nx) or not free[y, x]:
            return False
    return True


def shortcut_cells(free, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy string pulling: from each kept cell, jump to the farthest cell still in line of sight."""
    if len(cells) <= 2:
        return list(cells)
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_of_sight(free, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def path_is_collision_free(
    grid: OccupancyGrid,
    path: list[tuple[float, float]],
    *,
    inflated: bool = True,
    allow_unknown: bool = False,
    samples_per_cell: int = 4,
) -> bool:
    """True when every point along every segment of ``path`` is traversable.

    The check the eval uses on a planned path AND on the pose trace the robot actually walked: a controller that
    "succeeds" through a wall must be caught here, not inferred from the success flag.
    """
    import numpy as np

    if len(path) < 2:
        return False
    free = traversable_mask(grid, inflated=inflated, allow_unknown=allow_unknown)
    ny, nx = free.shape
    res = grid.resolution_m
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        steps = int(max(abs(x1 - x0), abs(y1 - y0)) / res * samples_per_cell) + 1
        for t in np.linspace(0.0, 1.0, steps):
            iy, ix = grid.cell_of(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if not (0 <= iy < ny and 0 <= ix < nx) or not free[iy, ix]:
                return False
    return True
