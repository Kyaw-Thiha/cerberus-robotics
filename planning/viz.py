"""Path-planning visualisations: grid, inflated zone, sampled paths, all as small PNGs.

Pure Python + matplotlib (Agg). Used by ``scripts/plan_preview.py`` to commit a few previews with the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from planning.occupancy import BLOCKED, FREE, UNKNOWN, OccupancyGrid

FREE_RGB = (1.0, 1.0, 1.0)
BLOCKED_RGB = (0.1, 0.1, 0.1)
UNKNOWN_RGB = (0.65, 0.65, 0.65)
INFLATED_RGB = (0.95, 0.78, 0.78)


def grid_rgb(grid: OccupancyGrid, *, show_inflation: bool = True) -> np.ndarray:
    rgb = np.zeros((*grid.grid.shape, 3), dtype=float)
    rgb[grid.grid == FREE] = FREE_RGB
    rgb[grid.grid == BLOCKED] = BLOCKED_RGB
    rgb[grid.grid == UNKNOWN] = UNKNOWN_RGB
    if show_inflation and grid.inflated is not None:
        mask = grid.inflated & (grid.grid == FREE)
        rgb[mask] = INFLATED_RGB
    return rgb


def _extent(grid: OccupancyGrid):
    x0, y0, x1, y1 = grid.extent()
    return (x0, x1, y0, y1)


def save_plan_png(
    grid: OccupancyGrid,
    out: str | Path,
    *,
    paths: Sequence[Sequence[tuple[float, float]]] = (),
    start: tuple[float, float] | None = None,
    goal: tuple[float, float] | None = None,
    title: str = "",
    show_inflation: bool = True,
    dpi: int = 110,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7 * grid.shape[0] / max(grid.shape[1], 1)), dpi=dpi)
    ax.imshow(grid_rgb(grid, show_inflation=show_inflation), origin="lower", extent=_extent(grid))
    for i, path in enumerate(paths):
        if not path:
            continue
        ax.plot([p[0] for p in path], [p[1] for p in path], linewidth=1.4, label=f"path {i}" if len(paths) < 8 else None)
    if start is not None:
        ax.plot(*start, "o", color="tab:green", markersize=6, label="start")
    if goal is not None:
        ax.plot(*goal, "*", color="tab:blue", markersize=10, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{grid.scene_id}: free {grid.free_fraction():.0%}, red = inflated")
    if start is not None or goal is not None or paths:
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def save_grid_png(grid: OccupancyGrid, out: str | Path, *, title: str = "", dpi: int = 110) -> Path:
    return save_plan_png(grid, out, title=title, dpi=dpi)


def summarise_pairs(pairs: Iterable) -> dict:
    pairs = list(pairs)
    lengths = [p.shortest_path_m for p in pairs]
    return {
        "pairs": len(pairs),
        "mean_shortest_path_m": (sum(lengths) / len(lengths)) if lengths else None,
        "min_shortest_path_m": min(lengths) if lengths else None,
        "max_shortest_path_m": max(lengths) if lengths else None,
    }
