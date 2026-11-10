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


REASON_COLOURS = {
    0: (1.0, 1.0, 1.0),    # free
    1: (0.65, 0.65, 0.65), # unknown
    2: (0.85, 0.33, 0.10), # vertical surface in band (walls, legs)
    3: (0.55, 0.15, 0.65), # horizontal surface in band (step/kerb)
    4: (0.95, 0.75, 0.10), # slanted surface
    5: (0.10, 0.55, 0.85), # overhang
    6: (0.15, 0.55, 0.15), # floor-height step rule
    7: (0.35, 0.35, 0.35), # thickening of a thin surface
}


def save_reason_png(grid: OccupancyGrid, out: str | Path, *, title: str = "", dpi: int = 110) -> Path:
    """Colour every cell by WHY it is blocked (``build_grid(collect_diagnostics=True)``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    diag = grid.diagnostics
    if not diag or "category" not in diag:
        raise ValueError("grid has no diagnostics; build it with collect_diagnostics=True")
    category = diag["category"]
    rgb = np.ones((*category.shape, 3), dtype=float)
    for reason, colour in REASON_COLOURS.items():
        rgb[category == reason] = colour
    fig, ax = plt.subplots(figsize=(7, 7 * category.shape[0] / max(category.shape[1], 1)), dpi=dpi)
    ax.imshow(rgb, origin="lower", extent=_extent(grid))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{grid.scene_id}: blocked-cell reasons")
    handles = [Patch(facecolor=REASON_COLOURS[r], label=f"{r}: {name}") for r, name in
               zip(sorted(REASON_COLOURS), _reason_names())]
    ax.legend(handles=handles, loc="upper right", fontsize=5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return Path(out)


def _reason_names():
    from planning.occupancy import REASON_NAMES

    return [REASON_NAMES[r] for r in sorted(REASON_COLOURS)]


def save_up_histogram_png(diag: dict, out: str | Path, *, title: str = "", dpi: int = 110) -> Path:
    """Up-facing surface height above the local floor, all cells vs blocked cells."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.array(diag["hist_bin_edges_m"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=dpi)
    ax.plot(centres, diag["hist_blocked"], label="blocked cells", color="tab:red")
    ax.plot(centres, diag["hist_all"], label="all cells", color="tab:blue", alpha=0.6)
    ax.set_xlabel("up-facing surface height above the local floor (m)")
    ax.set_ylabel("cells")
    ax.set_title(title or "up-facing surfaces above the local floor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return Path(out)
