#!/usr/bin/env python3
"""Local path-planning previews: synthetic house -> occupancy grid -> A* paths -> committed PNGs.

Runs with no Isaac, no GPU and no scene assets. The geometry is a deterministic synthetic flat (outer walls with
doorways, a table, a sofa) built here, because the GRScenes homes are not fetchable on this machine (see
``docs/h4_report.md``); the pipeline it exercises -- ``planning/occupancy.py`` -> ``planning/sampling.py`` ->
``planning/global_planner/core/astar.py`` -> ``planning/viz.py`` -- is the one the pod runs on the real grids.

Usage:
    python3 scripts/plan_preview.py [--out docs/reports/figures] [--seed 0] [--pairs 6]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from planning.global_planner.core.astar import astar, path_is_collision_free  # noqa: E402
from planning.occupancy import OccupancyConfig, build_grid  # noqa: E402
from planning.sampling import sample_pairs  # noqa: E402
from planning.viz import save_grid_png, save_plan_png, summarise_pairs  # noqa: E402


def quad(x0, x1, y0, y1, z):
    return (np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float),
            np.array([[0, 1, 2], [0, 2, 3]]))


def box(x0, x1, y0, y1, z0, z1):
    points = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=float)
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return points, faces


def wall_with_door(x0, x1, y, door_x, door_w=0.9, z=2.6):
    """A wall along x at height ``y`` with a doorway: two boxes."""
    left = box(x0, door_x - door_w / 2, y - 0.05, y + 0.05, 0.0, z)
    right = box(door_x + door_w / 2, x1, y - 0.05, y + 0.05, 0.0, z)
    return left, right


def synthetic_flat():
    parts = [
        quad(-6, 6, -4, 4, 0.0),                       # floor 12 x 8 m
        box(-6, -5.9, -4, 4, 0.0, 2.6),                # west wall
        box(5.9, 6, -4, 4, 0.0, 2.6),                  # east wall
        box(-6, 6, -4, -3.9, 0.0, 2.6),                # south wall
        box(-6, 6, 3.9, 4, 0.0, 2.6),                  # north wall
        *wall_with_door(-6, 6, 0.0, door_x=2.0),       # internal wall with a doorway
        *wall_with_door(-6, 0.0, -1.0, door_x=-2.5),   # a second internal wall (west half)
        quad(-4.5, -2.5, 2.0, 3.2, 0.75),              # table top
        box(-4.5, -4.4, 2.0, 2.1, 0.0, 0.75),          # table leg
        box(-2.6, -2.5, 3.1, 3.2, 0.0, 0.75),          # table leg
        box(3.0, 4.8, 1.5, 2.4, 0.0, 0.45),            # sofa block
        box(1.0, 1.4, -3.6, -2.2, 0.0, 1.2),           # wardrobe
    ]
    points, faces, offset = [], [], 0
    for p, f in parts:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(faces)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/reports/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--min-distance", type=float, default=3.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    config = OccupancyConfig.from_yaml()
    grid = build_grid(*synthetic_flat(), config=config, scene_id="synthetic_flat", floor_z=0.0)
    stats = sample_pairs(grid, n=args.pairs, seed=args.seed,
                         min_distance_m=args.min_distance, max_distance_m=args.max_distance)
    plans = [astar(grid, pair.start, pair.goal) for pair in stats.pairs]
    collisions = [not path_is_collision_free(grid, p.path) for p in plans if p.success]

    save_grid_png(grid, out / "h4_occupancy_synthetic.png",
                  title=f"synthetic flat: free {grid.free_fraction():.0%}, red = inflated by {config.robot_radius_m} m")
    save_plan_png(grid, out / "h4_paths_synthetic.png", paths=[p.path for p in plans if p.success],
                  title=f"A* on {len(plans)} seeded pairs (3-10 m geodesic), red = inflated")
    first = next((p for p in plans if p.success), None)
    if first is not None:
        save_plan_png(grid, out / "h4_path_single_synthetic.png", paths=[first.path],
                      start=first.path[0], goal=first.path[-1],
                      title=f"one A* path, {first.length_m:.1f} m, {first.expanded} cells expanded")

    summary = {
        "scene": "synthetic_flat",
        "grid_shape": list(grid.grid.shape),
        "resolution_m": grid.resolution_m,
        "free_fraction": grid.free_fraction(),
        "inflated_fraction": float(grid.inflated.mean()),
        "sampling": {
            "requested": stats.requested,
            "attempts": stats.attempts,
            "free_cells": stats.free_cells,
            "rejected_unreachable": stats.rejected_unreachable,
            "rejected_distance": stats.rejected_distance,
            "rejected_too_close": stats.rejected_too_close,
            **summarise_pairs(stats.pairs),
        },
        "plans": [
            {"success": p.success, "length_m": p.length_m, "expanded": p.expanded, "reason": p.reason}
            for p in plans
        ],
        "paths_collision_free": all(not c for c in collisions),
        "pngs": sorted(p.name for p in out.glob("h4_*synthetic*.png")),
    }
    (out / "h4_plan_preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = all(p.success for p in plans) and all(not c for c in collisions)
    if not ok:
        print("[plan_preview] FAIL: a path is unreachable or crosses blocked cells", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
