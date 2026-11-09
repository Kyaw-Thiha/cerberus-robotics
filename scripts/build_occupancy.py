#!/usr/bin/env python3
"""Build the ground-truth occupancy grid (+ PNG preview) of a GRScenes scene: ``<scene folder>/occupancy.npz``.

Needs only ``pxr`` and numpy (no Kit app, no GPU): locally ``pixi run python scripts/build_occupancy.py ...``, on a
pod either the same or ``isaaclab.sh -p``. The grid is what the H4 planner runs on and must be described as
ground-truth-derived (rasterised from the scene's static meshes), never as a sensor map.

Usage:
    python3 scripts/build_occupancy.py --scene grscenes:<id> [--floor-z 0.0] [--resolution 0.05]
                                       [--robot-radius 0.35] [--out-dir data/reports/occupancy]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.occupancy import (  # noqa: E402
    OccupancyConfig,
    from_usd,
    save_npz,
    save_preview_png,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="grscenes:<folder> (resolved through the plain_usd backend)")
    parser.add_argument("--floor-z", type=float, default=None, help="main floor height, metres (default: estimate)")
    parser.add_argument("--resolution", type=float, default=None, help="cell size, m (default: config)")
    parser.add_argument("--robot-radius", type=float, default=None, help="inflation radius, m (default: config)")
    parser.add_argument("--config", default=None, help="occupancy YAML (default: planning/config/occupancy.yaml)")
    parser.add_argument("--out-dir", default="data/reports/occupancy", help="where the PNG preview goes")
    parser.add_argument("--budget_s", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true", help="resolve paths only; no pxr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pod_guard import require_pod

    require_pod("build_occupancy.py")
    from common.cerberus_worlds import registry

    backend = registry.get_backend("plain_usd")
    info = backend.describe_scene(backend.normalize_scene_ref(args.scene))
    if not info.get("usd_path"):
        print(f"[occupancy] ERROR: {info.get('error') or 'scene has no USD (is it stock?)'}", file=sys.stderr)
        return 2
    usd = Path(info["usd_path"])
    npz = usd.parent / "occupancy.npz"
    png = Path(args.out_dir) / f"{usd.parent.name}_occupancy.png"
    print(f"[occupancy] scene={args.scene} usd={usd} npz={npz} png={png}", flush=True)
    if args.dry_run:
        return 0

    config = OccupancyConfig.from_yaml(args.config) if args.config else OccupancyConfig.from_yaml()
    if args.resolution is not None or args.robot_radius is not None:
        config = OccupancyConfig(**{
            **config.to_dict(),
            **({"resolution_m": args.resolution} if args.resolution else {}),
            **({"robot_radius_m": args.robot_radius} if args.robot_radius else {}),
        })
    grid = from_usd(usd, scene_id=args.scene, config=config, floor_z=args.floor_z)
    save_npz(grid, npz)
    save_preview_png(grid, png, title=f"{args.scene}: free {grid.free_fraction():.0%}")
    summary = {
        "scene": args.scene,
        "npz": str(npz),
        "png": str(png),
        "shape": list(grid.grid.shape),
        "resolution_m": grid.resolution_m,
        "floor_z": grid.floor_z,
        "free_fraction": grid.free_fraction(),
        "inflated_fraction": float(grid.inflated.mean()) if grid.inflated is not None else None,
        "config": grid.config.to_dict(),
    }
    print("[occupancy] " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
