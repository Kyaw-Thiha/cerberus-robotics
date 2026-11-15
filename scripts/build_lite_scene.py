#!/usr/bin/env python3
"""Flatten a GRScenes home into one collision-only USD (the "lite" scene).

Drops materials, textures and non-geometry prims, resolves references into world-space triangles (metres, Z-up), merges
them into a single Mesh and puts UsdPhysics collision on it. The lite file is what the next GPU session fetches, so the
geometry must match the original: the sample occupancy grids are compared with IoU and the run fails below 0.98.

    python3 scripts/build_lite_scene.py --scene-usd data/scenes/grscenes/home/scenes/<id>_usd/start_result_*.usd \
        --out-dir data/scenes/grscenes_lite/<id> --scene-id <id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from scene_id import folder_name  # noqa: E402


def grid_iou(a, b) -> float:
    """IoU of the blocked cells of two grids (same resolution/extent required)."""
    import numpy as np

    from planning.occupancy import BLOCKED

    blocked_a = np.asarray(a.grid) == BLOCKED
    blocked_b = np.asarray(b.grid) == BLOCKED
    if blocked_a.shape != blocked_b.shape:
        return 0.0
    union = np.logical_or(blocked_a, blocked_b).sum()
    return 1.0 if union == 0 else float(np.logical_and(blocked_a, blocked_b).sum()) / float(union)


def add_collision(usd_path: Path, mesh_path: str) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim:
        raise RuntimeError(f"{mesh_path} missing in {usd_path}")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.GetAttribute("physics:approximation").Set("none")
    stage.GetRootLayer().Save()


def build(scene_usd: Path, out_dir: Path, scene_id: str, budget_s: float = 1800.0) -> dict:

    from common.cerberus_worlds.scene_geometry import write_triangle_mesh_usd
    from planning.occupancy import from_usd, scene_triangles_m, save_npz

    out_dir.mkdir(parents=True, exist_ok=True)
    sid = folder_name(scene_id)
    points, faces = scene_triangles_m(scene_usd, budget_s=budget_s)
    lite = out_dir / f"{sid}.usd"
    write_triangle_mesh_usd(points, faces, lite, meters_per_unit=1.0,
                            root_prim="/World", mesh_prim="/World/CollisionMesh")
    add_collision(lite, "/World/CollisionMesh")

    # Same grid from both files; the lite file must reproduce the original's blocked cells.
    cfg_path = _REPO / "planning" / "config" / "occupancy.yaml"
    from planning.occupancy import OccupancyConfig

    try:
        config = OccupancyConfig.from_yaml(cfg_path) if cfg_path.is_file() else OccupancyConfig()
    except ImportError:  # PyYAML is optional on a bare CPU image; the defaults are the same policy
        config = OccupancyConfig()
    grid_orig = from_usd(scene_usd, scene_id=sid, config=config)
    grid_lite = from_usd(lite, scene_id=sid, config=config)
    iou = grid_iou(grid_orig, grid_lite)
    save_npz(grid_lite, out_dir / "occupancy.npz")

    report = {"scene_id": sid, "lite_file": lite.name, "lite_bytes": lite.stat().st_size,
              "triangles": len(faces), "vertices": len(points), "iou_blocked": round(iou, 6),
              "grid_shape": list(grid_orig.grid.shape), "source_bytes": scene_usd.stat().st_size}
    (out_dir / "lite_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        f"# Lite scene `{sid}`\n\n"
        "Built by `scripts/build_lite_scene.py` from the preprocessed GRScenes home. This USD carries **static\n"
        "collision geometry only**: every mesh under the source stage (floors, walls, furniture) was resolved into\n"
        "world-space triangles in metres, merged into a single `Mesh` and given `UsdPhysics` collision with\n"
        "`approximation = none`.\n\n"
        "Dropped: all materials and textures, shader and look layers, lights, cameras, articulations/joints,\n"
        "non-mesh prims and the source hierarchy.\n\n"
        f"Triangles {report['triangles']}, vertices {report['vertices']}, size {report['lite_bytes']} bytes,\n"
        f"blocked-cell IoU against the original occupancy grid {report['iou_blocked']}.\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--max-mb", type=float, default=100.0)
    parser.add_argument("--min-iou", type=float, default=0.98)
    args = parser.parse_args(argv)
    report = build(Path(args.scene_usd), Path(args.out_dir), args.scene_id)
    print(json.dumps(report, indent=2))
    ok = report["lite_bytes"] <= args.max_mb * 1e6 and report["iou_blocked"] >= args.min_iou
    print("[lite]", "OK" if ok else "FAILED acceptance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
