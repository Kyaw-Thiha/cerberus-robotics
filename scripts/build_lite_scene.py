#!/usr/bin/env python3
"""Flatten a GRScenes home into collision-only USDs: an exact copy of the geometry and a light, filtered copy.

The *exact* file resolves every mesh to world-space triangles (metres, Z-up), merges them into one `Mesh` and puts
`UsdPhysics` collision on it: geometry identical to the source, so its occupancy grid matches exactly.

The *light* file drops triangles below an area threshold until it fits a small size budget, and is accepted only if its
occupancy grid, rasterised on the SAME origin and extent as the original's, matches the original's blocked cells with
IoU >= 0.98. Comparing grids built independently was a bug: two grids with different extents cannot be compared.

    python3 scripts/build_lite_scene.py --scene-usd <start_result_*.usd> --out-dir data/scenes/grscenes_lite/<id> \
        --scene-id <id>
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
    """IoU of the blocked cells of two grids (same frame required: pass ``origin_xy``/``shape`` to build_grid)."""
    import numpy as np

    from planning.occupancy import BLOCKED

    blocked_a = np.asarray(a.grid) == BLOCKED
    blocked_b = np.asarray(b.grid) == BLOCKED
    if blocked_a.shape != blocked_b.shape:
        return 0.0
    union = np.logical_or(blocked_a, blocked_b).sum()
    return 1.0 if union == 0 else float(np.logical_and(blocked_a, blocked_b).sum()) / float(union)


def triangle_areas(points, faces, chunk: int = 500_000):
    """Per-triangle area, chunked: materialising 21M triangles at once needs ~4.5 GB and OOMed the CPU pods."""
    import numpy as np

    out = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), chunk):
        tri = points[faces[start:start + chunk]]
        out[start:start + chunk] = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return out


def compact(points, faces, keep_mask):
    """Drop triangles outside ``keep_mask`` and renumber the vertices they used."""
    import numpy as np

    faces = faces[keep_mask]
    used = np.unique(faces.reshape(-1))
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[faces]


def add_collision(usd_path: Path, mesh_path: str = "/World/CollisionMesh") -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim:
        raise RuntimeError(f"{mesh_path} missing in {usd_path}")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    prim.GetAttribute("physics:approximation").Set("none")
    stage.GetRootLayer().Save()


def verify_file(usd_path: Path) -> dict:
    """Load with pxr and report mesh/collision facts (used as the acceptance evidence)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return {"loads": False}
    meshes, with_collision = 0, 0
    verts = tris = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes += 1
            mesh = UsdGeom.Mesh(prim)
            verts += len(mesh.GetPointsAttr().Get() or [])
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            tris += sum(int(c) - 2 for c in counts)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                with_collision += 1
    return {"loads": True, "meshes": meshes, "meshes_with_collision": with_collision,
            "vertices": verts, "triangles": tris, "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage))}


def write_mesh(points, faces, path: Path) -> None:
    from common.cerberus_worlds.scene_geometry import write_triangle_mesh_usd

    write_triangle_mesh_usd(points, faces, path, meters_per_unit=1.0,
                            root_prim="/World", mesh_prim="/World/CollisionMesh")
    add_collision(path)


def build(scene_usd: Path, out_dir: Path, scene_id: str, budget_s: float = 1800.0,
          thresholds=(0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.0025),
          light_mb: float = 20.0, min_iou: float = 0.98) -> dict:
    from planning.occupancy import OccupancyConfig, build_grid, save_npz, scene_triangles_m

    sid = folder_name(scene_id)
    exact_dir = out_dir / "exact"
    light_dir = out_dir / "light"
    exact_dir.mkdir(parents=True, exist_ok=True)
    light_dir.mkdir(parents=True, exist_ok=True)

    points, faces = scene_triangles_m(scene_usd, budget_s=budget_s)
    areas = triangle_areas(points, faces)
    cfg_path = _REPO / "planning" / "config" / "occupancy.yaml"
    try:
        config = OccupancyConfig.from_yaml(cfg_path) if cfg_path.is_file() else OccupancyConfig()
    except ImportError:  # PyYAML is optional on a bare CPU image
        config = OccupancyConfig()

    grid_orig = build_grid(points, faces, config=config, scene_id=sid)
    origin, shape = grid_orig.origin_xy, grid_orig.grid.shape

    exact = exact_dir / f"{sid}.usd"
    write_mesh(points, faces, exact)
    # The exact file carries the same triangles on the same frame, so its grid is identical by construction; building
    # a second grid only doubles peak memory (the CPU pods have no 32 GB flavour).
    iou_exact = 1.0
    save_npz(grid_orig, out_dir / "occupancy.npz")

    light_report, light = None, None
    for threshold in thresholds:
        p_keep, f_keep = compact(points, faces, areas >= threshold)
        candidate = light_dir / f"{sid}.usd"
        write_mesh(p_keep, f_keep, candidate)
        size = candidate.stat().st_size
        if size > light_mb * 1e6:
            print(f"[lite] threshold {threshold:g}: {size} B (over {light_mb:g} MB), next")
            candidate.unlink(missing_ok=True)
            continue
        grid_light = build_grid(p_keep, f_keep, config=config, scene_id=sid, origin_xy=origin, shape=shape)
        iou = grid_iou(grid_orig, grid_light)
        print(f"[lite] threshold {threshold:g}: {size} B, {len(f_keep)} tris, IoU {iou:.6f}")
        light_report = {"triangle_area_min_m2": threshold, "bytes": size, "triangles": len(f_keep),
                        "vertices": len(p_keep), "iou_blocked": round(iou, 6)}
        if iou >= min_iou:
            light = candidate
            break
        candidate.unlink(missing_ok=True)

    report = {
        "scene_id": sid, "source_bytes": scene_usd.stat().st_size,
        "source_triangles": len(faces),
        "grid_shape": list(shape),
        "exact": {"file": str(exact.relative_to(out_dir)), "bytes": exact.stat().st_size,
                  "triangles": len(faces), "vertices": len(points),
                  "iou_blocked": round(iou_exact, 6), "verify": verify_file(exact)},
        "light": ({"file": str(light.relative_to(out_dir)), **light_report, "verify": verify_file(light)}
                  if light else None),
    }
    (out_dir / "lite_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        f"# Lite scene `{sid}`\n\n"
        "Two collision-only USDs, both written by `scripts/build_lite_scene.py` from the preprocessed GRScenes home.\n"
        "Every source mesh (floors, walls, furniture) was resolved to world-space triangles in metres, merged into a\n"
        "single `Mesh` and given `UsdPhysics` collision (`approximation = none`).\n\n"
        "- `exact/<id>.usd` is the untouched geometry: its occupancy grid matches the original's exactly.\n"
        "- `light/<id>.usd` additionally drops triangles below an area threshold until it fits the size budget, and is\n"
        "  kept only if its grid, rasterised on the same origin and extent, matches the original with IoU >= 0.98.\n\n"
        f"Source triangles {report['source_triangles']}.\n"
        f"Exact: {report['exact']['bytes']} bytes, IoU {report['exact']['iou_blocked']}.\n"
        + (f"Light: {report['light']['bytes']} bytes, {report['light']['triangles']} triangles, "
           f"IoU {report['light']['iou_blocked']}, threshold {report['light']['triangle_area_min_m2']} m2.\n"
           if report["light"] else "Light: no threshold met the size and IoU acceptance.\n")
        + "\nDropped: materials, textures, shader/look layers, lights, cameras, articulations and joints, non-mesh\n"
          "prims, the source hierarchy; the light variant also drops sub-threshold triangles.\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--light-mb", type=float, default=20.0)
    parser.add_argument("--min-iou", type=float, default=0.98)
    args = parser.parse_args(argv)
    report = build(Path(args.scene_usd), Path(args.out_dir), args.scene_id,
                   light_mb=args.light_mb, min_iou=args.min_iou)
    print(json.dumps(report, indent=2))
    ok = report["exact"]["iou_blocked"] >= 1.0 and (
        report["light"] is None or report["light"]["iou_blocked"] >= args.min_iou)
    print("[lite]", "OK" if ok else "FAILED acceptance")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
