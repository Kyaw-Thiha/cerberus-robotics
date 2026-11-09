"""World-frame triangles of a USD scene, for anything that needs one clean triangle mesh of it (the admission spawn
sampler, the SLAM LiDAR proxy mesh).

``pxr`` / ``numpy`` are imported inside the functions, so this module imports anywhere. ``pxr`` alone (``from pxr import
Usd``) is enough: no Kit app is needed, Isaac's ``python.sh`` or ``usd-core`` both work.

Why it skips meshes instead of failing: a GRScenes house has thousands of Mesh prims whose ``points`` /
``faceVertexCounts`` are empty (``Get()`` returns None, not time-sampled). Isaac Lab's MultiMeshRayCaster raises
``TypeError: iteration over a 0-d array`` on the first one, so the SLAM eval casts against a proxy built from the valid
geometry only (``scripts/build_lidar_proxy.py``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

METRIC_SCENE_NAME = "scene_metric.usd"
UNITS_SIDECAR_NAME = "scene_units.json"
"""Records what :func:`metric_scene_usd` found for a scene (source mtime, units, wrapper or None), so callers that must
not import ``pxr`` (see ``arena_assets.sim_usd_path``) can read the answer from plain JSON."""
"""Metre-scale wrapper USD written next to a scene whose own units are not metres (see :func:`metric_scene_usd`)."""

PROXY_NAME = "lidar_proxy.usd"
PROXY_SIDECAR_NAME = "lidar_proxy.json"
PROXY_ROOT_PRIM = "/LidarProxy"
"""Root Xform of the proxy USD, which is also its default prim.

It is deliberately an Xform with the mesh BELOW it (``PROXY_MESH_PRIM``) rather than a bare Mesh: Isaac Lab's
multi-mesh ray caster searches the target prim AND its descendants for ``Mesh`` prims, so a wrapper keeps the
target path (``/World/lidar_proxy``) resolvable and the geometry findable no matter how the spawner maps the
file's default prim onto the requested path.
"""

PROXY_MESH_PRIM = "/LidarProxy/Proxy"
"""The single Mesh prim holding the proxy geometry (a child of :data:`PROXY_ROOT_PRIM`)."""

SCAN_PROXY_NAME = "scan_proxy.usd"
"""Proxy for the locomotion policy's height scanner: same idea as the LiDAR proxy (one clean triangle mesh), but
with the ceiling/roof slabs removed so a downward ray from 20 m above the base hits the floor, walls and furniture
instead of the ceiling. Built by ``scripts/build_scan_proxy.py``; consumed by the H4 scene env."""

SCAN_PROXY_SIDECAR_NAME = "scan_proxy.json"
SCAN_PROXY_ROOT_PRIM = "/ScanProxy"
SCAN_PROXY_MESH_PRIM = "/ScanProxy/Proxy"
SCAN_PROXY_SCHEMA_VERSION = 1




def metric_scene_usd(source: str | Path, *, force: bool = False) -> Path:
    """The USD a simulation should load for the scene at `source`, in METRES.

    Isaac Lab spawns a scene USD as a plain reference into a stage that is 1 m per unit, and USD composition never
    converts units. GRScenes houses are authored in CENTIMETRES (their ``metersPerUnit`` is not even authored, so USD
    reports its 0.01 fallback), so spawned as-is a 2.4 m couch is 241 m long and a 13 m house is 1.3 km wide. That was
    measured on a pod (2026-10-06: couch 241 x 97 x 94 units in a 1.0 m/unit live stage, ``/World/ground`` scale 1)
    and it also made the robot a tiny ant, every spawn pose and clearance threshold wrong, and PhysX complain
    ``TriangleMesh: triangles are too big``.

    If the source stage is already 1 m per unit this returns `source` unchanged. Otherwise it writes (or reuses)
    ``scene_metric.usd`` beside it: an ``Xform`` ``/Scene`` scaled by ``metersPerUnit`` that references the original
    (relative path, so its own relative model/material references keep resolving), declared 1 m per unit, same up
    axis. Needs only ``pxr`` (no Kit app).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom

    source = Path(source)
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadNone)
    if not stage:
        raise RuntimeError(f"cannot open {source}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(mpu - 1.0) < 1e-9:
        _write_units_sidecar(source, mpu, None)
        return source
    up_axis = UsdGeom.GetStageUpAxis(stage)
    out = source.with_name(METRIC_SCENE_NAME)
    stamp = {"source": source.name, "source_mtime_ns": source.stat().st_mtime_ns, "meters_per_unit": mpu}
    if not force and out.is_file():
        layer = Sdf.Layer.FindOrOpen(str(out))
        if layer is not None and dict(layer.customLayerData) == stamp:
            _write_units_sidecar(source, mpu, out.name)
            return out
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    wrapper = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(wrapper, up_axis)
    UsdGeom.SetStageMetersPerUnit(wrapper, 1.0)
    scene = UsdGeom.Xform.Define(wrapper, "/Scene")
    scene.AddScaleOp().Set(Gf.Vec3d(mpu, mpu, mpu))
    wrapper.DefinePrim("/Scene/Source").GetReferences().AddReference(f"./{source.name}")
    wrapper.SetDefaultPrim(scene.GetPrim())
    wrapper.GetRootLayer().customLayerData = stamp
    wrapper.GetRootLayer().Save()
    del wrapper
    tmp.replace(out)
    _write_units_sidecar(source, mpu, out.name)
    return out


def _write_units_sidecar(source: Path, mpu: float, wrapper_name: str | None) -> None:
    import json

    sidecar = source.with_name(UNITS_SIDECAR_NAME)
    sidecar.write_text(
        json.dumps(
            {
                "source": source.name,
                "source_mtime_ns": source.stat().st_mtime_ns,
                "meters_per_unit": mpu,
                "wrapper": wrapper_name,
            },
            indent=2,
            sort_keys=True,
        )
    )


def cached_metric_scene_usd(source: str | Path) -> Path | None:
    """The answer :func:`metric_scene_usd` gave before for `source`, read from the JSON sidecar WITHOUT importing
    ``pxr``; None if there is no up-to-date sidecar (the source changed, or it was never computed)."""
    import json

    source = Path(source)
    sidecar = source.with_name(UNITS_SIDECAR_NAME)
    try:
        meta = json.loads(sidecar.read_text())
        mtime = source.stat().st_mtime_ns
    except (OSError, ValueError):
        return None
    if meta.get("source") != source.name or meta.get("source_mtime_ns") != mtime:
        return None
    wrapper = meta.get("wrapper")
    if wrapper is None:
        return source
    out = source.with_name(wrapper)
    return out if out.is_file() else None


PXR_MISSING_EXIT = 3


def main(argv: list[str] | None = None) -> int:
    """``python -m common.cerberus_worlds.scene_geometry <scene.usd> [--force]``: print the metre-scale USD path.

    Exists so a process that must not import ``pxr`` yet (anything before the Isaac app launches: importing it first
    breaks ``omni.usd``) can still get the wrapper built, by running this in a child interpreter. Exit 3 means ``pxr``
    is not importable in the child (nothing could be spawned there anyway).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("usd")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        import pxr  # noqa: F401
    except ImportError:
        return PXR_MISSING_EXIT
    try:
        print(metric_scene_usd(args.usd, force=args.force))
    except Exception as e:  # noqa: BLE001 - loud, nonzero, with the cause
        print(f"scene_geometry: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def lidar_proxy_path(scene_usd: str | Path) -> Path:
    """``<scene folder>/lidar_proxy.usd`` for a scene USD path (what ``scripts/build_lidar_proxy.py`` writes)."""
    return Path(scene_usd).parent / PROXY_NAME


def triangulate_faces(counts, indices):
    """Fan-triangulate a USD polygon mesh (``faceVertexCounts`` / ``faceVertexIndices``) -> (M, 3) int32.

    Same as ``scripts/scene_spawn_sampling.triangulate_faces`` (kept separate because that module is a flat script
    module and not importable from ``common``; dedupe by importing this one there).
    """
    import numpy as np

    counts = np.asarray(counts)
    indices = np.asarray(indices)
    if (counts == 3).all():
        return indices.reshape(-1, 3).astype(np.int32)
    starts = np.cumsum(counts) - counts
    tris = []
    for k in np.unique(counts):
        if k < 3:
            continue
        s = starts[counts == k]
        for j in range(k - 2):
            tris.append(np.stack([indices[s], indices[s + 1 + j], indices[s + 2 + j]], axis=1))
    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(tris).astype(np.int32)


def gather_world_triangles(
    usd_stage: Any,
    root_path: str,
    skip_names: frozenset[str],
    budget_s: float,
    stats: dict[str, int] | None = None,
):
    """World-frame triangles of every Mesh prim under ``root_path`` (instance proxies included, subtrees named in
    ``skip_names`` pruned): ``(points (N, 3) float32, triangles (M, 3) int32)``.

    Meshes with no points, no faces or no indices are skipped. If ``stats`` is given it is filled with
    ``meshes_used`` and ``meshes_skipped_empty`` (also counting meshes whose faces are all degenerate).

    Raises:
        RuntimeError: the root is missing, or no mesh with geometry exists under it.
        TimeoutError: gathering took longer than ``budget_s``.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    start = usd_stage.GetPseudoRoot() if root_path == "/" else usd_stage.GetPrimAtPath(root_path)
    if not start or not start.IsValid():
        raise RuntimeError(f"scene root {root_path!r} not found on the stage")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    t0 = time.perf_counter()
    points, faces, offset = [], [], 0
    used = skipped = 0
    prims = iter(Usd.PrimRange(start, Usd.TraverseInstanceProxies()))  # UNVERIFIED: instance proxies of GRScenes objects
    for prim in prims:
        if prim.GetName() in skip_names:
            prims.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if time.perf_counter() - t0 > budget_s:
            raise TimeoutError(f"gathering scene triangles exceeded {budget_s:.0f} s ({offset} vertices so far)")
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = (a.Get() for a in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()))
        if pts is None or counts is None or idx is None or len(pts) == 0 or len(counts) == 0:
            skipped += 1
            continue
        m = np.array(xforms.GetLocalToWorldTransform(prim)).reshape(4, 4)  # Gf row-vector convention: p' = p M
        world = np.asarray(pts, dtype=np.float64) @ m[:3, :3] + m[3, :3]
        tris = triangulate_faces(np.asarray(counts), np.asarray(idx))
        if len(tris) == 0:
            skipped += 1
            continue
        # winding is counter-clockwise-outward for rightHanded meshes; leftHanded or a mirroring transform flips it
        if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != (np.linalg.det(m[:3, :3]) < 0):
            tris = tris[:, [0, 2, 1]]
        points.append(world.astype(np.float32))
        faces.append(tris + offset)
        offset += len(pts)
        used += 1
    if stats is not None:
        stats.update(meshes_used=used, meshes_skipped_empty=skipped)
    if not points:
        raise RuntimeError(f"no Mesh prims with geometry under {root_path!r} ({skipped} empty)")
    return np.concatenate(points), np.concatenate(faces)


# ---------------------------------------------------------------------------------------------------------
# triangle-mesh writer and the height-scan proxy
# ---------------------------------------------------------------------------------------------------------
def write_triangle_mesh_usd(
    points: Any,
    triangles: Any,
    out: str | Path,
    *,
    meters_per_unit: float = 1.0,
    root_prim: str,
    mesh_prim: str,
) -> Path:
    """Write one static triangle Mesh under ``root_prim`` (default prim), nothing else: no physics or material.

    ``root_prim`` is an Xform with the Mesh below it, for the reason documented on :data:`PROXY_ROOT_PRIM`: the
    ray caster searches the target prim and its descendants.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, Vt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.partial{out.suffix}")
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
    root = UsdGeom.Xform.Define(stage, root_prim)
    mesh = UsdGeom.Mesh.Define(stage, mesh_prim)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    del stage
    tmp.replace(out)
    return out


def scan_proxy_path(scene_usd: str | Path) -> Path:
    """``<scene folder>/scan_proxy.usd`` for a scene USD path (what ``scripts/build_scan_proxy.py`` writes)."""
    return Path(scene_usd).parent / SCAN_PROXY_NAME


def filter_scan_triangles(
    points: Any,
    faces: Any,
    *,
    floor_z: float,
    cutoff_m: float = 0.6,
    ceiling_min_area_m2: float = 3.0,
    ceiling_min_height_m: float = 1.5,
):
    """Drop ceiling/roof slabs so a DOWNWARD height-scan ray hits the floor, walls and furniture.

    A vertical ray only registers a wall through the wall's top face, so walls must keep theirs; that makes "drop
    everything above a height" wrong (it would make walls invisible and read the floor through them). Instead this
    drops only what a ceiling is: a HORIZONTAL triangle, at least ``ceiling_min_area_m2`` in area, entirely above
    ``floor_z + ceiling_min_height_m``. Furniture tops and wall-top strips are small or low and stay.

    Multi-level scenes: an upper floor slab is a large horizontal plane high up and is dropped by the same rule, so
    the scan sees through to the level below. That matches the single-floor assumption of the occupancy grid
    (documented in the design note); it is not a claim about multi-floor navigation.

    Points must be in METRES. Returns compacted ``(points, faces)``.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("scan proxy filter needs at least one triangle")
    tri = points[faces]  # (M, 3, 3)
    min_z = tri[:, :, 2].min(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    area = 0.5 * norm
    with np.errstate(invalid="ignore", divide="ignore"):
        horizontal = np.abs(normal[:, 2]) >= 0.9 * norm
    is_ceiling = horizontal & (area >= ceiling_min_area_m2) & (min_z > floor_z + ceiling_min_height_m)
    kept = faces[~is_ceiling]
    if len(kept) == 0:
        raise ValueError("every triangle was classified as a ceiling; check floor_z/cutoff")
    used, inverse = np.unique(kept, return_inverse=True)
    return points[used], inverse.reshape(-1, 3).astype(np.int32)


def build_scan_proxy(
    source: str | Path,
    out: str | Path,
    *,
    floor_z: float,
    cutoff_m: float = 0.6,
    ceiling_min_area_m2: float = 3.0,
    ceiling_min_height_m: float = 1.5,
    budget_s: float = 1800.0,
) -> dict[str, Any]:
    """Gather ``source``'s world triangles, convert to metres, filter ceilings, write the scan proxy + sidecar."""
    import json

    import numpy as np
    from pxr import Usd, UsdGeom

    source, out = Path(source), Path(out)
    t0 = time.perf_counter()
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise RuntimeError(f"cannot open {source}")
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise ValueError(f"scene stage is up_axis={UsdGeom.GetStageUpAxis(stage)!r}; the scan proxy needs Z-up")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    stats: dict[str, int] = {}
    points, triangles = gather_world_triangles(stage, "/", frozenset(), budget_s, stats)
    points = points.astype(np.float64) * mpu
    filtered_points, filtered_faces = filter_scan_triangles(
        points,
        triangles,
        floor_z=floor_z,
        cutoff_m=cutoff_m,
        ceiling_min_area_m2=ceiling_min_area_m2,
        ceiling_min_height_m=ceiling_min_height_m,
    )
    write_triangle_mesh_usd(
        filtered_points,
        filtered_faces,
        out,
        meters_per_unit=1.0,
        root_prim=SCAN_PROXY_ROOT_PRIM,
        mesh_prim=SCAN_PROXY_MESH_PRIM,
    )
    meta = {
        "schema_version": SCAN_PROXY_SCHEMA_VERSION,
        "source": str(source),
        "source_mtime_ns": source.stat().st_mtime_ns,
        "source_meters_per_unit": mpu,
        "floor_z": float(floor_z),
        "cutoff_m": float(cutoff_m),
        "ceiling_min_area_m2": float(ceiling_min_area_m2),
        "ceiling_min_height_m": float(ceiling_min_height_m),
        "triangles_in": len(triangles),
        "triangles_out": len(filtered_faces),
        "vertices_out": len(filtered_points),
        "bounds_min": [round(float(v), 3) for v in filtered_points.min(axis=0)],
        "bounds_max": [round(float(v), 3) for v in filtered_points.max(axis=0)],
        "build_s": round(time.perf_counter() - t0, 1),
        **stats,
    }
    out.with_name(SCAN_PROXY_SIDECAR_NAME).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta
