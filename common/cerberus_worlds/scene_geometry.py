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
