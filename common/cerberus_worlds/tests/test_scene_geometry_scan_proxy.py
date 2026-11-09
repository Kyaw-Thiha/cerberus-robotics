"""Synthetic-USD tests for the height-scan proxy: writer, metre conversion and ceiling removal."""

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, Vt  # noqa: E402

from common.cerberus_worlds.scene_geometry import (  # noqa: E402
    SCAN_PROXY_MESH_PRIM,
    SCAN_PROXY_ROOT_PRIM,
    SCAN_PROXY_SIDECAR_NAME,
    build_scan_proxy,
    filter_scan_triangles,
    scan_proxy_path,
    write_triangle_mesh_usd,
)


def quad(x0, x1, y0, y1, z):
    points = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return points, faces


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box as 8 vertices / 12 triangles."""
    p = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=np.float64)
    faces = np.array([
        [0, 1, 3], [0, 3, 2],  # bottom
        [4, 6, 7], [4, 7, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 2, 6], [0, 6, 4],  # left
        [1, 5, 7], [1, 7, 3],  # right
    ], dtype=np.int32)
    return p, faces


def combine(*meshes):
    points, faces, offset = [], [], 0
    for p, f in meshes:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(faces)


def room_stage(path, *, meters_per_unit=1.0, up_axis="Z"):
    """Floor 4x4, ceiling 4x4 at 2.7 m, a wall box 0.2 x 3 x 2.6, a table top 1x1 at 0.75 m."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/World")
    parts = {
        "Floor": quad(-2, 2, -2, 2, 0.0),
        "Ceiling": quad(-2, 2, -2, 2, 2.7),
        "Wall": box(1.0, 1.2, -1.5, 1.5, 0.0, 2.6),
        "Table": quad(0.0, 1.0, 0.0, 1.0, 0.75),
    }
    for name, (points, faces) in parts.items():
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    del stage
    return path


def room_triangles():
    return combine(
        quad(-2, 2, -2, 2, 0.0),
        quad(-2, 2, -2, 2, 2.7),
        box(1.0, 1.2, -1.5, 1.5, 0.0, 2.6),
        quad(0.0, 1.0, 0.0, 1.0, 0.75),
    )


def test_only_the_ceiling_slab_is_dropped():
    points, faces = room_triangles()
    kept_points, kept_faces = filter_scan_triangles(points, faces, floor_z=0.0)
    assert len(kept_faces) == len(faces) - 2  # exactly the ceiling's two triangles
    tri = kept_points[kept_faces]
    assert np.isclose(tri[:, :, 2].max(), 2.6)  # the wall top survives: a downward ray must register the wall
    assert np.isclose(tri[:, :, 2].min(), 0.0)  # the floor survives


def test_wall_top_survives_although_it_is_high():
    wall = box(0.0, 0.2, 0.0, 1.0, 0.0, 2.6)
    _, kept_faces = filter_scan_triangles(*wall, floor_z=0.0)
    assert len(kept_faces) == len(wall[1])


def test_low_horizontal_slabs_are_never_treated_as_ceilings():
    table = quad(0.0, 4.0, 0.0, 4.0, 0.75)  # 16 m^2, but only 0.75 m up
    _, kept_faces = filter_scan_triangles(*table, floor_z=0.0)
    assert len(kept_faces) == 2


def test_filter_requires_triangles():
    with pytest.raises(ValueError):
        filter_scan_triangles(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int32), floor_z=0.0)


def test_build_scan_proxy_converts_centimetres_to_metres(tmp_path):
    source = room_stage(tmp_path / "scene.usd", meters_per_unit=0.01)
    stage = Usd.Stage.Open(str(source))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) * 100.0
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    stage.GetRootLayer().Save()
    del stage

    out = scan_proxy_path(source)
    meta = build_scan_proxy(source, out, floor_z=0.0)
    assert meta["source_meters_per_unit"] == pytest.approx(0.01)
    assert meta["triangles_out"] == meta["triangles_in"] - 2
    assert meta["bounds_max"][0] == pytest.approx(2.0, abs=1e-3)  # metres, not centimetres
    assert out.with_name(SCAN_PROXY_SIDECAR_NAME).is_file()

    written = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageMetersPerUnit(written) == pytest.approx(1.0)
    assert written.GetPrimAtPath(SCAN_PROXY_ROOT_PRIM).IsValid()
    assert written.GetPrimAtPath(SCAN_PROXY_MESH_PRIM).IsA(UsdGeom.Mesh)


def test_build_scan_proxy_refuses_non_z_up(tmp_path):
    source = room_stage(tmp_path / "scene.usd", up_axis="Y")
    with pytest.raises(ValueError, match="Z-up"):
        build_scan_proxy(source, tmp_path / "scan_proxy.usd", floor_z=0.0)


def test_write_triangle_mesh_usd_round_trip(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    out = write_triangle_mesh_usd(points, faces, tmp_path / "tri.usd", root_prim="/R", mesh_prim="/R/M")
    stage = Usd.Stage.Open(str(out))
    assert stage.GetDefaultPrim().GetPath() == "/R"
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/R/M"))
    assert len(mesh.GetFaceVertexIndicesAttr().Get()) == 3
