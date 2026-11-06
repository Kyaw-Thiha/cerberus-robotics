from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pxr = pytest.importorskip("pxr")
import build_lidar_proxy as blp
from pxr import Gf, Usd, UsdGeom, Vt

from common.cerberus_worlds.scene_geometry import (
    gather_world_triangles,
    lidar_proxy_path,
)


def _tri_mesh(stage, path, pts, tris=((0, 1, 2),)):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for t in tris for i in t]))
    return mesh


def _scene(tmp_path) -> Path:
    path = tmp_path / "scene" / "start_result_navigation.usd"
    path.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/Root")
    _tri_mesh(stage, "/Root/good", [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    moved = UsdGeom.Xform.Define(stage, "/Root/moved")
    moved.AddTranslateOp().Set(Gf.Vec3d(10, 0, 5))
    _tri_mesh(stage, "/Root/moved/m", [(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    UsdGeom.Mesh.Define(stage, "/Root/empty")  # no points/faces authored: what breaks Isaac Lab's ray caster
    stage.GetRootLayer().Save()
    return path


def test_gather_skips_empty_meshes_and_applies_transforms(tmp_path):
    stage = Usd.Stage.Open(str(_scene(tmp_path)))
    stats: dict[str, int] = {}
    points, tris = gather_world_triangles(stage, "/", frozenset(), 60.0, stats)
    assert stats == {"meshes_used": 2, "meshes_skipped_empty": 1}
    assert points.shape == (6, 3) and tris.shape == (2, 3)
    assert tris.max() == 5
    assert np.isclose(points[:, 2].max(), 5.0) and np.isclose(points[:, 0].max(), 11.0)


def test_gather_raises_when_nothing_valid(tmp_path):
    path = tmp_path / "e.usd"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Mesh.Define(stage, "/m")
    with pytest.raises(RuntimeError, match="no Mesh prims with geometry"):
        gather_world_triangles(stage, "/", frozenset(), 60.0)


def test_build_proxy_round_trip_and_cache(tmp_path):
    source = _scene(tmp_path)
    proxy = lidar_proxy_path(source)
    assert proxy == source.parent / "lidar_proxy.usd"
    assert not blp.is_fresh(source, proxy)
    meta = blp.build_proxy(source, proxy, 60.0)
    assert meta["vertices"] == 6 and meta["triangles"] == 2 and meta["meshes_skipped_empty"] == 1
    assert blp.is_fresh(source, proxy)
    out = Usd.Stage.Open(str(proxy))
    assert out.GetDefaultPrim().GetPath() == "/LidarProxy"
    mesh_prim = out.GetPrimAtPath("/LidarProxy/Proxy")
    assert mesh_prim and mesh_prim.GetTypeName() == "Mesh"
    mesh = UsdGeom.Mesh(mesh_prim)
    assert len(mesh.GetPointsAttr().Get()) == 6 and len(mesh.GetFaceVertexCountsAttr().Get()) == 2
    source.touch()  # a changed source invalidates the cache
    assert not blp.is_fresh(source, proxy)


def test_y_up_stage_is_refused_and_centimetre_units_are_mirrored():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    with pytest.raises(ValueError, match="Z-up"):
        blp.check_stage_units(stage)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # what a GRScenes house declares
    assert blp.check_stage_units(stage) == {"up_axis": "Z", "meters_per_unit": 0.01}


def test_proxy_mirrors_the_source_units(tmp_path):
    """A centimetre scene must yield a centimetre proxy: converting it here would misplace it in the sim."""
    source = tmp_path / "cm" / "start_result_navigation.usd"
    source.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.Mesh.Define(stage, "/a")
    _tri_mesh(stage, "/good", [(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    stage.GetRootLayer().Save()
    proxy = lidar_proxy_path(source)
    meta = blp.build_proxy(source, proxy, 60.0)
    assert meta["meters_per_unit"] == 0.01
    out = Usd.Stage.Open(str(proxy))
    assert UsdGeom.GetStageMetersPerUnit(out) == 0.01
    assert UsdGeom.GetStageUpAxis(out) == UsdGeom.Tokens.z


def test_dry_run_does_not_build(tmp_path, monkeypatch, capsys):
    source = _scene(tmp_path)
    monkeypatch.setenv("CERBERUS_SCENES_DIR", str(tmp_path / "scenes"))
    folder = tmp_path / "scenes" / "grscenes" / "home" / "scenes" / "abc_usd"
    folder.mkdir(parents=True)
    (folder / "start_result_navigation.usd").write_bytes(source.read_bytes())
    assert blp.main(["--scene", "grscenes:abc_usd", "--dry-run"]) == 0
    assert not (folder / "lidar_proxy.usd").exists()
    assert "cached=False" in capsys.readouterr().out


def _cm_scene(tmp_path, authored_mpu=None):
    """A centimetre scene with a 100 x 100 right triangle (1 m legs) under a default prim, like GRScenes."""
    source = tmp_path / "cmscene" / "start_result_navigation.usd"
    source.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if authored_mpu is not None:
        UsdGeom.SetStageMetersPerUnit(stage, authored_mpu)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    _tri_mesh(stage, "/Root/good", [(0, 0, 0), (100, 0, 0), (0, 100, 0)])
    stage.GetRootLayer().Save()
    return source


def test_metric_scene_usd_scales_a_centimetre_scene_to_metres(tmp_path):
    """GRScenes does not author metersPerUnit (USD's fallback is 0.01); Isaac spawns it unconverted, 100x too big."""
    from common.cerberus_worlds.scene_geometry import metric_scene_usd

    source = _cm_scene(tmp_path)
    assert not UsdGeom.StageHasAuthoredMetersPerUnit(Usd.Stage.Open(str(source), Usd.Stage.LoadNone))
    wrapped = metric_scene_usd(source)
    assert wrapped == source.with_name("scene_metric.usd")
    stage = Usd.Stage.Open(str(wrapped))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetPath() == "/Scene"
    points, tris = gather_world_triangles(stage, "/", frozenset(), 60.0)
    assert tris.shape == (1, 3)
    assert np.isclose(points[:, 0].max(), 1.0) and np.isclose(points[:, 1].max(), 1.0)  # 100 cm -> 1.0 m


def test_metric_scene_usd_leaves_a_metre_scene_alone_and_caches(tmp_path):
    from common.cerberus_worlds.scene_geometry import metric_scene_usd

    metre = _cm_scene(tmp_path, authored_mpu=1.0)
    assert metric_scene_usd(metre) == metre
    assert not metre.with_name("scene_metric.usd").exists()

    other = tmp_path / "second"
    other.mkdir()
    cm = _cm_scene(other)
    first = metric_scene_usd(cm)
    mtime = first.stat().st_mtime_ns
    assert metric_scene_usd(cm) == first and first.stat().st_mtime_ns == mtime  # cached
    cm.touch()  # a changed source rebuilds the wrapper
    metric_scene_usd(cm)
    assert first.stat().st_mtime_ns != mtime


def test_proxy_of_a_metric_wrapper_is_metric(tmp_path):
    """The proxy must live in the frame the sim sees, i.e. the wrapper's metres, with a 1 m/unit stage."""
    from common.cerberus_worlds.scene_geometry import metric_scene_usd

    wrapped = metric_scene_usd(_cm_scene(tmp_path))
    proxy = lidar_proxy_path(wrapped)
    meta = blp.build_proxy(wrapped, proxy, 60.0)
    assert meta["meters_per_unit"] == 1.0
    out = Usd.Stage.Open(str(proxy))
    pts = UsdGeom.Mesh(out.GetPrimAtPath("/LidarProxy/Proxy")).GetPointsAttr().Get()
    assert np.isclose(max(p[0] for p in pts), 1.0)


def test_sidecar_cache_round_trip_and_invalidation(tmp_path):
    from common.cerberus_worlds.scene_geometry import (
        cached_metric_scene_usd,
        metric_scene_usd,
    )

    cm = _cm_scene(tmp_path)
    assert cached_metric_scene_usd(cm) is None  # never computed
    wrapped = metric_scene_usd(cm)
    assert cached_metric_scene_usd(cm) == wrapped
    cm.touch()  # the source changed: the sidecar no longer applies
    assert cached_metric_scene_usd(cm) is None

    metre = _cm_scene(tmp_path / "m", authored_mpu=1.0) if (tmp_path / "m").mkdir() is None else None
    assert metric_scene_usd(metre) == metre
    assert cached_metric_scene_usd(metre) == metre  # "already metres" is cached too


def test_sim_usd_path_does_not_import_pxr_in_the_calling_process(tmp_path):
    """Importing pxr before the Isaac app launches breaks omni.usd ('no attribute get_context' on a pod), and
    sim_usd_path runs before launch. It must get the wrapper from a child interpreter, never in-process."""
    import subprocess
    import textwrap

    cm = _cm_scene(tmp_path)
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from common.cerberus_worlds.arena_assets import SceneFiles, sim_usd_path
        src = Path({str(cm)!r})
        files = SceneFiles("grscenes:x", "grscenes", "home", src.parent, src, "navigation", None)
        out = sim_usd_path(files)
        assert out.name == "scene_metric.usd" and out.is_file(), out
        assert "pxr" not in sys.modules, "pxr was imported in the calling process"
        """
    )
    repo = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-800:]


def test_scene_geometry_main_reports_missing_pxr(monkeypatch, tmp_path):
    from common.cerberus_worlds import scene_geometry

    monkeypatch.setitem(sys.modules, "pxr", None)  # `import pxr` now raises ImportError
    assert scene_geometry.main([str(tmp_path / "x.usd")]) == scene_geometry.PXR_MISSING_EXIT
