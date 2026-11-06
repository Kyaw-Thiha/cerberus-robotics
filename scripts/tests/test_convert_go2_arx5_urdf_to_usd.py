import json
import subprocess
import sys
from pathlib import Path

import convert_go2_arx5_urdf_to_usd as conv
import pytest

from common.cerberus_description.go2_arx5_spec import (
    GO2_ARX5_BASE_PRIM_REL_PATH,
    GO2_ARX5_TCP_PRIM_REL_PATH,
    default_fused_usd_path,
)
from common.cerberus_description.urdf_patch import expected_joint_names

SCRIPT = Path(conv.__file__)


def test_parser_defaults():
    args = conv.build_parser().parse_args([])
    assert args.urdf == Path("data/assets/go2_arx5/urdf/go2_arx5.urdf")
    assert args.out == default_fused_usd_path()
    assert args.force is False
    assert args.merge_fixed_joints is False
    assert args.collider == "convex_hull"
    assert args.watchdog_s == 420.0
    assert args.dry_run is False


def test_collider_choices():
    with pytest.raises(SystemExit):
        conv.build_parser().parse_args(["--collider", "sphere"])
    assert conv.build_parser().parse_args(["--collider", "none"]).collider == "none"


def test_plan_and_validation(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text("<robot/>")
    out = tmp_path / "o" / "r.usd"
    args = conv.build_parser().parse_args(["--urdf", str(urdf), "--out", str(out)])
    plan = conv.build_plan(args)
    assert (
        plan["fix_base"] is False
        and plan["make_instanceable"] is False
        and plan["self_collision"] is False
    )
    assert plan["expected_joint_count"] == 20
    assert plan["record"].endswith("r.usd.convert.json")
    conv.validate_plan(plan)
    out.parent.mkdir()
    out.write_text("x")
    with pytest.raises(FileExistsError):
        conv.validate_plan(plan)
    plan["force"] = True
    conv.validate_plan(plan)
    urdf.unlink()
    with pytest.raises(FileNotFoundError):
        conv.validate_plan(plan)
    with pytest.raises(ValueError):
        conv.build_plan(
            conv.build_parser().parse_args(["--out", str(tmp_path / "x.txt")])
        )


def test_dry_run_subprocess_prints_plan():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["collider"] == "convex_hull" and plan["merge_fixed_joints"] is False
    assert plan["out"].endswith("data/assets/go2_arx5/go2_arx5.usd")


def test_missing_urdf_exit_2(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--urdf", str(tmp_path / "nope.urdf")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2


def test_import_does_not_import_isaac():
    code = (
        "import sys; sys.path.insert(0, %r); import convert_go2_arx5_urdf_to_usd\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('isaaclab', 'omni', 'isaacsim', 'pxr'))\n"
        "assert not bad, bad\n" % str(SCRIPT.parent)
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_classify_body():
    assert conv.classify_body("FL_thigh") == "legs"
    assert conv.classify_body("base") == "trunk"
    assert conv.classify_body("Head_upper") == "trunk"
    assert conv.classify_body("link3") == "arm"


def _make_stage(
    joints=None, nested=True, extra_root=False, with_eef=True, ref_layer=None
):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/go2_arx5").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    if extra_root:
        UsdPhysics.ArticulationRootAPI.Apply(
            UsdGeom.Xform.Define(stage, "/go2_arx5/base2").GetPrim()
        )

    def body(path, mass):
        p = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        if mass is not None:
            UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)

    body("/go2_arx5/Geometry/base", 6.0)
    body("/go2_arx5/Geometry/base/FL_hip", 0.5)
    arm = "/go2_arx5/Geometry/base/base_link" if nested else "/go2_arx5"
    if nested:
        UsdGeom.Xform.Define(stage, arm)
    body(f"{arm}/link1", 1.0)
    if with_eef:
        eef = (
            f"/go2_arx5/{GO2_ARX5_TCP_PRIM_REL_PATH}"
            if nested
            else "/go2_arx5/eef_link"
        )
        for parent in range(eef.count("/") - 1, 2, -1):
            UsdGeom.Xform.Define(stage, "/".join(eef.split("/")[: parent + 1]))
        body(eef, 0.001)
    for name in joints if joints is not None else expected_joint_names():
        UsdPhysics.RevoluteJoint.Define(stage, f"/go2_arx5/joints/{name}")
    j = UsdPhysics.FixedJoint.Define(stage, "/go2_arx5/root_joint")
    j.CreateBody1Rel().SetTargets(["/go2_arx5/base"])
    if ref_layer:
        UsdGeom.Xform.Define(
            stage, "/go2_arx5/extra"
        ).GetPrim().GetReferences().AddReference(ref_layer)
    return stage


def test_inspect_stage_ok_nested():
    stage = _make_stage()
    info = conv.inspect_stage(stage, fix_base=False)
    assert info["default_prim_path"] == "/go2_arx5"
    assert info["rigid_body_count"] == 4
    assert info["articulation_root_count"] == 1
    assert info["tcp_prim_relative_path"] == GO2_ARX5_TCP_PRIM_REL_PATH
    assert info["base_prim_relative_path"] == GO2_ARX5_BASE_PRIM_REL_PATH
    assert info["root_joint_present"] and info["root_joint_must_be_removed"]
    assert (
        conv.inspect_stage(stage, fix_base=True)["root_joint_must_be_removed"] is False
    )
    assert info["total_mass_kg"] == pytest.approx(7.501)
    assert info["mass_by_group_kg"]["arm"] == pytest.approx(1.001)
    assert info["mass_by_group_kg"]["legs"] == pytest.approx(0.5)
    assert info["external_arcs"] == []
    ok, failures = conv.evaluate_checks(info)
    assert ok, failures


def test_inspect_flat_tcp_path_fails_conformance():
    info = conv.inspect_stage(_make_stage(nested=False))
    assert info["tcp_prim_relative_path"] == "eef_link"
    ok, failures = conv.evaluate_checks(info)
    assert not ok and any(
        "tcp_prim_relative_path" in f and GO2_ARX5_TCP_PRIM_REL_PATH in f
        for f in failures
    )


def test_base_path_mismatch_fails():
    stage = _make_stage()
    stage.RemovePrim("/go2_arx5/Geometry/base/FL_hip")
    info = conv.inspect_stage(stage)
    info["base_prim_relative_path"] = "base"
    ok, failures = conv.evaluate_checks(info)
    assert not ok and any("base_prim_relative_path" in f for f in failures)


def test_checks_fail():
    names = list(expected_joint_names())[:-1]
    ok, failures = conv.evaluate_checks(conv.inspect_stage(_make_stage(joints=names)))
    assert not ok and any("missing" in f for f in failures)
    ok, failures = conv.evaluate_checks(
        conv.inspect_stage(_make_stage(extra_root=True))
    )
    assert not ok and any("PhysicsArticulationRootAPI" in f for f in failures)
    ok, failures = conv.evaluate_checks(conv.inspect_stage(_make_stage(with_eef=False)))
    assert not ok and any("eef_link" in f for f in failures)


def test_external_arcs_and_flatten(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd

    sub = tmp_path / "sub.usda"
    sub.write_text('#usda 1.0\n(defaultPrim = "S")\ndef Xform "S" {}\n')
    stage = _make_stage(ref_layer="./sub.usda")
    main_usd = tmp_path / "main.usda"
    stage.GetRootLayer().Export(str(main_usd))
    # exported layer's relative reference resolves next to main.usda
    (tmp_path / "sub.usda").write_text(sub.read_text())
    s = Usd.Stage.Open(str(main_usd))
    arcs = conv.external_arcs(s)
    assert [a["kind"] for a in arcs] == ["reference"]
    del s
    conv.flatten_in_place(main_usd)
    s2 = Usd.Stage.Open(str(main_usd))
    assert conv.external_arcs(s2) == []
    assert s2.GetDefaultPrim().GetPath().pathString == "/go2_arx5"


def test_strip_newton_root_props(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/go2_arx5", "Xform")
    base = stage.DefinePrim("/go2_arx5/base", "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(
        True
    )
    base.CreateAttribute("newton:inertia", Sdf.ValueTypeNames.Matrix3d).Set(
        Gf.Matrix3d(1)
    )
    link = stage.DefinePrim("/go2_arx5/link1", "Xform")
    link.CreateAttribute("newton:inertia", Sdf.ValueTypeNames.Matrix3d).Set(
        Gf.Matrix3d(1)
    )
    stage.SetDefaultPrim(root)
    out = tmp_path / "x.usda"
    stage.GetRootLayer().Export(str(out))
    removed = conv.strip_newton_root_props(out)
    assert sorted(removed) == [
        "/go2_arx5/base.newton:inertia",
        "/go2_arx5/base.newton:selfCollisionEnabled",
    ]
    s2 = Usd.Stage.Open(str(out))
    assert not s2.GetPrimAtPath("/go2_arx5/base").GetAttribute(
        "newton:selfCollisionEnabled"
    )
    assert s2.GetPrimAtPath("/go2_arx5/link1").GetAttribute("newton:inertia")


def test_build_report_and_write(tmp_path):
    out = tmp_path / "r.usd"
    args = conv.build_parser().parse_args(["--out", str(out), "--dry-run"])
    plan = conv.build_plan(args)
    report = conv.build_report(
        plan, conv.inspect_stage(_make_stage()), {"input_sha256": "abc"}
    )
    assert report["validation"] == "OK" and report["self_contained"] is True
    assert report["missing_joints"] == [] and report["unexpected_joints"] == []
    path = conv.write_report(plan, report)
    assert path.name == "r.usd.convert.json"
    assert (
        json.loads(path.read_text())["tcp_prim_relative_path"]
        == GO2_ARX5_TCP_PRIM_REL_PATH
    )


def test_sha256(tmp_path):
    f = tmp_path / "a"
    f.write_bytes(b"abc")
    assert conv.sha256_file(f).startswith("ba7816bf")
