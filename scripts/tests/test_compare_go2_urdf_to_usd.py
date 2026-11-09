import math
import subprocess
import sys
from pathlib import Path

import compare_go2_urdf_to_usd as c
import pytest

SCRIPT = Path(c.__file__)

URDF = """<robot name="r">
<link name="base"><inertial><origin xyz="0.01 0 0"/><mass value="6.921"/>
<inertia ixx="0.1" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.2"/></inertial></link>
<link name="FL_hip"><inertial><origin xyz="0 0 0"/><mass value="0.678"/>
<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
<link name="link1"/>
<joint name="FL_hip_joint" type="revolute"><parent link="base"/><child link="FL_hip"/><axis xyz="1 0 0"/>
<limit lower="-1.0472" upper="1.0472" effort="23.7" velocity="30"/></joint>
<joint name="arm_mount_joint" type="fixed"><parent link="base"/><child link="link1"/></joint>
</robot>"""


def test_inertia_eigenvalues_sorted():
    ev, _ = c.principal_inertia(c.inertia_matrix(0.3, 0, 0, 0.1, 0, 0.2))
    assert ev.tolist() == pytest.approx([0.1, 0.2, 0.3])
    ev, _ = c.principal_inertia(c.inertia_matrix(2, 1, 0, 2, 0, 5))
    assert ev.tolist() == pytest.approx([1, 3, 5])


def test_parse_and_go2_names():
    u = c.parse_urdf(URDF)
    assert c.go2_names(u) == {"base", "FL_hip"}
    assert u["joints"]["FL_hip_joint"]["effort"] == 23.7


def test_deg_rad():
    assert c.deg_to_rad(180) == pytest.approx(math.pi)


def _stage(
    mass=6.921,
    hip_mass=0.678,
    lo=-60.0,
    hi=60.0,
    extra_body=None,
    arm_parent="/go2_arx5/arx5_arm",
):
    pxr = pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/go2_arx5")

    def body(name, m, com, diag, parent="/go2_arx5"):
        p = UsdGeom.Xform.Define(stage, f"{parent}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        api = UsdPhysics.MassAPI.Apply(p)
        api.CreateMassAttr(m)
        api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
        return p

    body("base", mass, (0.01, 0, 0), (0.3, 0.1, 0.2))
    body("FL_hip", hip_mass, (0, 0, 0), (1, 1, 1))
    body("link1", 1.0, (0, 0, 0), (1, 1, 1), parent=arm_parent)
    mount = UsdPhysics.FixedJoint.Define(stage, "/go2_arx5/base/arm_mount_joint")
    mount.CreateBody0Rel().SetTargets(["/go2_arx5/base"])
    mount.CreateBody1Rel().SetTargets([f"{arm_parent}/link1"])
    if extra_body:
        body(extra_body, 0.1, (0, 0, 0), (1, 1, 1))
    j = UsdPhysics.RevoluteJoint.Define(stage, "/go2_arx5/base/FL_hip_joint")
    j.CreateBody0Rel().SetTargets(["/go2_arx5/base"])
    j.CreateBody1Rel().SetTargets(["/go2_arx5/FL_hip"])
    j.CreateAxisAttr("X")
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    return stage


@pytest.mark.parametrize(
    "arm_parent", ["/go2_arx5/base/base_link", "/go2_arx5/arx5_arm", "/go2_arx5"]
)
def test_read_usd_skips_arm_by_name(arm_parent):
    data = c.read_usd(_stage(arm_parent=arm_parent))
    assert set(data["bodies"]) == {"base", "FL_hip"}
    assert set(data["joints"]) == {"FL_hip_joint"}
    assert data["joints"]["FL_hip_joint"]["child"] == "FL_hip"


def test_compare_pass():
    rep = c.compare(c.parse_urdf(URDF), c.read_usd(_stage()))
    assert rep["result"] == "PASS", rep["failures"]
    assert rep["stats"]["max_limit_diff_rad"] < 1e-3
    assert rep["stats"]["max_inertia_rel_diff"] == pytest.approx(0, abs=1e-6)


def test_threshold_fail_mass_and_limits():
    u = c.parse_urdf(URDF)
    assert c.compare(u, c.read_usd(_stage(mass=7.5)))["result"] == "FAIL"
    assert (
        c.compare(u, c.read_usd(_stage(mass=7.5)), {"mass_rel": 0.2})["result"]
        == "PASS"
    )
    rep = c.compare(u, c.read_usd(_stage(lo=-50.0)))
    assert rep["result"] == "FAIL" and rep["stats"]["max_limit_diff_rad"] > 0.1


def test_missing_names_detected():
    rep = c.compare(c.parse_urdf(URDF), c.read_usd(_stage(extra_body="Head_upper")))
    assert rep["name_diffs"]["links_only_in_usd"] == ["Head_upper"]
    assert rep["result"] == "FAIL"


def test_cli_help():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0 and "--urdf" in out.stdout


def test_cli_missing_inputs(tmp_path):
    assert (
        c.main(["--urdf", str(tmp_path / "x.urdf"), "--usd", str(tmp_path / "x.usd")])
        == 2
    )


# ---- synthetic dict-level cases (no pxr needed) ----

URDF2 = """<robot name="r">
<link name="base"><inertial><origin xyz="0 0 0"/><mass value="5"/>
<inertia ixx="0.1" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.2"/></inertial></link>
<link name="FL_calf"><inertial><origin xyz="0 0 0"/><mass value="0.2"/>
<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
<link name="FL_foot"><inertial><origin xyz="0 0 0"/><mass value="0.04"/>
<inertia ixx="6.4e-6" ixy="0" ixz="0" iyy="6.4e-6" iyz="0" izz="6.4e-6"/></inertial></link>
<joint name="FL_calf_joint" type="revolute"><parent link="base"/><child link="FL_calf"/><axis xyz="0 1 0"/>
<limit lower="-2.7" upper="-0.8" effort="35.55" velocity="20.06"/></joint>
<joint name="FL_foot_fixed" type="fixed"><parent link="FL_calf"/><child link="FL_foot"/></joint>
</robot>"""


def _body(mass, diag, com=(0.0, 0.0, 0.0)):
    return {"path": "/x", "mass": mass, "com": list(com), "diag": list(diag)}


def _joint(typ, parent, child, axis=None, lo=None, hi=None, effort=None, vel=None):
    return {
        "path": "/j",
        "type": typ,
        "parent": parent,
        "child": child,
        "axis": axis,
        "lower_deg": lo,
        "upper_deg": hi,
        "effort": effort,
        "velocity": vel,
    }


def _usd2(foot_diag=(6.4e-6,) * 3, extra_bodies=None, extra_joints=None, **calf):
    kw = {
        "axis": "X",
        "lo": math.degrees(-2.7),
        "hi": math.degrees(-0.8),
        "effort": 45.43,
        "vel": 900.0,
    }
    kw.update(calf)
    bodies = {
        "base": _body(5.0, (0.1, 0.2, 0.3)),
        "FL_calf": _body(0.2, (0.01,) * 3),
        "FL_foot": _body(0.04, foot_diag),
    }
    bodies.update(extra_bodies or {})
    joints = {
        "FL_calf_joint": _joint("PhysicsRevoluteJoint", "base", "FL_calf", **kw),
        "FL_foot_joint": _joint("PhysicsFixedJoint", "FL_calf", "FL_foot"),
    }
    joints.update(extra_joints or {})
    return {"bodies": bodies, "joints": joints}


def test_synthetic_baseline_passes_with_rename_info():
    rep = c.compare(c.parse_urdf(URDF2), _usd2())
    assert rep["result"] == "PASS", rep["failures"]
    assert rep["info"]["fixed_joint_renames"] == [
        {
            "urdf": "FL_foot_fixed",
            "usd": "FL_foot_joint",
            "parent": "FL_calf",
            "child": "FL_foot",
        }
    ]
    assert rep["name_diffs"]["joints_only_in_urdf"] == []
    assert rep["name_diffs"]["joints_only_in_usd"] == []


def test_inertia_absolute_floor():
    u = c.parse_urdf(URDF2)
    # 50% relative diff but only 3.2e-6 absolute: passes with the default floor.
    small = _usd2(foot_diag=(9.6e-6,) * 3)
    rep = c.compare(u, small)
    assert rep["result"] == "PASS", rep["failures"]
    row = next(r for r in rep["bodies"] if r["name"] == "FL_foot")
    assert row["inertia_rel_diff"] > 0.3 and row["inertia_abs_diff"] < 1e-5
    # Tightening the absolute tolerance makes the same data fail.
    assert c.compare(u, small, {"inertia_abs": 1e-7})["result"] == "FAIL"
    # Large absolute AND relative difference fails.
    big = _usd2(foot_diag=(2e-3,) * 3)
    assert c.compare(u, big)["result"] == "FAIL"
    # Large absolute but small relative difference passes.
    u2 = c.parse_urdf(URDF2.replace('ixx="0.01"', 'ixx="0.0100"'))
    ok = _usd2()
    ok["bodies"]["FL_calf"]["diag"] = [0.0105, 0.01, 0.01]
    assert c.compare(u2, ok, {"inertia_rel": 0.10})["result"] == "PASS"


def test_ignored_light_usd_only_links():
    head = {
        "Head_upper": _body(0.001, (1e-6,) * 3),
        "Head_lower": _body(0.001, (1e-6,) * 3),
    }
    hj = {
        "Head_upper_joint": _joint("PhysicsFixedJoint", "base", "Head_upper"),
        "Head_lower_joint": _joint("PhysicsFixedJoint", "Head_upper", "Head_lower"),
    }
    rep = c.compare(c.parse_urdf(URDF2), _usd2(extra_bodies=head, extra_joints=hj))
    assert rep["result"] == "PASS", rep["failures"]
    ign = rep["info"]["ignored_usd_only"]
    assert [x["name"] for x in ign["links"]] == ["Head_lower", "Head_upper"]
    assert [x["name"] for x in ign["joints"]] == [
        "Head_lower_joint",
        "Head_upper_joint",
    ]
    assert rep["name_diffs"]["links_only_in_usd"] == []
    # Lowering the ignore threshold makes them failures again.
    rep = c.compare(
        c.parse_urdf(URDF2),
        _usd2(extra_bodies=head, extra_joints=hj),
        {"ignore_usd_only_mass": 1e-4},
    )
    assert rep["result"] == "FAIL"
    assert rep["name_diffs"]["links_only_in_usd"] == ["Head_lower", "Head_upper"]


def test_heavy_usd_only_link_fails():
    heavy = {"Head_upper": _body(0.5, (1e-3,) * 3)}
    hj = {"Head_upper_joint": _joint("PhysicsFixedJoint", "base", "Head_upper")}
    rep = c.compare(c.parse_urdf(URDF2), _usd2(extra_bodies=heavy, extra_joints=hj))
    assert rep["result"] == "FAIL"
    assert rep["name_diffs"]["links_only_in_usd"] == ["Head_upper"]
    assert rep["name_diffs"]["joints_only_in_usd"] == ["Head_upper_joint"]
    assert rep["info"]["ignored_usd_only"] == {"links": [], "joints": []}


def test_urdf_only_link_fails():
    usd = _usd2()
    del usd["bodies"]["FL_foot"], usd["joints"]["FL_foot_joint"]
    rep = c.compare(c.parse_urdf(URDF2), usd)
    assert rep["result"] == "FAIL"
    assert rep["name_diffs"]["links_only_in_urdf"] == ["FL_foot"]


def test_fixed_joint_with_wrong_connectivity_fails():
    usd = _usd2()
    usd["joints"]["FL_foot_joint"]["parent"] = "base"
    rep = c.compare(c.parse_urdf(URDF2), usd)
    assert rep["result"] == "FAIL"
    assert rep["name_diffs"]["joints_only_in_urdf"] == ["FL_foot_fixed"]


def test_axis_and_effort_mismatch_never_fail():
    rep = c.compare(c.parse_urdf(URDF2), _usd2(axis="Z", effort=1000.0, vel=1.0))
    assert rep["result"] == "PASS", rep["failures"]
    assert rep["stats"]["n_axis_mismatch_informational"] == 1
    assert rep["info"]["axis_mismatches"] == ["FL_calf_joint"]
    assert rep["info"]["max_effort_rel_diff"] > 0.9
    assert rep["info"]["max_velocity_rel_diff"] > 0.9
    text = c.summary(rep)
    assert "axis" not in text.lower()
    assert "info: max effort" in text and "info: max velocity" in text


def test_cli_flags_exist():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert "--inertia-abs-tol" in out.stdout and "--ignore-usd-only-mass" in out.stdout
