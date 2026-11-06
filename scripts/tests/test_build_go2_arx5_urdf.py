import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import build_go2_arx5_urdf as builder
import pytest

from common.cerberus_description.go2_arx5_spec import GO2_ARX5_MOUNT_XYZ

SCRIPT = Path(builder.__file__)
X5A_URDF = (
    builder.REPO_ROOT / "third_party" / "arx-model" / "X5" / "X5A" / "urdf" / "X5A.urdf"
)


def _link(name, mass=None, mesh=None):
    inertial = (
        f'<inertial><mass value="{mass}"/></inertial>' if mass is not None else ""
    )
    visual = (
        f'<visual><geometry><mesh filename="{mesh}"/></geometry></visual>'
        if mesh
        else ""
    )
    return f'<link name="{name}">{inertial}{visual}</link>'


def _joint(name, parent, child, kind="revolute"):
    return f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/></joint>'


def _robot(body, name="r"):
    return ET.fromstring(f'<robot name="{name}">{body}</robot>')


@pytest.fixture
def mesh(tmp_path):
    path = tmp_path / "m.dae"
    path.write_text("x")
    return str(path)


@pytest.fixture
def go2(mesh):
    return _robot(
        '<material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>'
        + _link("base", 6.0, mesh)
        + _link("FL_hip", 1.0)
        + _link("FL_foot", 0.5)
        + _joint("FL_hip_joint", "base", "FL_hip")
        + _joint("FL_foot_joint", "FL_hip", "FL_foot", "fixed")
    )


@pytest.fixture
def arm(mesh):
    return _robot(
        '<material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>'
        '<material name="red"><color rgba="1 0 0 1"/></material>'
        + _link("base_link", 0.5, mesh)
        + _link("link1", 0.25)
        + _link("eef_link", 0.001)
        + _joint("joint1", "base_link", "link1")
        + _joint("tcp_fixed_joint", "link1", "eef_link", "fixed")
    )


MINI_NAMES = ["FL_hip_joint", "joint1"]


def _merged(go2, arm, **kw):
    return builder.merge_urdfs(go2, arm, **kw)


def test_parser_defaults():
    args = builder.build_parser().parse_args([])
    assert args.out == builder.DEFAULT_OUT
    assert args.variant == "X5A"
    assert args.go2_xacro == builder.DEFAULT_GO2_XACRO
    assert not (args.no_sim or args.dry_run or args.force)
    assert builder.DEFAULT_OUT.as_posix().endswith(
        "data/assets/go2_arx5/urdf/go2_arx5.urdf"
    )


def test_merge_structure_and_mount(go2, arm):
    merged, info = _merged(
        go2, arm, mount_xyz=(0.1, 0.0, 0.07), mount_rpy_radians=[0, 0, math.pi / 2]
    )
    assert merged.get("name") == "go2_arx5"
    assert [m.get("name") for m in merged.findall("material")] == ["grey", "red"]
    assert info["materials"] == 2 and info["material_conflicts"] == []
    assert [e.get("name") for e in merged.findall("link")] == [
        "base",
        "FL_hip",
        "FL_foot",
        "base_link",
        "link1",
        "eef_link",
    ]
    mount = next(
        j for j in merged.findall("joint") if j.get("name") == "arm_mount_joint"
    )
    assert mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == "base"
    assert mount.find("child").get("link") == "base_link"
    assert [float(v) for v in mount.find("origin").get("xyz").split()] == [
        0.1,
        0.0,
        0.07,
    ]
    assert float(mount.find("origin").get("rpy").split()[2]) == pytest.approx(
        math.pi / 2
    )


def test_default_mount_is_spec(go2, arm):
    merged, _ = _merged(go2, arm)
    origin = next(
        j for j in merged.findall("joint") if j.get("name") == "arm_mount_joint"
    ).find("origin")
    assert [float(v) for v in origin.get("xyz").split()] == list(GO2_ARX5_MOUNT_XYZ)


def test_mount_rpy_converts_degrees_to_radians():
    assert builder.mount_rpy_rad((0, 0, 180)) == [0.0, 0.0, pytest.approx(math.pi)]


def test_inputs_not_modified(go2, arm):
    before = ET.tostring(go2), ET.tostring(arm)
    _merged(go2, arm)
    assert (ET.tostring(go2), ET.tostring(arm)) == before


def test_link_collision_detected(go2, mesh):
    arm = _robot(
        _link("base_link") + _link("FL_hip") + _joint("joint1", "base_link", "FL_hip")
    )
    with pytest.raises(ValueError, match=r"collision.*FL_hip"):
        _merged(go2, arm)


def test_joint_collision_detected(go2):
    arm = _robot(
        _link("base_link")
        + _link("link1")
        + _joint("FL_hip_joint", "base_link", "link1")
    )
    with pytest.raises(ValueError, match=r"collision.*FL_hip_joint"):
        _merged(go2, arm)


def test_mount_joint_name_collision_detected(go2):
    arm = _robot(
        _link("base_link")
        + _link("link1")
        + _joint("arm_mount_joint", "base_link", "link1")
    )
    with pytest.raises(ValueError, match="arm_mount_joint"):
        _merged(go2, arm)


def test_missing_mount_links(go2, arm):
    with pytest.raises(ValueError, match="mount parent"):
        _merged(go2, arm, parent_link="trunk")
    with pytest.raises(ValueError, match="mount child"):
        _merged(go2, arm, child_link="nope")


def test_validate_ok_and_mass(go2, arm):
    merged, _ = _merged(go2, arm)
    checks = builder.validate_merged(merged, expected_joints=MINI_NAMES)
    assert checks["root_link"] == "base"
    assert checks["total_mass_kg"] == pytest.approx(
        6.0 + 1.0 + 0.5 + 0.5 + 0.25 + 0.001
    )
    assert checks["actuated_joints"] == MINI_NAMES
    assert checks["meshes_checked"] == 2


def test_validate_detects_wrong_root(mesh, arm):
    go2 = _robot(
        _link("world")
        + _link("base", 6.0)
        + _link("FL_hip", 1.0)
        + _joint("floating", "world", "base", "fixed")
        + _joint("FL_hip_joint", "base", "FL_hip")
    )
    merged, _ = _merged(go2, arm)
    with pytest.raises(ValueError, match="root link is 'world'"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES)
    assert (
        builder.validate_merged(merged, expected_joints=MINI_NAMES, expected_root=None)[
            "root_link"
        ]
        == "world"
    )


def test_validate_detects_two_roots_and_two_parents(go2, arm):
    merged, _ = _merged(go2, arm)
    merged.append(ET.fromstring('<link name="orphan"/>'))
    with pytest.raises(ValueError, match="exactly one root"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES)
    merged, _ = _merged(go2, arm)
    merged.append(ET.fromstring(_joint("again", "FL_hip", "link1", "fixed")))
    with pytest.raises(ValueError, match="more than one parent"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES)


def test_validate_expected_names(go2, arm):
    merged, _ = _merged(go2, arm)
    with pytest.raises(ValueError, match=r"missing=\['joint2'\].*unexpected=\[\]"):
        builder.validate_merged(merged, expected_joints=[*MINI_NAMES, "joint2"])
    with pytest.raises(ValueError, match=r"unexpected=\['joint1'\]"):
        builder.validate_merged(merged, expected_joints=["FL_hip_joint"])
    # the default contract is the real 20-joint set, which the mini robot does not satisfy
    with pytest.raises(ValueError, match="actuated joints differ"):
        builder.validate_merged(merged)


def test_validate_requires_eef_and_existing_meshes(go2, arm, tmp_path):
    merged, _ = _merged(go2, arm)
    with pytest.raises(ValueError, match="TCP link"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES, tcp_link="tcp")
    for m in merged.iter("mesh"):
        m.set("filename", str(tmp_path / "missing.dae"))
        break
    with pytest.raises(ValueError, match="mesh path"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES)
    for m in merged.iter("mesh"):
        m.set("filename", "package://x/meshes/a.dae")
        break
    with pytest.raises(ValueError, match="mesh path"):
        builder.validate_merged(merged, expected_joints=MINI_NAMES)


def test_check_overwrite(tmp_path):
    out = tmp_path / "a.urdf"
    builder.check_overwrite([out], False)
    out.write_text("x")
    with pytest.raises(FileExistsError):
        builder.check_overwrite([out], False)
    builder.check_overwrite([out], True)


def test_record_path():
    assert builder.build_record_path(Path("/a/go2_arx5.urdf")) == Path(
        "/a/go2_arx5.urdf.build.json"
    )


def test_dry_run_plan_in_process(tmp_path, capsys):
    out = tmp_path / "x" / "go2_arx5.urdf"
    assert builder.main(["--dry-run", "--out", str(out)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["out"] == str(out.resolve())
    assert plan["record"] == str(out.resolve()) + ".build.json"
    assert plan["xacro_args"] == {"sim": "true"}
    assert plan["root_link"] == "base"
    assert plan["mount_joint"] == "arm_mount_joint"
    assert plan["mount_xyz"] == list(GO2_ARX5_MOUNT_XYZ)
    assert plan["variant"] == "X5A"
    assert len(plan["expected_joint_names"]) == 20
    assert not out.parent.exists()


def test_dry_run_no_sim(tmp_path, capsys):
    assert (
        builder.main(["--dry-run", "--no-sim", "--out", str(tmp_path / "o.urdf")]) == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["xacro_args"] == {} and plan["root_link"] is None


def test_dry_run_subprocess(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--out", str(tmp_path / "o.urdf")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["robot_name"] == "go2_arx5"


def test_refuses_overwrite_without_force(tmp_path, capsys):
    out = tmp_path / "o.urdf"
    out.write_text("keep")
    rc = builder.main(["--out", str(out)])
    assert rc == 2
    assert out.read_text() == "keep"
    assert "--force" in capsys.readouterr().err


def test_missing_go2_xacro_is_bad_args(tmp_path):
    if not X5A_URDF.is_file():
        pytest.skip("X5A not fetched; the missing-arm path is exercised instead")
    assert (
        builder.main(
            [
                "--out",
                str(tmp_path / "o.urdf"),
                "--go2-xacro",
                str(tmp_path / "none.xacro"),
            ]
        )
        == 2
    )


def test_missing_arm_urdf_is_bad_args(tmp_path, monkeypatch):
    monkeypatch.setattr(
        builder,
        "arm_urdf_path",
        lambda variant: (_ for _ in ()).throw(FileNotFoundError("x")),
    )
    assert builder.main(["--out", str(tmp_path / "o.urdf")]) == 2


def test_run_build_with_synthetic_inputs(tmp_path, go2, arm, monkeypatch):
    """Whole run_build flow with a tiny Go2 'xacro' (expansion stubbed) and a tiny arm URDF."""
    arm_path = tmp_path / "arm.urdf"
    ET.ElementTree(arm).write(arm_path)
    go2_xacro = tmp_path / "go2.urdf.xacro"
    go2_xacro.write_text("<robot/>")
    monkeypatch.setattr(
        builder,
        "expand_xacro",
        lambda path, args=None: ET.tostring(go2, encoding="unicode"),
    )
    monkeypatch.setattr(
        builder, "patch_arm_urdf", lambda src, dst: _copy_file(src, dst)
    )
    monkeypatch.setattr(builder, "expected_joint_names", lambda: tuple(MINI_NAMES))
    monkeypatch.setattr(builder, "arm_urdf_path", lambda variant: arm_path)
    out = tmp_path / "out" / "go2_arx5.urdf"
    args = builder.build_parser().parse_args(
        ["--out", str(out), "--go2-xacro", str(go2_xacro)]
    )
    plan = builder.build_plan(args)
    result = builder.run_build(plan)
    assert out.is_file()
    written = json.loads(Path(plan["record"]).read_text())
    assert written["masses"]["total_kg"] == pytest.approx(8.251)
    assert written["inputs_sha256"]["arm_urdf"] == builder.sha256_file(arm_path)
    assert written["mount_joint"] == "arm_mount_joint"
    assert result["output_sha256"] == builder.sha256_file(out)
    with pytest.raises(FileExistsError):
        builder.run_build(plan)
    assert builder.run_build(plan | {"force": True})["checks"]["ok"]


def _copy_file(src, dst):
    Path(dst).write_bytes(Path(src).read_bytes())
    return Path(dst)


@pytest.mark.skipif(
    not X5A_URDF.is_file(),
    reason="X5A URDF not fetched (third_party/arx-model, no license)",
)
def test_real_build_end_to_end(tmp_path):
    pytest.importorskip("xacro")
    out = tmp_path / "go2_arx5.urdf"
    rc = builder.main(["--out", str(out)])
    assert rc == 0
    record = json.loads(Path(str(out) + ".build.json").read_text())
    assert record["checks"]["root_link"] == "base"
    assert len(record["checks"]["actuated_joints"]) == 20
    assert 15.0 < record["masses"]["total_kg"] < 20.0
    root = ET.parse(out).getroot()
    assert root.get("name") == "go2_arx5"
    assert not any(
        link.get("name", "").endswith("_rotor") for link in root.findall("link")
    )
