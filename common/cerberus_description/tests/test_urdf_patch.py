import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from common.cerberus_description import go2_arx5_spec as spec
from common.cerberus_description import urdf_patch
from common.cerberus_description.urdf_patch import (
    expected_joint_names,
    patch_arm_urdf,
    validate_joint_names,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_URDF = REPO_ROOT / "third_party" / "arx5-sdk" / "models" / "X5.urdf"
X5A_URDF = REPO_ROOT / "third_party" / "arx-model" / "X5" / "X5A" / "urdf" / "X5A.urdf"

SYNTHETIC_URDF = """<?xml version="1.0"?>
<robot name="tiny">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <visual><geometry><mesh filename="./meshes/base_link.STL"/></geometry></visual>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="./meshes/X5/link1.STL"/></geometry></visual>
  </link>
  <link name="eef_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="gripper_fixed_joint" type="fixed">
    <parent link="link1"/><child link="eef_link"/>
  </joint>
</robot>
"""


def _make_synthetic(tmp_path, with_meshes=True):
    models = tmp_path / "models"
    (models / "meshes" / "X5").mkdir(parents=True)
    if with_meshes:
        (models / "meshes" / "base_link.STL").write_text("solid")
        (models / "meshes" / "X5" / "link1.STL").write_text("solid")
    src = models / "tiny.urdf"
    src.write_text(SYNTHETIC_URDF)
    return src


def _eef_mass(path):
    root = ET.parse(path).getroot()
    eef = next(link for link in root.iter("link") if link.get("name") == "eef_link")
    return float(eef.find("inertial/mass").get("value")), eef.find("inertial/inertia")


def test_patch_synthetic_sets_eef_mass_and_absolute_meshes(tmp_path):
    src = _make_synthetic(tmp_path)
    dst = tmp_path / "out" / "nested" / "patched.urdf"
    result = patch_arm_urdf(src, dst, eef_mass_kg=2e-3)
    assert result == dst and dst.is_file()
    mass, inertia = _eef_mass(dst)
    assert mass == pytest.approx(2e-3)
    assert float(inertia.get("ixx")) > 0
    filenames = [m.get("filename") for m in ET.parse(dst).getroot().iter("mesh")]
    assert len(filenames) == 2
    for name in filenames:
        assert Path(name).is_absolute() and Path(name).is_file()
    # the existing base_link inertial is untouched
    root = ET.parse(dst).getroot()
    base = next(link for link in root.iter("link") if link.get("name") == "base_link")
    assert float(base.find("inertial/mass").get("value")) == 1.0


def test_patch_repairs_wrong_mesh_path_with_unique_basename(tmp_path, monkeypatch):
    src = _make_synthetic(tmp_path)
    meshes = src.parent / "meshes"
    # upstream-style bug: urdf says ./meshes/base_link.STL but the file lives in meshes/X5/
    (meshes / "base_link.STL").rename(meshes / "X5" / "base_link.STL")
    # caplog is unreliable in the pixi ROS pytest env (records stay empty), so spy on the logger.
    warnings: list[str] = []
    monkeypatch.setattr(urdf_patch.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    dst = patch_arm_urdf(src, tmp_path / "p.urdf")
    filenames = [m.get("filename") for m in ET.parse(dst).getroot().iter("mesh")]
    assert str((meshes / "X5" / "base_link.STL").resolve()) in filenames
    assert all(Path(f).is_file() for f in filenames)
    assert any("repaired upstream mesh path" in w for w in warnings)


def test_patch_ambiguous_mesh_basename_raises(tmp_path):
    src = _make_synthetic(tmp_path)
    meshes = src.parent / "meshes"
    (meshes / "base_link.STL").unlink()
    (meshes / "X5" / "base_link.STL").write_text("solid")
    (meshes / "other").mkdir()
    (meshes / "other" / "base_link.STL").write_text("solid")
    with pytest.raises(FileNotFoundError, match="2 candidate"):
        patch_arm_urdf(src, tmp_path / "p.urdf")



def test_patch_synthetic_default_mass(tmp_path):
    src = _make_synthetic(tmp_path)
    dst = patch_arm_urdf(src, tmp_path / "p.urdf")
    assert _eef_mass(dst)[0] == pytest.approx(1e-3)


def test_patch_missing_mesh_raises(tmp_path):
    src = _make_synthetic(tmp_path, with_meshes=False)
    with pytest.raises(FileNotFoundError):
        patch_arm_urdf(src, tmp_path / "p.urdf")


def test_patch_does_not_modify_src(tmp_path):
    src = _make_synthetic(tmp_path)
    before = src.read_bytes()
    patch_arm_urdf(src, tmp_path / "p.urdf")
    assert src.read_bytes() == before


def test_patch_is_idempotent(tmp_path):
    src = _make_synthetic(tmp_path)
    a = patch_arm_urdf(src, tmp_path / "a.urdf")
    first = a.read_bytes()
    patch_arm_urdf(src, a)
    assert a.read_bytes() == first
    # patching the already-patched file (absolute meshes) is also stable
    b = patch_arm_urdf(a, tmp_path / "b.urdf")
    assert b.read_bytes() == first


def test_patch_rejects_non_positive_mass(tmp_path):
    src = _make_synthetic(tmp_path)
    with pytest.raises(ValueError):
        patch_arm_urdf(src, tmp_path / "p.urdf", eef_mass_kg=0.0)


def test_patch_real_x5_urdf(tmp_path):
    if not REAL_URDF.is_file():
        pytest.skip("X5.urdf not fetched (run scripts/fetch_arx5_assets.py)")
    before = REAL_URDF.read_bytes()
    dst = patch_arm_urdf(REAL_URDF, tmp_path / "X5_patched.urdf")
    assert REAL_URDF.read_bytes() == before
    assert _eef_mass(dst)[0] > 0
    meshes = [m.get("filename") for m in ET.parse(dst).getroot().iter("mesh")]
    assert meshes
    assert all(Path(m).is_absolute() and Path(m).is_file() for m in meshes)


def test_expected_joint_names_length_and_order():
    names = expected_joint_names()
    assert len(names) == 20
    assert len(set(names)) == 20
    assert names[0] == "FL_hip_joint"
    assert names[-8:] == tuple(f"joint{i}" for i in range(1, 9))
    assert names[-2:] == ("joint7", "joint8")


def test_validate_joint_names_rejects_missing_fingers():
    with pytest.raises(ValueError, match="joint8"):
        validate_joint_names(n for n in expected_joint_names() if n != "joint8")


def _links(path):
    return {link.get("name"): link for link in ET.parse(path).getroot().iter("link")}


def _joints(path):
    return {j.get("name"): j for j in ET.parse(path).getroot().iter("joint")}


def _patch_x5a(tmp_path, name="X5A_patched.urdf"):
    return patch_arm_urdf(X5A_URDF, tmp_path / name)


def test_patch_real_x5a_urdf(tmp_path):
    if not X5A_URDF.is_file():
        pytest.skip("X5A.urdf not fetched (run scripts/fetch_arx5_assets.py)")
    before = X5A_URDF.read_bytes()
    dst = _patch_x5a(tmp_path)
    assert X5A_URDF.read_bytes() == before
    root = ET.parse(dst).getroot()
    meshes = [m.get("filename") for m in root.iter("mesh")]
    assert meshes
    for m in meshes:
        assert not m.startswith("package://")
        assert Path(m).is_absolute() and Path(m).is_file()
        assert Path(m).parent.name == "meshes"


def test_patch_real_x5a_adds_tcp_joint_and_link(tmp_path):
    if not X5A_URDF.is_file():
        pytest.skip("X5A.urdf not fetched")
    dst = _patch_x5a(tmp_path)
    joint = _joints(dst)["tcp_fixed_joint"]
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "link6"
    assert joint.find("child").get("link") == "eef_link"
    xyz = tuple(float(v) for v in joint.find("origin").get("xyz").split())
    assert xyz == pytest.approx(spec.GO2_ARX5_TCP_OFFSET_XYZ)
    eef = _links(dst)["eef_link"]
    assert float(eef.find("inertial/mass").get("value")) == pytest.approx(1e-3)
    assert eef.find("visual") is None and eef.find("collision") is None


def test_patch_real_x5a_limits_effort_velocity(tmp_path):
    if not X5A_URDF.is_file():
        pytest.skip("X5A.urdf not fetched")
    joints = _joints(_patch_x5a(tmp_path))
    for name in spec.ARX5_ARM_JOINTS:
        lim = joints[name].find("limit")
        lo, hi = spec.ARX5_JOINT_LIMITS_RAD[name]
        assert float(lim.get("lower")) == pytest.approx(lo)
        assert float(lim.get("upper")) == pytest.approx(hi)
        assert float(lim.get("effort")) == pytest.approx(spec.ARX5_ACTUATOR_GAINS[name].effort_limit)
        assert float(lim.get("velocity")) == pytest.approx(spec.ARX5_JOINT_VELOCITY_LIMIT_RAD_S)
    for name, finger in spec.ARX5_FINGER_SPECS.items():
        assert joints[name].get("type") == "prismatic"
        lim = joints[name].find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == (0.0, 0.044)
        assert float(lim.get("effort")) == pytest.approx(finger.effort_n)
        assert float(lim.get("velocity")) == pytest.approx(finger.velocity_m_s)
        assert joints[name].find("mimic") is None


def test_patch_real_x5a_every_link_has_inertia(tmp_path):
    if not X5A_URDF.is_file():
        pytest.skip("X5A.urdf not fetched")
    links = _links(_patch_x5a(tmp_path))
    assert set(links) == {*spec.ARX5_LINK_MASSES_KG, "eef_link"}
    for name, link in links.items():
        assert float(link.find("inertial/mass").get("value")) > 0, name
        inertia = link.find("inertial/inertia")
        assert all(float(inertia.get(k)) > 0 for k in ("ixx", "iyy", "izz")), name


def test_patch_real_x5a_logs_changes_and_is_idempotent(tmp_path, monkeypatch):
    if not X5A_URDF.is_file():
        pytest.skip("X5A.urdf not fetched")
    infos: list[str] = []
    monkeypatch.setattr(urdf_patch.logger, "info", lambda msg, *a: infos.append(msg % a))
    a = _patch_x5a(tmp_path, "a.urdf")
    assert any("tcp_fixed_joint" in m for m in infos)
    assert any("joint2" in m and "lower" in m for m in infos)
    assert any("package://X5A/meshes/link7.STL" in m for m in infos)
    b = patch_arm_urdf(a, tmp_path / "b.urdf")
    assert b.read_bytes() == a.read_bytes()


def test_patch_x5a_synthetic_package_uri_and_missing_mesh(tmp_path):
    pkg = tmp_path / "X5A"
    (pkg / "urdf").mkdir(parents=True)
    (pkg / "meshes").mkdir()
    urdf = """<robot name="X5A">
  <link name="link6"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <visual><geometry><mesh filename="package://X5A/meshes/link6.STL"/></geometry></visual></link>
  %s
</robot>"""
    joints = "".join(
        f'<joint name="joint{i}" type="{"prismatic" if i > 6 else "revolute"}"><parent link="link6"/>'
        f'<child link="c{i}"/><limit lower="-10" upper="10" effort="100" velocity="1000"/></joint>'
        f'<link name="c{i}"/>'
        for i in range(1, 9)
    )
    src = pkg / "urdf" / "X5A.urdf"
    src.write_text(urdf % joints)
    with pytest.raises(FileNotFoundError):
        patch_arm_urdf(src, tmp_path / "o.urdf")
    (pkg / "meshes" / "link6.STL").write_text("solid")
    dst = patch_arm_urdf(src, tmp_path / "o.urdf")
    root = ET.parse(dst).getroot()
    assert next(root.iter("mesh")).get("filename") == str((pkg / "meshes" / "link6.STL").resolve())
    # childless-inertia links (c1..c8) got a positive inertial
    assert float(_links(dst)["c3"].find("inertial/mass").get("value")) > 0


def test_patch_package_uri_rejected_for_non_x5a(tmp_path):
    src = _make_synthetic(tmp_path)
    text = src.read_text().replace("./meshes/base_link.STL", "package://tiny/meshes/base_link.STL")
    src.write_text(text)
    with pytest.raises(ValueError, match="package://"):
        patch_arm_urdf(src, tmp_path / "p.urdf")


def test_validate_joint_names_happy_path():
    validate_joint_names(reversed(expected_joint_names()))


def test_validate_joint_names_missing():
    names = list(expected_joint_names())
    names.remove("joint3")
    with pytest.raises(ValueError, match="joint3"):
        validate_joint_names(names)


def test_validate_joint_names_unexpected():
    names = [*expected_joint_names(), "FL_foot_joint"]
    with pytest.raises(ValueError, match="FL_foot_joint"):
        validate_joint_names(names)
