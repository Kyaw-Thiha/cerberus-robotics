"""Tests for the ``sim`` argument of ``urdf/go2.urdf.xacro``.

``$(find cerberus_description)`` needs an ament index, so the helper copies the
urdf dir to a temp dir and substitutes the token (includes -> temp copy, meshes
-> real package path) before expanding with the ``xacro`` python API.
"""

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

xacro = pytest.importorskip("xacro")

PKG = Path(__file__).resolve().parent.parent
GOLDEN = PKG / "test_data" / "go2_default_expanded.urdf"
PLACEHOLDER = "@CERBERUS_DESCRIPTION@"
TOKEN = "$(find cerberus_description)"
STOCK_TOTAL_MASS = 15.019
LEGS = ("FL", "FR", "RL", "RR")


def expand(**mappings) -> str:
    """Expand go2.urdf.xacro; returns URDF text with the package path replaced by PLACEHOLDER."""
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "urdf"
        shutil.copytree(PKG / "urdf", dst)
        for f in dst.glob("*.xacro"):
            text = f.read_text()
            text = text.replace(f"{TOKEN}/urdf/", f"{dst}/")
            text = text.replace(TOKEN, str(PKG))
            f.write_text(text)
        doc = xacro.process_file(str(dst / "go2.urdf.xacro"), mappings=mappings)
        out = doc.toprettyxml(indent="  ")
    return out.replace(str(PKG), PLACEHOLDER)


def parse(text: str) -> ET.Element:
    return ET.fromstring(
        text.split("?>", 1)[1] if text.lstrip().startswith("<?xml") else text
    )


def masses(root):
    return {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in root.findall("link")
        if link.find("inertial/mass") is not None
    }


@pytest.fixture(scope="module")
def sim():
    return parse(expand(sim="true"))


def canonical(text: str) -> str:
    """Element tree with comments (xacro header, nested doc comments) and whitespace layout dropped."""
    root = parse(text)
    for el in root.iter():
        el.text = (el.text or "").strip() or None
        el.tail = None
    return ET.tostring(root, encoding="unicode")


def test_default_matches_golden():
    golden = canonical(GOLDEN.read_text())
    assert canonical(expand()) == golden
    assert canonical(expand(sim="false")) == golden
    root = parse(GOLDEN.read_text())
    assert len(root.findall("link")) == 37 and len(root.findall("joint")) == 36


FRONT_THIGH = (-1.5708, 3.4907)
REAR_THIGH = (-0.5236, 4.5379)
OLD_REAR_THIGH = FRONT_THIGH  # previous (buggy) golden used front limits for all legs


def thigh_limits(root, leg):
    lim = root.find(f"joint[@name='{leg}_thigh_joint']/limit")
    return float(lim.get("lower")), float(lim.get("upper"))


def test_golden_differs_from_old_only_in_rear_thigh_limits():
    root = parse(GOLDEN.read_text())
    for leg in ("RL", "RR"):
        lim = root.find(f"joint[@name='{leg}_thigh_joint']/limit")
        assert thigh_limits(root, leg) == REAR_THIGH
        lim.set("lower", str(OLD_REAR_THIGH[0]))
        lim.set("upper", str(OLD_REAR_THIGH[1]))
    # With only those two limits restored, every thigh equals the old uniform front values,
    # i.e. the old golden differs from the new one in exactly these four attributes.
    for leg in LEGS:
        assert thigh_limits(root, leg) == OLD_REAR_THIGH
    other = {
        j.get("name"): dict(j.find("limit").attrib)
        for j in root.findall("joint")
        if j.find("limit") is not None and "thigh" not in j.get("name")
    }
    assert other  # untouched joints keep their limits (covered by golden comparison)


@pytest.mark.parametrize("variant", [{}, {"sim": "true"}])
def test_thigh_limits_front_vs_rear(variant):
    root = parse(expand(**variant))
    for leg in ("FL", "FR"):
        assert thigh_limits(root, leg) == FRONT_THIGH
    for leg in ("RL", "RR"):
        assert thigh_limits(root, leg) == REAR_THIGH


def test_sim_root_and_joints(sim):
    links = {l.get("name") for l in sim.findall("link")}
    children = {j.find("child").get("link") for j in sim.findall("joint")}
    assert links - children == {"base"}
    assert not {"world", "trunk"} & links
    joints = sim.findall("joint")
    assert sum(j.get("type") == "revolute" for j in joints) == 12
    assert all(j.get("type") in ("revolute", "fixed") for j in joints)
    assert not any(
        j.get("name") in ("floating_base", "base_static_joint") for j in joints
    )


def test_sim_no_rotor_sensor_gazebo_transmission(sim):
    names = {l.get("name") for l in sim.findall("link")} | {
        j.get("name") for j in sim.findall("joint")
    }
    assert not [n for n in names if "rotor" in n]
    for bad in ("imu", "camera", "cam_", "lidar", "load"):
        assert not [n for n in names if bad in n], bad
    assert not sim.findall("transmission")
    assert not sim.findall("gazebo")


def test_sim_leg_names(sim):
    links = {l.get("name") for l in sim.findall("link")}
    joints = {j.get("name") for j in sim.findall("joint")}
    for leg in LEGS:
        for part in ("hip", "thigh", "calf", "foot"):
            assert f"{leg}_{part}" in links
        for part in ("hip", "thigh", "calf"):
            assert f"{leg}_{part}_joint" in joints
    assert len(links) == 1 + 16


def test_sim_leg_joints_match_default(sim):
    default = parse(expand())
    for leg in LEGS:
        for part in ("hip", "thigh", "calf"):
            name = f"{leg}_{part}_joint"
            a = next(j for j in default.findall("joint") if j.get("name") == name)
            b = next(j for j in sim.findall("joint") if j.get("name") == name)
            for tag in ("origin", "axis", "limit", "dynamics"):
                assert a.find(tag).attrib == b.find(tag).attrib, (name, tag)
            expected_parent = (
                "base"
                if part == "hip"
                else f"{leg}_{'hip' if part == 'thigh' else 'thigh'}"
            )
            assert b.find("parent").get("link") == expected_parent


def test_sim_masses(sim):
    default_m = masses(parse(expand()))
    sim_m = masses(sim)
    assert sim_m["base"] == pytest.approx(6.921)
    for leg in LEGS:
        for part in ("hip", "thigh", "calf"):
            assert sim_m[f"{leg}_{part}"] == pytest.approx(default_m[f"{leg}_{part}"])
        assert sim_m[f"{leg}_foot"] == pytest.approx(0.04)
    assert sum(sim_m.values()) == pytest.approx(STOCK_TOTAL_MASS, rel=0.005)


def test_sim_trunk_inertial_matches_default(sim):
    default = parse(expand())
    a = next(l for l in default.findall("link") if l.get("name") == "trunk").find(
        "inertial"
    )
    b = next(l for l in sim.findall("link") if l.get("name") == "base").find("inertial")
    assert ET.tostring(a) == ET.tostring(b)


def test_sim_meshes_absolute(sim):
    files = [m.get("filename") for m in sim.iter("mesh")]
    assert files
    assert all(f.startswith(PLACEHOLDER + "/meshes/") for f in files), files
