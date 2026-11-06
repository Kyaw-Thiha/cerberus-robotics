import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("xacro")

from common.cerberus_description.xacro_expand import expand_xacro

REPO_ROOT = Path(__file__).resolve().parents[3]
GO2_XACRO = REPO_ROOT / "common" / "cerberus_description" / "urdf" / "go2.urdf.xacro"

MAIN = """<?xml version="1.0"?>
<robot name="tiny" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="extra" default="false"/>
  <xacro:include filename="$(find tinypkg)/urdf/inc.xacro"/>
  <link name="base">
    <visual><geometry><mesh filename="package://tinypkg/meshes/a.dae"/></geometry></visual>
    <collision><geometry><mesh filename="$(find tinypkg)/meshes/b.dae"/></geometry></collision>
  </link>
  <xacro:if value="$(arg extra)"><xacro:extra_link/></xacro:if>
</robot>
"""
INC = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="extra_link"><link name="extra"/></xacro:macro>
</robot>
"""


@pytest.fixture
def tiny_pkg(tmp_path):
    pkg = tmp_path / "pkg_real"
    (pkg / "urdf").mkdir(parents=True)
    (pkg / "meshes").mkdir()
    (pkg / "urdf" / "main.xacro").write_text(MAIN)
    (pkg / "urdf" / "inc.xacro").write_text(INC)
    return pkg


def _link_names(xml):
    return [link.get("name") for link in ET.fromstring(xml).findall("link")]


def test_expands_without_ament_index(tiny_pkg):
    xml = expand_xacro(
        tiny_pkg / "urdf" / "main.xacro", package_name="tinypkg", package_dir=tiny_pkg
    )
    assert _link_names(xml) == ["base"]


def test_args_are_passed_as_mappings(tiny_pkg):
    xml = expand_xacro(
        tiny_pkg / "urdf" / "main.xacro",
        args={"extra": "true"},
        package_name="tinypkg",
        package_dir=tiny_pkg,
    )
    assert _link_names(xml) == ["base", "extra"]


def test_mesh_paths_point_at_real_package_dir(tiny_pkg):
    xml = expand_xacro(
        tiny_pkg / "urdf" / "main.xacro", package_name="tinypkg", package_dir=tiny_pkg
    )
    files = [m.get("filename") for m in ET.fromstring(xml).iter("mesh")]
    assert files == [
        str(tiny_pkg.resolve() / "meshes" / "a.dae"),
        str(tiny_pkg.resolve() / "meshes" / "b.dae"),
    ]
    assert "xacro_expand_" not in xml


def test_sources_are_not_modified(tiny_pkg):
    expand_xacro(
        tiny_pkg / "urdf" / "main.xacro", package_name="tinypkg", package_dir=tiny_pkg
    )
    assert (tiny_pkg / "urdf" / "main.xacro").read_text() == MAIN
    assert "$(find tinypkg)" in (tiny_pkg / "urdf" / "main.xacro").read_text()


def test_missing_xacro_raises(tiny_pkg):
    with pytest.raises(FileNotFoundError):
        expand_xacro(
            tiny_pkg / "urdf" / "nope.xacro",
            package_name="tinypkg",
            package_dir=tiny_pkg,
        )


def test_missing_package_dir_raises(tiny_pkg):
    with pytest.raises(FileNotFoundError):
        expand_xacro(
            tiny_pkg / "urdf" / "main.xacro",
            package_name="tinypkg",
            package_dir=tiny_pkg / "absent",
        )


def test_repo_go2_xacro_expands_with_real_meshes():
    xml = expand_xacro(GO2_XACRO)
    root = ET.fromstring(xml)
    assert root.get("name") == "go2"
    meshes = [m.get("filename") for m in root.iter("mesh")]
    assert meshes
    for filename in meshes:
        path = Path(filename)
        assert path.is_absolute() and path.is_file()
        assert path.parent == REPO_ROOT / "common" / "cerberus_description" / "meshes"
