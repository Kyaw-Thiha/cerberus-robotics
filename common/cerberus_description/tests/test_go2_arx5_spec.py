import ast
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from common.cerberus_description import go2_arx5_spec as spec

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "third_party" / "arx5-sdk" / "models" / "X5.urdf"
X5A_DIR = REPO / "third_party" / "arx-model" / "X5" / "X5A"
X5A_URDF = X5A_DIR / "urdf" / "X5A.urdf"


def test_joint_names_disjoint_and_unique():
    assert not set(spec.GO2_LEG_JOINTS) & set(spec.ARX5_ARM_JOINTS)
    names = spec.GO2_ARX5_EMBODIMENT.joint_names
    assert len(names) == 20
    assert len(set(names)) == 20


def test_embodiment_shape():
    emb = spec.GO2_ARX5_EMBODIMENT
    assert emb.checkpoint_prefix == "go2_arx5"
    assert list(emb.joint_groups) == ["legs", "arm", "gripper"]
    assert len(emb.joint_groups["legs"]) == 12
    assert emb.joint_groups["arm"] == tuple(f"joint{i}" for i in range(1, 7))
    assert (
        emb.joint_groups["gripper"] == ("joint7", "joint8") == spec.GRIPPER_JOINT_NAMES
    )


def test_gripper_constants_and_stowed_pose():
    assert spec.GRIPPER_OPEN_M == 0.044 and spec.GRIPPER_CLOSED_M == 0.0
    assert spec.TCP_BODY_NAME == "eef_link" == spec.ARX5_EEF_BODY
    for j in spec.GRIPPER_JOINT_NAMES:
        assert spec.ARX5_STOWED_JOINT_POS[j] == spec.GRIPPER_OPEN_M
    for j in spec.ARX5_ARM_JOINTS:
        assert spec.ARX5_STOWED_JOINT_POS[j] == 0.0
    assert set(spec.ARX5_STOWED_JOINT_POS) == {
        *spec.ARX5_ARM_JOINTS,
        *spec.GRIPPER_JOINT_NAMES,
    }


def test_finger_specs():
    assert set(spec.ARX5_FINGER_SPECS) == set(spec.GRIPPER_JOINT_NAMES)
    f7, f8 = spec.ARX5_FINGER_SPECS["joint7"], spec.ARX5_FINGER_SPECS["joint8"]
    assert (f7.lower_m, f7.upper_m) == (0.0, 0.044) == (f8.lower_m, f8.upper_m)
    assert f7.axis == (0.0, 1.0, 0.0) and f8.axis == (0.0, -1.0, 0.0)
    assert f7.mass_kg == f8.mass_kg == 0.0648
    assert spec.GRIPPER_ACTUATOR_GAINS.effort_limit == 50.0
    assert (
        spec.GRIPPER_ACTUATOR_GAINS.stiffness > 0
        and spec.GRIPPER_ACTUATOR_GAINS.damping > 0
    )


def test_arm_tables_complete_and_stowed_in_limits():
    for j in spec.ARX5_ARM_JOINTS:
        lo, hi = spec.ARX5_JOINT_LIMITS_RAD[j]
        assert j in spec.ARX5_ACTUATOR_GAINS
        assert lo <= spec.ARX5_STOWED_JOINT_POS[j] <= hi


def test_gains():
    assert set(spec.ARX5_ACTUATOR_GAINS) == set(spec.ARX5_ARM_JOINTS)
    for g in spec.ARX5_ACTUATOR_GAINS.values():
        assert (
            g.stiffness > 0 and g.damping > 0 and g.armature > 0 and g.effort_limit > 0
        )


def test_masses_and_mount():
    assert sum(spec.ARX5_LINK_MASSES_KG.values()) == pytest.approx(3.454, abs=0.001)
    assert sum(spec.ARX5_SDK_X5_LINK_MASSES_KG.values()) == pytest.approx(
        3.52, abs=0.01
    )
    assert spec.GO2_ARX5_MOUNT_XYZ == (0.0, 0.0, 0.07)
    assert spec.GO2_ARX5_MOUNT_RPY_DEG == (0.0, 0.0, 0.0)


def test_usd_path(monkeypatch):
    monkeypatch.delenv(spec.GO2_ARX5_USD_ENV, raising=False)
    assert spec.fused_usd_path() == spec.default_fused_usd_path()
    assert (
        spec.default_fused_usd_path()
        .as_posix()
        .endswith("data/assets/go2_arx5/go2_arx5.usd")
    )
    monkeypatch.setenv(spec.GO2_ARX5_USD_ENV, "/tmp/x.usd")
    assert spec.fused_usd_path() == Path("/tmp/x.usd")


def test_entry_points():
    for emb in (spec.GO2_EMBODIMENT, spec.ARX5_EMBODIMENT, spec.GO2_ARX5_EMBODIMENT):
        mod, sep, attr = emb.articulation_cfg_entry_point.partition(":")
        assert sep and mod and attr
    mod = spec.GO2_ARX5_ARTICULATION_ENTRY_POINT.partition(":")[0]
    assert (REPO / (mod.replace(".", "/") + ".py")).is_file()


def test_cfg_module_defines_top_level_name():
    src = (Path(__file__).parent.parent / "go2_arx5_cfg.py").read_text()
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    assert "GO2_ARX5_CFG" in names


def test_matches_sdk_x5_urdf():
    if not URDF.exists():
        pytest.skip("X5.urdf not fetched")
    root = ET.parse(URDF).getroot()
    joints = {
        j.get("name"): j
        for j in root.findall("joint")
        if j.get("type") in ("revolute", "continuous")
    }
    assert set(joints) == set(spec.ARX5_ARM_JOINTS)
    for name, j in joints.items():
        lim = j.find("limit")
        lo, hi = spec.ARX5_JOINT_LIMITS_RAD[name]
        assert float(lim.get("lower")) == pytest.approx(lo, abs=1e-6)
        assert float(lim.get("upper")) == pytest.approx(hi, abs=1e-6)
    masses = {}
    for link in root.findall("link"):
        m = link.find("inertial/mass")
        if m is not None and float(m.get("value")) > 0:
            masses[link.get("name")] = float(m.get("value"))
    assert set(masses) == set(spec.ARX5_SDK_X5_LINK_MASSES_KG)
    for k, v in masses.items():
        assert v == pytest.approx(spec.ARX5_SDK_X5_LINK_MASSES_KG[k], abs=1e-6)


def test_matches_x5a_urdf():
    if not X5A_URDF.exists():
        pytest.skip("X5A.urdf not fetched")
    root = ET.parse(X5A_URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert set(joints) == {*spec.ARX5_ARM_JOINTS, *spec.GRIPPER_JOINT_NAMES}
    for name, finger in spec.ARX5_FINGER_SPECS.items():
        j = joints[name]
        assert j.get("type") == "prismatic"
        assert j.find("mimic") is None
        assert tuple(float(v) for v in j.find("axis").get("xyz").split()) == finger.axis
        lim = j.find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == (
            finger.lower_m,
            finger.upper_m,
        )
        assert j.find("child").get("link") == finger.link
    masses = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in root.findall("link")
    }
    assert set(masses) == set(spec.ARX5_LINK_MASSES_KG)
    for k, v in masses.items():
        assert v == pytest.approx(spec.ARX5_LINK_MASSES_KG[k], abs=1e-6)
    assert sum(masses.values()) == pytest.approx(3.454, abs=1e-3)
    assert masses["link7"] == pytest.approx(
        spec.ARX5_FINGER_SPECS["joint7"].mass_kg, abs=1e-4
    )


def _stl_vertices(path):
    """Vertices of a binary STL (little-endian float32 triangles)."""
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    assert len(data) == 84 + 50 * count, f"{path} is not a binary STL"
    verts = []
    for i in range(count):
        f = struct.unpack_from("<12fH", data, 84 + 50 * i)
        verts += [f[3:6], f[6:9], f[9:12]]
    return verts


def test_tcp_offset_matches_x5a_geometry():
    """Recompute GO2_ARX5_TCP_OFFSET_XYZ: midpoint of the two fingertip-pad bbox centres, in the link6 frame."""
    if not X5A_URDF.exists():
        pytest.skip("X5A.urdf not fetched")
    root = ET.parse(X5A_URDF).getroot()
    centres = []
    for name, finger in spec.ARX5_FINGER_SPECS.items():
        joint = next(j for j in root.findall("joint") if j.get("name") == name)
        assert joint.find("parent").get("link") == "link6"
        origin = tuple(float(v) for v in joint.find("origin").get("xyz").split())
        assert joint.find("origin").get("rpy").split() == ["0", "0", "0"]
        # the constants stored in the spec are the URDF numbers
        assert origin == pytest.approx(spec.ARX5_FINGER_JOINT_ORIGINS_LINK6[name])
        verts = _stl_vertices(X5A_DIR / "meshes" / f"{finger.link}.STL")
        x_max = max(v[0] for v in verts)
        pad = [v for v in verts if v[0] >= x_max - spec.ARX5_FINGER_PAD_DEPTH_M]
        assert pad
        centres.append(
            [
                origin[k] + (min(v[k] for v in pad) + max(v[k] for v in pad)) / 2
                for k in range(3)
            ]
        )
    midpoint = [(a + b) / 2 for a, b in zip(*centres)]
    # spec value is the midpoint rounded to 0.1 mm
    assert spec.GO2_ARX5_TCP_OFFSET_XYZ == pytest.approx(midpoint, abs=1e-4)
    # sanity: pads touch on the centre line when closed, tip is beyond the link6 mesh end (x = 0.084)
    assert abs(midpoint[1]) < 1e-3
    assert midpoint[0] > 0.084


def test_tcp_offset_is_plain_metres_tuple():
    assert (
        isinstance(spec.GO2_ARX5_TCP_OFFSET_XYZ, tuple)
        and len(spec.GO2_ARX5_TCP_OFFSET_XYZ) == 3
    )


def test_implicit_actuator_kwargs_detects_effort_field_name():
    from dataclasses import dataclass, field

    @dataclass
    class OldLab:  # Isaac Lab 2.3.2 naming
        joint_names_expr: list = field(default_factory=list)
        stiffness: float = 0.0
        damping: float = 0.0
        armature: float | None = None
        effort_limit_sim: float | None = None

    @dataclass
    class NewLab:  # Isaac Lab 3.0-line naming, no armature field
        joint_names_expr: list = field(default_factory=list)
        stiffness: float = 0.0
        damping: float = 0.0
        joint_effort_limit: float | None = None

    old = spec.implicit_actuator_kwargs(
        OldLab, ["joint1"], 50.0, 3.0, 20.0, armature=0.01
    )
    assert old == {
        "joint_names_expr": ["joint1"],
        "stiffness": 50.0,
        "damping": 3.0,
        "effort_limit_sim": 20.0,
        "armature": 0.01,
    }
    assert OldLab(**old).effort_limit_sim == 20.0
    new = spec.implicit_actuator_kwargs(
        NewLab, ("joint7", "joint8"), 1e3, 1e2, 50.0, armature=0.01
    )
    assert new == {
        "joint_names_expr": ["joint7", "joint8"],
        "stiffness": 1e3,
        "damping": 1e2,
        "joint_effort_limit": 50.0,
    }
    assert NewLab(**new).joint_effort_limit == 50.0


def test_implicit_actuator_kwargs_errors():
    from dataclasses import dataclass

    @dataclass
    class NoEffort:
        stiffness: float = 0.0

    with pytest.raises(ValueError, match="effort-limit"):
        spec.implicit_actuator_kwargs(NoEffort, ["j"], 1.0, 1.0, 1.0)
    with pytest.raises(TypeError):
        spec.implicit_actuator_kwargs(dict, ["j"], 1.0, 1.0, 1.0)


def test_cfg_module_exports_and_lazy_isaaclab_imports():
    src = (Path(__file__).parent.parent / "go2_arx5_cfg.py").read_text()
    tree = ast.parse(src)
    names = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    } | {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"GO2_ARX5_CFG", "GO2_ARX5_FIXED_BASE_CFG", "make_go2_arx5_cfg"} <= names
    assert "fix_root_link" in src


def test_default_fused_usd_structure():
    pytest.importorskip("pxr")
    usd_path = spec.fused_usd_path()
    if not usd_path.is_file():
        pytest.skip(f"fused USD not built: {usd_path}")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    assert default and default.GetPath().pathString == "/go2_arx5"
    root = default.GetPath()
    base = stage.GetPrimAtPath(root.AppendPath(spec.GO2_ARX5_BASE_PRIM_REL_PATH))
    tcp = stage.GetPrimAtPath(root.AppendPath(spec.GO2_ARX5_TCP_PRIM_REL_PATH))
    assert base and tcp
    assert tcp.GetName() == spec.TCP_BODY_NAME
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) == 1
    joints = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    assert set(spec.GO2_ARX5_EMBODIMENT.joint_names) <= joints
    assert len(spec.GO2_ARX5_EMBODIMENT.joint_names) == 20
    arm_base = stage.GetPrimAtPath(base.GetPath().AppendChild("base_link"))
    assert arm_base
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    z_min = cache.ComputeWorldBound(arm_base).ComputeAlignedRange().GetMin()[2]
    assert z_min == pytest.approx(spec.GO2_ARX5_MOUNT_XYZ[2], abs=1e-3)
