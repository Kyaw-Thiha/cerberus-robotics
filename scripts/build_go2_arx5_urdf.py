#!/usr/bin/env python3
"""Builds the fused Go2 + ARX5 (X5A) robot as ONE URDF (local, pure python, no Isaac).

Flow: expand the ROS xacro Go2 with `sim:=true` (`common/cerberus_description/xacro_expand.py`, no ament index
needed) -> patch the fetched X5A URDF (`urdf_patch.patch_arm_urdf`: absolute meshes, `eef_link`, spec limits) -> merge
both into `<robot name="go2_arx5">` with a fixed `arm_mount_joint` from Go2 `base` to the arm `base_link` at the spec
mount -> validate -> write the URDF and `<out>.build.json`. The result is the input of the single UrdfConverter run in
`scripts/convert_go2_arx5_urdf_to_usd.py`.

Contract with the Go2 xacro (`sim:=true`): root link `base` carrying the trunk inertial, no `world`/floating joint,
no rotor/sensor links, no gazebo/transmission output, absolute mesh paths, leg links `{FL,FR,RL,RR}_{hip,thigh,calf,
foot}`, leg joints `{leg}_{hip,thigh,calf}_joint`.

Exit codes: 0 ok, 1 validation failed, 2 bad args / missing inputs / refusing to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fetch_arx5_assets import DEFAULT_VARIANT, VARIANTS, arm_urdf_path

from common.cerberus_description.go2_arx5_spec import (
    GO2_ARX5_MOUNT_RPY_DEG,
    GO2_ARX5_MOUNT_XYZ,
    GO2_ROOT_BODY,
    TCP_BODY_NAME,
)
from common.cerberus_description.urdf_patch import expected_joint_names, patch_arm_urdf
from common.cerberus_description.xacro_expand import expand_xacro

ROBOT_NAME = "go2_arx5"
ARM_BASE_LINK = "base_link"
MOUNT_JOINT_NAME = "arm_mount_joint"
DEFAULT_GO2_XACRO = (
    REPO_ROOT / "common" / "cerberus_description" / "urdf" / "go2.urdf.xacro"
)
DEFAULT_OUT = REPO_ROOT / "data" / "assets" / "go2_arx5" / "urdf" / "go2_arx5.urdf"
ACTUATED_JOINT_TYPES = ("revolute", "continuous", "prismatic")
EXIT_OK, EXIT_VALIDATION_FAIL, EXIT_BAD_ARGS = 0, 1, 2


# ---------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"merged URDF path (default {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=VARIANTS,
        help="arm variant (default X5A)",
    )
    parser.add_argument(
        "--go2-xacro",
        type=Path,
        default=DEFAULT_GO2_XACRO,
        help="Go2 xacro (default: repo xacro)",
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="debug: expand the xacro defaults instead of sim:=true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan as JSON and exit 0",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing output"
    )
    return parser


def build_record_path(out: Path) -> Path:
    """`<out>.build.json`, e.g. go2_arx5.urdf -> go2_arx5.urdf.build.json."""
    return out.with_name(out.name + ".build.json")


def mount_rpy_rad(rpy_deg: Sequence[float] = GO2_ARX5_MOUNT_RPY_DEG) -> list[float]:
    return [math.radians(float(v)) for v in rpy_deg]


def build_plan(args: argparse.Namespace) -> dict:
    """Every resolved input/output/transform, without expanding or reading any URDF."""
    out = Path(args.out).expanduser().resolve()
    try:
        arm_urdf: Path | None = arm_urdf_path(args.variant).resolve()
    except FileNotFoundError:
        arm_urdf = None
    return {
        "variant": args.variant,
        "go2_xacro": str(Path(args.go2_xacro).expanduser().resolve()),
        "xacro_args": {} if args.no_sim else {"sim": "true"},
        "arm_urdf": str(arm_urdf) if arm_urdf is not None else None,
        "arm_urdf_present": arm_urdf is not None,
        "out": str(out),
        "record": str(build_record_path(out)),
        "robot_name": ROBOT_NAME,
        "root_link": None if args.no_sim else GO2_ROOT_BODY,
        "mount_joint": MOUNT_JOINT_NAME,
        "mount_parent_link": GO2_ROOT_BODY,
        "mount_child_link": ARM_BASE_LINK,
        "mount_xyz": [float(v) for v in GO2_ARX5_MOUNT_XYZ],
        "mount_rpy_deg": [float(v) for v in GO2_ARX5_MOUNT_RPY_DEG],
        "mount_rpy_rad": mount_rpy_rad(),
        "tcp_body": TCP_BODY_NAME,
        "expected_joint_names": list(expected_joint_names()),
        "force": bool(args.force),
    }


def check_overwrite(paths: Iterable[Path], force: bool) -> None:
    """Refuse to overwrite an existing output (URDF or sidecar) unless --force."""
    for path in paths:
        if Path(path).exists() and not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(values: Iterable[float]) -> str:
    return " ".join(repr(float(v)) for v in values)


def _names(root: ET.Element, tag: str) -> list[str]:
    return [e.get("name") for e in root.findall(tag)]


def _duplicates(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for n in names:
        if n in seen and n not in dup:
            dup.append(n)
        seen.add(n)
    return dup


def parse_robot(text_or_path: str | Path, label: str) -> ET.Element:
    """Parse a URDF from an XML string or a path and check that the root is `<robot>`."""
    if isinstance(text_or_path, Path):
        root = ET.parse(text_or_path).getroot()
    else:
        root = ET.fromstring(text_or_path)
    if root.tag != "robot":
        raise ValueError(f"{label}: root element is <{root.tag}>, expected <robot>")
    return root


def link_mass(link: ET.Element) -> float:
    mass = link.find("inertial/mass")
    return float(mass.get("value", "0")) if mass is not None else 0.0


def total_mass(root: ET.Element) -> float:
    return sum(link_mass(link) for link in root.findall("link"))


def merge_urdfs(
    go2: ET.Element,
    arm: ET.Element,
    mount_xyz: Sequence[float] = GO2_ARX5_MOUNT_XYZ,
    mount_rpy_radians: Sequence[float] | None = None,
    robot_name: str = ROBOT_NAME,
    parent_link: str = GO2_ROOT_BODY,
    child_link: str = ARM_BASE_LINK,
    mount_joint_name: str = MOUNT_JOINT_NAME,
) -> tuple[ET.Element, dict]:
    """Merge two parsed `<robot>` elements and add the fixed mount joint. Returns (merged robot, info).

    Materials are deduplicated by name (first definition wins; a differing redefinition is reported in
    `info['material_conflicts']`). Any link or joint name present in both robots, or equal to the mount joint name,
    raises ValueError. Neither input is modified.
    """
    if mount_rpy_radians is None:
        mount_rpy_radians = mount_rpy_rad()
    go2_links, arm_links = _names(go2, "link"), _names(arm, "link")
    go2_joints, arm_joints = _names(go2, "joint"), _names(arm, "joint")
    for what, names in (
        ("Go2 links", go2_links),
        ("arm links", arm_links),
        ("Go2 joints", go2_joints),
        ("arm joints", arm_joints),
    ):
        dup = _duplicates(names)
        if dup:
            raise ValueError(f"duplicate names inside {what}: {dup}")
    link_clash = sorted(set(go2_links) & set(arm_links))
    joint_clash = sorted(
        (set(go2_joints) & set(arm_joints))
        | ({mount_joint_name} & (set(go2_joints) | set(arm_joints)))
    )
    if link_clash or joint_clash:
        raise ValueError(
            f"name collision between Go2 and arm: links={link_clash}, joints={joint_clash}"
        )
    if parent_link not in go2_links:
        raise ValueError(f"Go2 URDF has no mount parent link {parent_link!r}")
    if child_link not in arm_links:
        raise ValueError(f"arm URDF has no mount child link {child_link!r}")

    merged = ET.Element("robot", name=robot_name)
    materials: dict[str, ET.Element] = {}
    conflicts: list[str] = []
    for src in (go2, arm):
        for material in src.findall("material"):
            name = material.get("name")
            if name in materials:
                if ET.tostring(materials[name]) != ET.tostring(material):
                    conflicts.append(name)
                continue
            materials[name] = material
    for material in materials.values():
        merged.append(_copy(material))
    for src in (go2, arm):
        for link in src.findall("link"):
            merged.append(_copy(link))
    for joint in go2.findall("joint"):
        merged.append(_copy(joint))
    mount = ET.SubElement(merged, "joint", name=mount_joint_name, type="fixed")
    ET.SubElement(mount, "origin", xyz=_fmt(mount_xyz), rpy=_fmt(mount_rpy_radians))
    ET.SubElement(mount, "parent", link=parent_link)
    ET.SubElement(mount, "child", link=child_link)
    for joint in arm.findall("joint"):
        merged.append(_copy(joint))
    # Anything else (gazebo/transmission/... in `--no-sim` debug expansions) is carried over untouched.
    handled = {"link", "joint", "material"}
    for src in (go2, arm):
        for child in src:
            if isinstance(child.tag, str) and child.tag not in handled:
                merged.append(_copy(child))
    info = {
        "go2_links": len(go2_links),
        "arm_links": len(arm_links),
        "go2_joints": len(go2_joints),
        "arm_joints": len(arm_joints),
        "materials": len(materials),
        "material_conflicts": sorted(set(conflicts)),
    }
    return merged, info


def _copy(element: ET.Element) -> ET.Element:
    clone = ET.Element(element.tag, dict(element.attrib))
    clone.text, clone.tail = element.text, None
    for child in element:
        if isinstance(child.tag, str):
            clone.append(_copy(child))
    return clone


def find_root_links(root: ET.Element) -> list[str]:
    children = {j.find("child").get("link") for j in root.findall("joint")}
    return [n for n in _names(root, "link") if n not in children]


def mesh_files(root: ET.Element) -> list[str]:
    return [
        m.get("filename") for m in root.iter("mesh") if m.get("filename") is not None
    ]


def validate_merged(
    root: ET.Element,
    expected_joints: Iterable[str] | None = None,
    expected_root: str | None = GO2_ROOT_BODY,
    tcp_link: str = TCP_BODY_NAME,
    mount_joint_name: str = MOUNT_JOINT_NAME,
) -> dict:
    """Validate the merged robot; raise ValueError listing every failure, else return the check results.

    `expected_root=None` only requires exactly one root link (used by the `--no-sim` debug expansion).
    """
    expected = list(
        expected_joint_names() if expected_joints is None else expected_joints
    )
    errors: list[str] = []
    links = _names(root, "link")
    joints = root.findall("joint")
    link_set = set(links)
    for what, names in (("link", links), ("joint", _names(root, "joint"))):
        dup = _duplicates(names)
        if dup:
            errors.append(f"duplicate {what} names: {dup}")

    parents: dict[str, list[str]] = {}
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            errors.append(f"joint {joint.get('name')} has no parent/child")
            continue
        for end in (parent.get("link"), child.get("link")):
            if end not in link_set:
                errors.append(
                    f"joint {joint.get('name')} references unknown link {end!r}"
                )
        parents.setdefault(child.get("link"), []).append(parent.get("link"))
    multi = sorted(c for c, p in parents.items() if len(p) != 1)
    if multi:
        errors.append(f"links with more than one parent joint: {multi}")
    roots = find_root_links(root)
    if len(roots) != 1:
        errors.append(f"expected exactly one root link, found {roots}")
    elif expected_root is not None and roots[0] != expected_root:
        errors.append(f"root link is {roots[0]!r}, expected {expected_root!r}")
    if (
        roots and len(roots) == 1
    ):  # every link reachable from the root (no cycles / detached loops)
        children_of: dict[str, list[str]] = {}
        for child, plist in parents.items():
            for p in plist:
                children_of.setdefault(p, []).append(child)
        reached, stack = set(), [roots[0]]
        while stack:
            node = stack.pop()
            if node not in reached:
                reached.add(node)
                stack.extend(children_of.get(node, []))
        unreachable = sorted(link_set - reached)
        if unreachable:
            errors.append(f"links not reachable from the root: {unreachable}")

    actuated = [j.get("name") for j in joints if j.get("type") in ACTUATED_JOINT_TYPES]
    missing = [n for n in expected if n not in actuated]
    unexpected = sorted(set(actuated) - set(expected))
    if missing or unexpected:
        errors.append(
            f"actuated joints differ from the contract: missing={missing}, unexpected={unexpected}"
        )
    if tcp_link not in link_set:
        errors.append(f"TCP link {tcp_link!r} is missing")
    mount = next((j for j in joints if j.get("name") == mount_joint_name), None)
    if mount is None or mount.get("type") != "fixed":
        errors.append(f"fixed mount joint {mount_joint_name!r} is missing")

    bad_meshes: list[str] = []
    for filename in mesh_files(root):
        path = Path(filename.removeprefix("file://"))
        if not path.is_absolute() or not path.is_file():
            bad_meshes.append(filename)
    if bad_meshes:
        errors.append(
            f"{len(bad_meshes)} mesh path(s) not absolute or not on disk, e.g. {bad_meshes[:3]}"
        )

    if errors:
        raise ValueError("merged URDF validation failed:\n  - " + "\n  - ".join(errors))
    return {
        "root_link": roots[0],
        "links": len(links),
        "joints": len(joints),
        "actuated_joints": actuated,
        "mount_joint": mount_joint_name,
        "tcp_link": tcp_link,
        "meshes_checked": len(mesh_files(root)),
        "total_mass_kg": total_mass(root),
        "ok": True,
    }


def write_urdf(root: ET.Element, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)


def framework_versions() -> dict:
    versions = {"python": platform.python_version()}
    try:
        from importlib import metadata

        versions["xacro"] = metadata.version("xacro")
    except Exception:  # noqa: BLE001 - version is informational only
        versions["xacro"] = None
    return versions


# ---------------------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------------------


def run_build(plan: dict) -> dict:
    """Expand, patch, merge, validate and write. Returns the sidecar record. Raises on any failure."""
    out, record = Path(plan["out"]), Path(plan["record"])
    check_overwrite((out, record), plan["force"])
    if not plan["arm_urdf_present"]:
        raise FileNotFoundError(
            f"ARX5 URDF for {plan['variant']} not found; run `python scripts/fetch_arx5_assets.py` to fetch it"
        )
    go2_xacro, arm_urdf = Path(plan["go2_xacro"]), Path(plan["arm_urdf"])
    if not go2_xacro.is_file():
        raise FileNotFoundError(f"--go2-xacro not found: {go2_xacro}")

    go2_xml = expand_xacro(go2_xacro, args=plan["xacro_args"])
    go2 = parse_robot(go2_xml, "expanded Go2 xacro")
    with tempfile.TemporaryDirectory(prefix="go2_arx5_arm_") as tmp:
        patched = patch_arm_urdf(arm_urdf, Path(tmp) / arm_urdf.name)
        patched_text = patched.read_text(encoding="utf-8")
        arm = parse_robot(patched, "patched arm URDF")

    merged, merge_info = merge_urdfs(
        go2, arm, plan["mount_xyz"], plan["mount_rpy_rad"], plan["robot_name"]
    )
    checks = validate_merged(merged, expected_root=plan["root_link"])
    write_urdf(merged, out)
    masses = {
        "go2_kg": total_mass(go2),
        "arm_kg": total_mass(arm),
        "total_kg": checks["total_mass_kg"],
    }
    result = {
        **plan,
        "inputs_sha256": {
            "go2_xacro_dir": {
                str(p.relative_to(go2_xacro.parent)): sha256_file(p)
                for p in sorted(go2_xacro.parent.rglob("*"))
                if p.is_file() and p.suffix in (".xacro", ".urdf")
            },
            "go2_expanded": sha256_text(go2_xml),
            "arm_urdf": sha256_file(arm_urdf),
            "arm_patched": sha256_text(patched_text),
        },
        "output_sha256": sha256_file(out),
        "merge": merge_info,
        "masses": masses,
        "checks": checks,
        "versions": framework_versions(),
    }
    record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return EXIT_OK
    try:
        result = run_build(plan)
    except (FileNotFoundError, FileExistsError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_ARGS
    except (ValueError, ET.ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_VALIDATION_FAIL
    masses = result["masses"]
    print(f"wrote {result['out']}")
    print(f"wrote {result['record']}")
    print(
        f"links={result['checks']['links']} joints={result['checks']['joints']} "
        f"dynamic mass: go2={masses['go2_kg']:.3f} kg arm={masses['arm_kg']:.3f} kg total={masses['total_kg']:.3f} kg"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
