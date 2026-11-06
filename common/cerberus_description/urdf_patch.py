"""Helpers for building the fused Go2 + ARX5 robot (arm URDF patching, joint-name contract).

Pure python: no simulation or USD library is imported anywhere in this module.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from common.cerberus_description.go2_arx5_spec import (
    ARX5_ACTUATOR_GAINS,
    ARX5_ARM_JOINTS,
    ARX5_FINGER_SPECS,
    ARX5_JOINT_LIMITS_RAD,
    ARX5_JOINT_VELOCITY_LIMIT_RAD_S,
    GO2_ARX5_EMBODIMENT,
    GO2_ARX5_TCP_OFFSET_XYZ,
    TCP_BODY_NAME,
)

logger = logging.getLogger(__name__)

EEF_LINK_NAME = TCP_BODY_NAME
TCP_PARENT_LINK = "link6"
TCP_JOINT_NAME = "tcp_fixed_joint"
# Small isotropic inertia (kg*m^2) given to the patched end-effector link.
EEF_INERTIA_KG_M2 = 1e-7
X5A_ROBOT_NAME = "X5A"


def expected_joint_names() -> tuple[str, ...]:
    """All 20 actuated joint names of the fused robot (12 legs, joint1..6, then joint7/joint8 fingers)."""
    return tuple(GO2_ARX5_EMBODIMENT.joint_names)


def validate_joint_names(found: Iterable[str]) -> None:
    """Raise ValueError listing missing and unexpected names.

    `found` must contain only ACTUATED joints (revolute/prismatic); fixed joints, including the Go2 foot joints
    and the Go2->arm mount joint, are not part of the expected set.
    """
    found_set = set(found)
    expected = expected_joint_names()
    expected_set = set(expected)
    missing = [n for n in expected if n not in found_set]
    unexpected = sorted(found_set - expected_set)
    if missing or unexpected:
        raise ValueError(
            f"fused robot joint names do not match the contract: missing={missing}, unexpected={unexpected}"
        )


def _set_eef_inertial(link: ET.Element, mass_kg: float) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
    for tag in ("origin", "mass", "inertia"):
        for child in inertial.findall(tag):
            inertial.remove(child)
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=repr(float(mass_kg)))
    i = repr(EEF_INERTIA_KG_M2)
    ET.SubElement(inertial, "inertia", ixx=i, ixy="0", ixz="0", iyy=i, iyz="0", izz=i)


def _find_mesh_fallback(src: Path, base_dir: Path, filename: str, tried: Path) -> Path:
    """Repair a wrong upstream mesh path: unique basename match under `<src dir>/meshes`, else FileNotFoundError."""
    meshes_dir = base_dir / "meshes"
    candidates = (
        sorted(p for p in meshes_dir.rglob(Path(filename).name) if p.is_file())
        if meshes_dir.is_dir()
        else []
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"mesh referenced by {src} not found: {tried}; fallback search for {Path(filename).name!r} under "
            f"{meshes_dir} found {len(candidates)} candidate(s): {[str(c) for c in candidates]}"
        )
    repaired = candidates[0].resolve()
    logger.warning(
        "repaired upstream mesh path in %s: %s -> %s", src, filename, repaired
    )
    return repaired


def _resolve_package_uri(src: Path, filename: str) -> Path:
    """`package://<pkg>/<rel>` -> `<ancestor of src named <pkg>>/<rel>` (ROS package root = the dir named <pkg>)."""
    pkg, _, rel = filename[len("package://") :].partition("/")
    for parent in src.resolve().parents:
        if parent.name == pkg:
            return parent / rel
    raise ValueError(
        f"package:// mesh path {filename} cannot be resolved: no ancestor directory of {src} is named {pkg!r}"
    )


def _is_x5a(root: ET.Element) -> bool:
    return root.get("name") == X5A_ROBOT_NAME


def _fmt(value: float) -> str:
    return repr(float(value))


def _patch_x5a_joints(root: ET.Element) -> None:
    """Replace the X5A placeholder limits/effort/velocity with the spec values (joint1..6 and joint7/8)."""
    joints = {j.get("name"): j for j in root.findall("joint")}
    wanted: dict[str, dict[str, float]] = {}
    for name in ARX5_ARM_JOINTS:
        lo, hi = ARX5_JOINT_LIMITS_RAD[name]
        wanted[name] = {
            "lower": lo,
            "upper": hi,
            "effort": ARX5_ACTUATOR_GAINS[name].effort_limit,
            "velocity": ARX5_JOINT_VELOCITY_LIMIT_RAD_S,
        }
    for name, finger in ARX5_FINGER_SPECS.items():
        wanted[name] = {
            "lower": finger.lower_m,
            "upper": finger.upper_m,
            "effort": finger.effort_n,
            "velocity": finger.velocity_m_s,
        }
    for name, values in wanted.items():
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"X5A URDF has no joint {name!r}")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        for key, value in values.items():
            new = _fmt(value)
            old = limit.get(key)
            if old is None or float(old) != float(new):
                logger.info("joint %s: limit %s %s -> %s", name, key, old, new)
            limit.set(key, new)


def _ensure_tcp(root: ET.Element, eef_mass_kg: float) -> None:
    """X5A has no eef_link: add it as a fixed child of link6 at GO2_ARX5_TCP_OFFSET_XYZ (idempotent)."""
    links = {link.get("name"): link for link in root.findall("link")}
    if TCP_PARENT_LINK not in links:
        raise ValueError(
            f"URDF has no link {TCP_PARENT_LINK!r} to attach {EEF_LINK_NAME!r} to"
        )
    xyz = " ".join(_fmt(v) for v in GO2_ARX5_TCP_OFFSET_XYZ)
    if EEF_LINK_NAME not in links:
        ET.SubElement(root, "link", name=EEF_LINK_NAME)
        logger.info("added link %s", EEF_LINK_NAME)
    existing = next(
        (
            j
            for j in root.findall("joint")
            if j.find("child").get("link") == EEF_LINK_NAME
        ),
        None,
    )
    if existing is None:
        joint = ET.SubElement(root, "joint", name=TCP_JOINT_NAME, type="fixed")
        ET.SubElement(joint, "origin", xyz=xyz, rpy="0 0 0")
        ET.SubElement(joint, "parent", link=TCP_PARENT_LINK)
        ET.SubElement(joint, "child", link=EEF_LINK_NAME)
        logger.info(
            "added fixed joint %s: %s -> %s at xyz=%s",
            TCP_JOINT_NAME,
            TCP_PARENT_LINK,
            EEF_LINK_NAME,
            xyz,
        )
    elif existing.get("name") == TCP_JOINT_NAME:
        origin = existing.find("origin")
        if origin is None:
            origin = ET.SubElement(existing, "origin", rpy="0 0 0")
        origin.set("xyz", xyz)
        origin.set("rpy", "0 0 0")


def _ensure_nonzero_inertia(root: ET.Element, fallback_mass_kg: float) -> None:
    """Every link needs a positive mass and positive diagonal inertia, else the converter emits an invalid body."""
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        mass_el = inertial.find("mass") if inertial is not None else None
        inertia_el = inertial.find("inertia") if inertial is not None else None
        mass = float(mass_el.get("value", "0")) if mass_el is not None else 0.0
        diag = (
            [float(inertia_el.get(k, "0")) for k in ("ixx", "iyy", "izz")]
            if inertia_el is not None
            else [0.0]
        )
        if mass > 0 and inertia_el is not None and all(d > 0 for d in diag):
            continue
        keep_mass = mass if mass > 0 else fallback_mass_kg
        logger.info(
            "link %s: mass %s / inertia %s is degenerate, using mass %s and isotropic %s",
            name,
            mass,
            diag,
            keep_mass,
            EEF_INERTIA_KG_M2,
        )
        _set_eef_inertial(link, keep_mass)


def patch_arm_urdf(src: Path, dst: Path, eef_mass_kg: float = 1e-3) -> Path:
    """Write a converter-friendly copy of the arm URDF to `dst` (never modifies `src`).

    All variants:
    - the massless `eef_link` gets mass `eef_mass_kg` and a small isotropic inertia, so the converter does not
      emit an invalid (-inf) centre of mass (as in the LeggedManip_Lab USD);
    - every mesh `filename` becomes an absolute path resolved relative to `src.parent`; a missing mesh raises
      FileNotFoundError.

    Only for ARX's X5A model (robot name `X5A`), additionally:
    - `package://X5A/meshes/<f>` is resolved to an absolute file under the package's `meshes/`;
    - `eef_link` is appended as a fixed child of `link6` through `tcp_fixed_joint` at GO2_ARX5_TCP_OFFSET_XYZ;
    - joint1..6 get the spec limits/effort/velocity (X5A ships +-10 rad and 100/1000 placeholders); joint7/joint8
      keep 0..0.044 m and get the spec finger effort/velocity;
    - every link is guaranteed a positive mass and positive diagonal inertia.

    Every change is logged at INFO. Patching an already patched file is stable.
    """
    if eef_mass_kg <= 0:
        raise ValueError(f"eef_mass_kg must be positive, got {eef_mass_kg}")
    src = Path(src)
    dst = Path(dst)
    tree = ET.parse(src)
    root = tree.getroot()
    x5a = _is_x5a(root)

    if x5a:
        _patch_x5a_joints(root)
        _ensure_tcp(root, eef_mass_kg)

    eef = next(
        (link for link in root.iter("link") if link.get("name") == EEF_LINK_NAME), None
    )
    if eef is None:
        raise ValueError(f"{src} has no link named {EEF_LINK_NAME!r}")
    _set_eef_inertial(eef, eef_mass_kg)
    if x5a:
        _ensure_nonzero_inertia(root, eef_mass_kg)

    base_dir = src.resolve().parent
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        if filename.startswith("package://"):
            if not x5a:
                raise ValueError(f"package:// mesh paths are not supported: {filename}")
            path = _resolve_package_uri(src, filename)
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = base_dir / path
        path = Path(path.resolve())
        if not path.is_file():
            path = _find_mesh_fallback(
                src, base_dir if not x5a else path.parent.parent, filename, path
            )
        if str(path) != filename:
            logger.info("mesh %s -> %s", filename, path)
        mesh.set("filename", str(path))

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst
