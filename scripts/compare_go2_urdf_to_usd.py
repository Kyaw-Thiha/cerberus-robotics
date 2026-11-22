#!/usr/bin/env python3
"""Offline comparison of the Go2 body in a URDF against the Go2 body inside a fused USD.

Compares links/joints present in BOTH (arm excluded by name: any link not in the Go2 name set; works for any USD
hierarchy): mass, centre of mass, principal inertia (fails only above BOTH the relative and absolute tolerance), joint
type/limits and parent/child relations. Actuated joints pair by name, fixed joints by (parent, child) so renames are
informational. Light USD-only links (--ignore-usd-only-mass) and their fixed joints are informational, as are joint
axis, effort and velocity differences. Writes a JSON report and prints a short summary. Needs `pxr` (usd-core) and
numpy only; no Isaac.

The default --usd is the legacy `data/assets/go2_arx5/legacy/go2_arx5_assembler.usd`: a Go2-body golden reference
only (stock Go2 USD inertials). Its arm is mounted 0.07 m lower than in the URDF build (an assembler artefact), so
never compare arm poses against it.

Usage: pixi run python scripts/compare_go2_urdf_to_usd.py [--urdf P] [--usd P] [--report P]
Exit codes: 0 PASS, 1 FAIL, 2 bad args / missing inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_URDF = "data/assets/go2_arx5/urdf/go2_arx5.urdf"
DEFAULT_USD = "data/assets/go2_arx5/legacy/go2_arx5_assembler.usd"
DEFAULT_REPORT = "data/reports/go2_urdf_vs_usd.json"

LEGS = ("FL", "FR", "RL", "RR")
GO2_LINK_NAMES = frozenset(
    ["base", "Head_upper", "Head_lower"]
    + [f"{leg}_{p}" for leg in LEGS for p in ("hip", "thigh", "calf", "foot")]
)
GO2_JOINT_NAMES_HINT = frozenset(
    f"{leg}_{p}_joint" for leg in LEGS for p in ("hip", "thigh", "calf", "foot")
)
DEFAULT_THRESHOLDS = {
    "mass_rel": 0.02,
    "com_m": 0.002,
    "inertia_rel": 0.10,
    "inertia_abs": 1e-5,
    "limit_rad": 0.01,
    "ignore_usd_only_mass": 0.005,
}
_EPS = 1e-12


# ----------------------------------------------------------------------------- URDF


def _floats(text, n, default):
    if text is None:
        return list(default)
    vals = [float(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} floats, got {text!r}")
    return vals


def inertia_matrix(ixx, ixy, ixz, iyy, iyz, izz):
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)


def principal_inertia(tensor):
    """Return (eigenvalues ascending, eigenvector columns) of a symmetric 3x3 inertia tensor."""
    vals, vecs = np.linalg.eigh(np.asarray(tensor, dtype=float))
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def parse_urdf(source):
    """Parse URDF text or a path into {'links': {name: {...}}, 'joints': {name: {...}}}."""
    text = source
    if isinstance(source, Path) or (isinstance(source, str) and "<" not in source):
        text = Path(source).read_text()
    root = ET.fromstring(text)
    links = {}
    for el in root.findall("link"):
        entry = {"mass": None, "com": None, "inertia": None}
        inertial = el.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            entry["com"] = _floats(
                origin.get("xyz") if origin is not None else None, 3, (0, 0, 0)
            )
            mass = inertial.find("mass")
            if mass is not None:
                entry["mass"] = float(mass.get("value"))
            inertia = inertial.find("inertia")
            if inertia is not None:
                entry["inertia"] = inertia_matrix(
                    *(
                        float(inertia.get(k, 0.0))
                        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                    )
                )
        links[el.get("name")] = entry
    joints = {}
    for el in root.findall("joint"):
        limit = el.find("limit")
        axis = el.find("axis")
        joints[el.get("name")] = {
            "type": el.get("type"),
            "parent": el.find("parent").get("link"),
            "child": el.find("child").get("link"),
            "axis": _floats(
                axis.get("xyz") if axis is not None else None, 3, (1, 0, 0)
            ),
            "lower": float(limit.get("lower", 0.0)) if limit is not None else None,
            "upper": float(limit.get("upper", 0.0)) if limit is not None else None,
            "effort": float(limit.get("effort"))
            if limit is not None and limit.get("effort") is not None
            else None,
            "velocity": float(limit.get("velocity"))
            if limit is not None and limit.get("velocity") is not None
            else None,
        }
    return {"name": root.get("name"), "links": links, "joints": joints}


def go2_names(urdf):
    """Link names of the Go2 body in the URDF (arm = everything else)."""
    return {n for n in urdf["links"] if n in GO2_LINK_NAMES}


# ----------------------------------------------------------------------------- USD

_AXIS_VEC = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}


def _get(prim, name):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.HasAuthoredValue() else None


def read_usd(stage, names=GO2_LINK_NAMES):
    """Read Go2 bodies and joints from a stage by name: bodies not in `names` and joints whose child body is not in `names` (the arm) are skipped, so any hierarchy works."""
    from pxr import UsdPhysics

    bodies, joints = {}, {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and name in names
            and name not in bodies
        ):
            mass = _get(prim, "physics:mass")
            com = _get(prim, "physics:centerOfMass")
            diag = _get(prim, "physics:diagonalInertia")
            bodies[name] = {
                "path": path,
                "mass": None if mass is None else float(mass),
                "com": None if com is None else [float(c) for c in com],
                "diag": None if diag is None else [float(c) for c in diag],
            }
        elif prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)

            def _target(rel):
                t = rel.GetTargets()
                return t[0].name if t else None

            # Arm = any body not in the Go2 name set: its joints (incl. the mount joint) are out of scope.
            if _target(joint.GetBody1Rel()) not in names:
                continue

            entry = {
                "path": path,
                "type": prim.GetTypeName(),
                "parent": _target(joint.GetBody0Rel()),
                "child": _target(joint.GetBody1Rel()),
                "axis": None,
                "lower_deg": None,
                "upper_deg": None,
                "effort": None,
                "velocity": None,
            }
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(
                UsdPhysics.PrismaticJoint
            ):
                tok = _get(prim, "physics:axis")
                entry["axis"] = tok
                entry["lower_deg"] = _get(prim, "physics:lowerLimit")
                entry["upper_deg"] = _get(prim, "physics:upperLimit")
                for dname in ("angular", "linear"):
                    mf = _get(prim, f"drive:{dname}:physics:maxForce")
                    if mf is not None and math.isfinite(mf) and mf < 1e30:
                        entry["effort"] = float(mf)
                vel = _get(prim, "physxJoint:maxJointVelocity")
                if vel is not None:
                    entry["velocity"] = float(vel)  # deg/s for revolute
            joints.setdefault(prim.GetName(), entry)
    return {"bodies": bodies, "joints": joints}


def usd_joint_type(type_name):
    return {
        "PhysicsRevoluteJoint": "revolute",
        "PhysicsPrismaticJoint": "prismatic",
        "PhysicsFixedJoint": "fixed",
    }.get(type_name, type_name)


# ----------------------------------------------------------------------------- compare


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), _EPS)


def deg_to_rad(x):
    return math.radians(x)


ACTUATED = ("revolute", "prismatic", "continuous")


def body_rows(urdf, usd, thresholds=None):
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = []
    for name in sorted(go2_names(urdf) & set(usd["bodies"])):
        u, d = urdf["links"][name], usd["bodies"][name]
        row = {"name": name, "usd_path": d["path"]}
        if u["mass"] is not None and d["mass"] is not None:
            row.update(
                mass_urdf=u["mass"],
                mass_usd=d["mass"],
                mass_rel_diff=_rel(u["mass"], d["mass"]),
            )
        if u["com"] is not None and d["com"] is not None:
            row.update(
                com_urdf=u["com"],
                com_usd=d["com"],
                com_dist_m=float(
                    np.linalg.norm(np.array(u["com"]) - np.array(d["com"]))
                ),
            )
        if u["inertia"] is not None and d["diag"] is not None:
            ev, _ = principal_inertia(u["inertia"])
            ud = sorted(d["diag"])
            rels = [_rel(a, b) for a, b in zip(ev, ud)]
            abss = [abs(a - b) for a, b in zip(ev, ud)]
            row.update(
                inertia_urdf_principal=ev.tolist(),
                inertia_usd_principal=ud,
                inertia_rel_diff=max(rels),
                inertia_abs_diff=max(abss),
                # Fails only when a component exceeds BOTH the relative and the absolute tolerance.
                inertia_exceeds=any(
                    r > t["inertia_rel"] and a > t["inertia_abs"]
                    for r, a in zip(rels, abss)
                ),
            )
        rows.append(row)
    return rows


def _urdf_scope(urdf):
    names = go2_names(urdf)
    return {
        j: v
        for j, v in urdf["joints"].items()
        if v["child"] in names and v["parent"] in names
    }


def match_joints(urdf, usd, ignore_usd_only_mass=None):
    """Pair URDF and USD joints.

    Actuated joints pair by name; fixed joints pair by (parent link, child link) so renames do not count.
    Returns dict: pairs [(urdf_name, usd_name)], urdf_only, usd_only (failures), fixed_joint_renames and
    ignored_usd_only (informational).
    """
    if ignore_usd_only_mass is None:
        ignore_usd_only_mass = DEFAULT_THRESHOLDS["ignore_usd_only_mass"]
    names = go2_names(urdf)
    usd_bodies = set(usd["bodies"])
    light = {
        n
        for n in usd_bodies - names
        if usd["bodies"][n]["mass"] is not None
        and usd["bodies"][n]["mass"] < ignore_usd_only_mass
    }
    urdf_j = _urdf_scope(urdf)
    usd_j = {
        j: v
        for j, v in usd["joints"].items()
        if v["child"] in usd_bodies or v["child"] in GO2_LINK_NAMES
    }
    pairs, claimed = [], set()
    for j, v in sorted(urdf_j.items()):
        if v["type"] in ACTUATED and j in usd_j:
            pairs.append((j, j))
            claimed.add(j)
    by_conn = {}
    for j, v in sorted(usd_j.items()):
        if j not in claimed:
            by_conn.setdefault((v["parent"], v["child"]), []).append(j)
    renames, urdf_only = [], []
    for j, v in sorted(urdf_j.items()):
        if v["type"] in ACTUATED:
            if j not in claimed:
                urdf_only.append(j)
            continue
        cands = by_conn.get((v["parent"], v["child"]))
        if not cands:
            urdf_only.append(j)
            continue
        # Prefer a same-named candidate; otherwise the first remaining one.
        uj = j if j in cands else cands[0]
        cands.remove(uj)
        claimed.add(uj)
        pairs.append((j, uj))
        if uj != j:
            renames.append(
                {
                    "urdf": j,
                    "usd": uj,
                    "parent": v["parent"],
                    "child": v["child"],
                }
            )
    usd_only, ignored_joints = [], []
    for j, v in sorted(usd_j.items()):
        if j in claimed:
            continue
        if v["type"] == "PhysicsFixedJoint" and (
            v["child"] in light or v["parent"] in light
        ):
            ignored_joints.append(
                {"name": j, "parent": v["parent"], "child": v["child"]}
            )
        else:
            usd_only.append(j)
    return {
        "pairs": pairs,
        "urdf_only": urdf_only,
        "usd_only": usd_only,
        "fixed_joint_renames": renames,
        "ignored_usd_only": {
            "links": [
                {"name": n, "mass": usd["bodies"][n]["mass"]} for n in sorted(light)
            ],
            "joints": ignored_joints,
        },
        "light_links": light,
    }


def joint_rows(urdf, usd, pairs):
    rows = []
    for name, uname in pairs:
        u, d = urdf["joints"][name], usd["joints"][uname]
        utype, dtype = u["type"], usd_joint_type(d["type"])
        row = {
            "name": name,
            "name_usd": uname,
            "type_urdf": utype,
            "type_usd": dtype,
            "type_match": utype == dtype,
            "parent_urdf": u["parent"],
            "child_urdf": u["child"],
            "parent_usd": d["parent"],
            "child_usd": d["child"],
            "relation_match": (u["parent"], u["child"]) == (d["parent"], d["child"]),
        }
        if utype in ("revolute", "prismatic") and d["axis"] in _AXIS_VEC:
            row["axis_urdf"], row["axis_usd"] = u["axis"], d["axis"]
            # Informational only: the USD axis is a token in the joint's local frame, the URDF axis a vector in
            # the URDF joint frame; the converters may rotate frames, so this never fails the comparison.
            row["axis_match"] = [abs(c) for c in u["axis"]] == [
                float(abs(c)) for c in _AXIS_VEC[d["axis"]]
            ]
        if (
            utype == "revolute"
            and d["lower_deg"] is not None
            and d["upper_deg"] is not None
            and u["lower"] is not None
        ):
            lo, hi = deg_to_rad(d["lower_deg"]), deg_to_rad(d["upper_deg"])
            row.update(
                lower_urdf=u["lower"],
                upper_urdf=u["upper"],
                lower_usd_rad=lo,
                upper_usd_rad=hi,
                limit_diff_rad=max(abs(u["lower"] - lo), abs(u["upper"] - hi)),
            )
        if u["effort"] is not None and d["effort"] is not None:
            row.update(
                effort_urdf=u["effort"],
                effort_usd=d["effort"],
                effort_rel_diff=_rel(u["effort"], d["effort"]),
            )
        if u["velocity"] is not None and d["velocity"] is not None:
            vusd = deg_to_rad(d["velocity"]) if utype == "revolute" else d["velocity"]
            row.update(
                velocity_urdf=u["velocity"],
                velocity_usd=vusd,
                velocity_rel_diff=_rel(u["velocity"], vusd),
            )
        rows.append(row)
    return rows


def name_diffs(urdf, usd, matching=None, ignore_usd_only_mass=None):
    """Failing one-sided names (arm excluded). Light USD-only links and their fixed joints are not included."""
    matching = matching or match_joints(urdf, usd, ignore_usd_only_mass)
    names = go2_names(urdf)
    usd_bodies = set(usd["bodies"])
    return {
        "links_only_in_urdf": sorted(names - usd_bodies),
        "links_only_in_usd": sorted(usd_bodies - names - matching["light_links"]),
        "joints_only_in_urdf": sorted(matching["urdf_only"]),
        "joints_only_in_usd": sorted(matching["usd_only"]),
    }


def _max(rows, key):
    vals = [r[key] for r in rows if key in r]
    return max(vals) if vals else None


def aggregate(brows, jrows):
    return {
        "max_mass_rel_diff": _max(brows, "mass_rel_diff"),
        "max_com_dist_m": _max(brows, "com_dist_m"),
        "max_inertia_rel_diff": _max(brows, "inertia_rel_diff"),
        "max_inertia_abs_diff": _max(brows, "inertia_abs_diff"),
        "n_inertia_exceeds": sum(1 for r in brows if r.get("inertia_exceeds")),
        "max_limit_diff_rad": _max(jrows, "limit_diff_rad"),
        "max_effort_rel_diff": _max(jrows, "effort_rel_diff"),
        "max_velocity_rel_diff": _max(jrows, "velocity_rel_diff"),
        "n_bodies": len(brows),
        "n_joints": len(jrows),
        "n_type_mismatch": sum(1 for r in jrows if not r["type_match"]),
        "n_relation_mismatch": sum(1 for r in jrows if not r["relation_match"]),
        "n_axis_mismatch_informational": sum(
            1 for r in jrows if r.get("axis_match") is False
        ),
    }


def evaluate(stats, diffs, thresholds, brows=()):
    """Return (passed, failures list).

    Mass/CoM/limits use their thresholds; inertia fails only when a principal moment exceeds both the relative and
    the absolute tolerance. Heavy one-sided names and type/relation mismatches also fail. Axis, effort, velocity,
    fixed-joint renames and light USD-only links are informational and never fail.
    """
    fails = []
    for key, stat, lim in (
        ("mass", "max_mass_rel_diff", thresholds["mass_rel"]),
        ("com", "max_com_dist_m", thresholds["com_m"]),
        ("limits", "max_limit_diff_rad", thresholds["limit_rad"]),
    ):
        if stats[stat] is not None and stats[stat] > lim:
            fails.append(f"{key}: {stats[stat]:.6g} > {lim}")
    if stats["n_inertia_exceeds"]:
        bad = [r["name"] for r in brows if r.get("inertia_exceeds")]
        fails.append(
            f"inertia: {stats['n_inertia_exceeds']} bodies exceed rel {thresholds['inertia_rel']} AND abs "
            f"{thresholds['inertia_abs']} kg m^2: {bad}"
        )
    if stats["n_type_mismatch"]:
        fails.append(f"{stats['n_type_mismatch']} joint type mismatches")
    if stats["n_relation_mismatch"]:
        fails.append(f"{stats['n_relation_mismatch']} joint parent/child mismatches")
    for k, v in diffs.items():
        if v:
            fails.append(f"{k}: {v}")
    if stats["n_bodies"] == 0:
        fails.append("no common bodies compared")
    return not fails, fails


def compare(urdf, usd, thresholds=None):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    brows = body_rows(urdf, usd, thresholds)
    matching = match_joints(urdf, usd, thresholds["ignore_usd_only_mass"])
    jrows = joint_rows(urdf, usd, matching["pairs"])
    diffs = name_diffs(urdf, usd, matching)
    stats = aggregate(brows, jrows)
    passed, fails = evaluate(stats, diffs, thresholds, brows)
    info = {
        "fixed_joint_renames": matching["fixed_joint_renames"],
        "ignored_usd_only": matching["ignored_usd_only"],
        "axis_mismatches": [r["name"] for r in jrows if r.get("axis_match") is False],
        "max_effort_rel_diff": stats["max_effort_rel_diff"],
        "max_velocity_rel_diff": stats["max_velocity_rel_diff"],
    }
    return {
        "result": "PASS" if passed else "FAIL",
        "failures": fails,
        "info": info,
        "thresholds": thresholds,
        "stats": stats,
        "name_diffs": diffs,
        "bodies": brows,
        "joints": jrows,
        "total_mass_urdf_go2": sum(
            urdf["links"][r["name"]]["mass"] or 0.0 for r in brows
        ),
        "total_mass_usd_go2": sum(r["mass_usd"] for r in brows if "mass_usd" in r),
    }


def summary(report, top=5):
    lines = [f"Go2 URDF vs USD: {report['result']}"]
    s = report["stats"]
    lines.append(
        f"  bodies={s['n_bodies']} joints={s['n_joints']} mass_total urdf={report['total_mass_urdf_go2']:.4f} "
        f"usd={report['total_mass_usd_go2']:.4f} kg"
    )
    for k in (
        "max_mass_rel_diff",
        "max_com_dist_m",
        "max_inertia_rel_diff",
        "max_inertia_abs_diff",
        "max_limit_diff_rad",
    ):
        lines.append(f"  {k}: {s[k]}")
    for label, key, rows in (
        ("mass", "mass_rel_diff", report["bodies"]),
        ("CoM", "com_dist_m", report["bodies"]),
        ("inertia", "inertia_rel_diff", report["bodies"]),
        ("limit", "limit_diff_rad", report["joints"]),
    ):
        worst = sorted((r for r in rows if key in r), key=lambda r: -r[key])[:top]
        if worst:
            if key == "inertia_rel_diff":
                text = ", ".join(
                    f"{r['name']}=rel {r[key]:.4g}/abs {r['inertia_abs_diff']:.3g}"
                    for r in worst
                )
            else:
                text = ", ".join(f"{r['name']}={r[key]:.4g}" for r in worst)
            lines.append(f"  worst {label}: {text}")
    info = report.get("info", {})
    ign = info.get("ignored_usd_only", {})
    if ign.get("links") or ign.get("joints"):
        lines.append(
            "  info: ignored light USD-only links "
            + str([f"{x['name']} ({x['mass']:.3g} kg)" for x in ign["links"]])
            + ", joints "
            + str([x["name"] for x in ign["joints"]])
        )
    if info.get("fixed_joint_renames"):
        lines.append(
            "  info: fixed joint renames (matched by parent/child): "
            + ", ".join(f"{r['urdf']}->{r['usd']}" for r in info["fixed_joint_renames"])
        )
    for label, key in (
        ("effort", "max_effort_rel_diff"),
        ("velocity", "max_velocity_rel_diff"),
    ):
        if info.get(key) is not None:
            lines.append(
                f"  info: max {label} rel diff {info[key]:.4g} (not a failure)"
            )
    for f in report["failures"]:
        lines.append(f"  FAIL: {f}")
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--usd", default=DEFAULT_USD)
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument(
        "--mass-tol",
        type=float,
        default=DEFAULT_THRESHOLDS["mass_rel"],
        help="relative (0.02 = 2%%)",
    )
    p.add_argument(
        "--com-tol",
        type=float,
        default=DEFAULT_THRESHOLDS["com_m"],
        help="metres (0.002 = 2 mm)",
    )
    p.add_argument(
        "--inertia-tol",
        type=float,
        default=DEFAULT_THRESHOLDS["inertia_rel"],
        help="relative (0.10 = 10%%); fails only if ALSO above --inertia-abs-tol",
    )
    p.add_argument(
        "--inertia-abs-tol",
        type=float,
        default=DEFAULT_THRESHOLDS["inertia_abs"],
        help="absolute principal-moment tolerance in kg m^2 (default 1e-5)",
    )
    p.add_argument(
        "--ignore-usd-only-mass",
        type=float,
        default=DEFAULT_THRESHOLDS["ignore_usd_only_mass"],
        help="USD-only links lighter than this (kg) and their fixed joints are informational",
    )
    p.add_argument(
        "--limit-tol",
        type=float,
        default=DEFAULT_THRESHOLDS["limit_rad"],
        help="radians",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for label, path in (("URDF", args.urdf), ("USD", args.usd)):
        if not Path(path).is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2
    from pxr import Usd

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        print(f"error: cannot open USD {args.usd}", file=sys.stderr)
        return 2
    thresholds = {
        "mass_rel": args.mass_tol,
        "com_m": args.com_tol,
        "inertia_rel": args.inertia_tol,
        "inertia_abs": args.inertia_abs_tol,
        "limit_rad": args.limit_tol,
        "ignore_usd_only_mass": args.ignore_usd_only_mass,
    }
    report = compare(parse_urdf(Path(args.urdf)), read_usd(stage), thresholds)
    report["urdf"], report["usd"] = str(args.urdf), str(args.usd)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(summary(report))
    print(f"report: {out}")
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
