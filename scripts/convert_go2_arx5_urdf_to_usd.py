#!/usr/bin/env python3
"""Converts the merged Go2 + ARX5 URDF to one USD with Isaac Lab's `UrdfConverter` and verifies it. RUN ON THE POD.

Run: `$ISAACLAB_PATH/isaaclab.sh -p scripts/convert_go2_arx5_urdf_to_usd.py [--urdf ...] [--out ...] [--force]`

Builds the default fused robot ("merge at URDF level, convert once"; see `scripts/build_go2_arx5_urdf.py`). Works on
the OLD stack (Sim 5.1 / Lab 2.3.2) and the NEW one (Lab 3.0 line EA); the converter is imported defensively and its
cfg is built only from fields that exist (`common/cerberus_compat/urdf_converter.py`). Default `--out` is
`go2_arx5_spec.default_fused_usd_path()` (`data/assets/go2_arx5/go2_arx5.usd`).

Steps: sha256 of the URDF -> UrdfConverter (floating base, not instanceable, no self collision, joint-drive gains
zero/none because gains live in the ArticulationCfg) -> open with pxr -> if the converter left external
references/payloads/sublayers, flatten (`stage.Flatten()`) and re-export so the file is self-contained -> inspect ->
`<out>.convert.json`. The report is written FIRST, then the process hard-exits (`os._exit`), never relying on
`SimulationApp.close()`.

Report keys of note: `tcp_prim_relative_path` (path of `eef_link` relative to the default prim) and
`base_prim_relative_path`; both MUST equal `GO2_ARX5_TCP_PRIM_REL_PATH` / `GO2_ARX5_BASE_PRIM_REL_PATH` from
`go2_arx5_spec` (downstream configs use those literal paths), otherwise validation fails and the exit code is 1.
Also `root_joints` and `root_joint_must_be_removed` (a world-fixed joint must not exist for a floating base),
per-body masses of arm vs legs.

--merge-fixed-joints defaults to False: with True the converter collapses fixed-joint children into their parent and
`eef_link` (attached via the fixed `tcp_fixed_joint`) would NOT survive as a body, breaking the TCP contract.

Everything at module level is pure python (no isaaclab/omni/isaacsim import; pxr is imported lazily inside helpers), so
the helpers are testable locally. `--dry-run` prints the plan without Isaac.

Exit codes: 0 ok, 1 validation failed, 2 bad args / missing inputs / refusing to overwrite.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.cerberus_compat.urdf_converter import (
    filter_cfg_kwargs,
    framework_versions,
    import_urdf_converter,
)
from common.cerberus_description.go2_arx5_spec import (
    GO2_ARX5_BASE_PRIM_REL_PATH,
    GO2_ARX5_TCP_PRIM_REL_PATH,
    GO2_ROOT_BODY,
    TCP_BODY_NAME,
    default_fused_usd_path,
)
from common.cerberus_description.urdf_patch import (
    expected_joint_names,
    validate_joint_names,
)

DEFAULT_URDF = Path("data/assets/go2_arx5/urdf/go2_arx5.urdf")
DEFAULT_OUT = default_fused_usd_path()
COLLIDERS = ("convex_hull", "convex_decomposition", "none")
EXIT_OK, EXIT_VALIDATION_FAIL, EXIT_BAD_ARGS = 0, 1, 2
LEG_BODY_RE = re.compile(r"^(FL|FR|RL|RR)_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help=f"merged URDF (default: {DEFAULT_URDF})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output USD (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing --out"
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        default=False,
        help="let the converter merge fixed-joint children into parents (default False so eef_link survives as a body)",
    )
    parser.add_argument(
        "--collider",
        choices=COLLIDERS,
        default="convex_hull",
        help="collision approximation",
    )
    parser.add_argument(
        "--watchdog_s",
        type=float,
        default=420.0,
        help="dump all thread stacks to stderr every N seconds if hung (0 disables; default 420)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit (no Isaac)"
    )
    return parser


def _abs(path: Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def record_path(out: Path) -> Path:
    """`<out>.convert.json`, e.g. go2_arx5.usd -> go2_arx5.usd.convert.json."""
    return out.with_name(out.name + ".convert.json")


def build_plan(args: argparse.Namespace) -> dict:
    """Resolved inputs/outputs/options. Relative paths resolve against the repo root. Does not need the URDF to exist."""
    urdf, out = _abs(args.urdf), _abs(args.out)
    if out.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError(f"--out must end in .usd/.usda/.usdc, got {out.name}")
    return {
        "urdf": str(urdf),
        "urdf_exists": urdf.is_file(),
        "out": str(out),
        "usd_dir": str(out.parent),
        "usd_file": out.name,
        "record": str(record_path(out)),
        "merge_fixed_joints": bool(args.merge_fixed_joints),
        "collider": args.collider,
        "fix_base": False,
        "make_instanceable": False,
        "self_collision": False,
        "joint_drive": {"gains": [0.0, 0.0], "target_type": "none"},
        "base_body": GO2_ROOT_BODY,
        "tcp_body": TCP_BODY_NAME,
        "expected_joint_count": len(expected_joint_names()),
        "force": bool(args.force),
    }


def validate_plan(plan: dict) -> None:
    """Raises FileNotFoundError for a missing URDF; FileExistsError when --out exists without --force."""
    urdf = Path(plan["urdf"])
    if not urdf.is_file():
        raise FileNotFoundError(
            f"--urdf not found: {urdf} (build it with the merge script first)"
        )
    if Path(plan["out"]).exists() and not plan["force"]:
        raise FileExistsError(
            f"{plan['out']} already exists; pass --force to overwrite"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# USD inspection (pxr imported lazily; works on any Usd.Stage, including in-memory ones)
# ---------------------------------------------------------------------------------------------------------------


def _relative_to_default(path: str, default_prim_path: str) -> str:
    if default_prim_path and (
        path == default_prim_path or path.startswith(default_prim_path + "/")
    ):
        return path[len(default_prim_path) :].lstrip("/")
    return path


def classify_body(name: str) -> str:
    """'legs' for {FL,FR,RL,RR}_*, 'trunk' for the Go2 root `base` / Head_*, else 'arm'."""
    if LEG_BODY_RE.match(name):
        return "legs"
    if name == GO2_ROOT_BODY or name.startswith("Head_"):
        return "trunk"
    return "arm"


def external_arcs(stage) -> list[dict]:
    """Sublayers, references and payloads with a non-empty asset path (anything that makes the file non-self-contained)."""
    arcs: list[dict] = []
    for sub in stage.GetRootLayer().subLayerPaths:
        arcs.append({"kind": "sublayer", "prim": "/", "asset": str(sub)})
    for prim in stage.TraverseAll():
        for spec in prim.GetPrimStack():
            for kind, lst in (
                ("reference", spec.referenceList),
                ("payload", spec.payloadList),
            ):
                for item in lst.GetAddedOrExplicitItems():
                    if item.assetPath:
                        arcs.append(
                            {
                                "kind": kind,
                                "prim": str(prim.GetPath()),
                                "asset": str(item.assetPath),
                            }
                        )
    return arcs


def inspect_stage(stage, fix_base: bool = False) -> dict:
    """Structural facts about a converted stage (see module docstring for the keys)."""
    from pxr import UsdPhysics

    default = stage.GetDefaultPrim()
    default_path = str(default.GetPath()) if default and default.IsValid() else ""

    bodies: list[dict] = []
    joint_names: list[str] = []
    root_joints: list[str] = []
    articulation_roots = 0
    tcp_candidates: list[tuple[bool, str]] = []
    base_candidates: list[tuple[bool, str]] = []
    for prim in stage.Traverse():
        name = prim.GetName()
        is_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots += 1
        if is_body:
            mass = None
            if prim.HasAPI(UsdPhysics.MassAPI):
                value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                # MassAPI mass 0 means "not authored, computed from collision/density" (UNVERIFIED for Lab 3.0 line).
                mass = float(value) if value else None
            bodies.append(
                {
                    "name": name,
                    "path": str(prim.GetPath()),
                    "mass_kg": mass,
                    "group": classify_body(name),
                }
            )
        if name == TCP_BODY_NAME:
            tcp_candidates.append((is_body, str(prim.GetPath())))
        if name == GO2_ROOT_BODY:
            base_candidates.append((is_body, str(prim.GetPath())))
        if prim.IsA(UsdPhysics.Joint):
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(
                UsdPhysics.PrismaticJoint
            ):
                joint_names.append(name)
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            # A joint with no body0 (or no body1) is anchored to the world: a fixed base. Not valid for a floating base.
            if not body0 or not body1:
                root_joints.append(str(prim.GetPath()))

    def pick(cands):
        cands = sorted(cands, key=lambda c: (not c[0], len(c[1])))
        return (
            _relative_to_default(cands[0][1], default_path) if cands else None
        ), len(cands)

    tcp_rel, tcp_count = pick(tcp_candidates)
    base_rel, _ = pick(base_candidates)
    masses = {"total": 0.0, "legs": 0.0, "trunk": 0.0, "arm": 0.0}
    for b in bodies:
        if b["mass_kg"] is not None:
            masses[b["group"]] += b["mass_kg"]
            masses["total"] += b["mass_kg"]
    return {
        "default_prim_path": default_path,
        "rigid_body_count": len(bodies),
        "rigid_body_names": [b["name"] for b in bodies],
        "bodies": bodies,
        "bodies_without_authored_mass": [
            b["name"] for b in bodies if b["mass_kg"] is None
        ],
        "actuated_joint_names": joint_names,
        "articulation_root_count": articulation_roots,
        "root_joints": root_joints,
        "root_joint_present": bool(root_joints),
        "root_joint_must_be_removed": bool(root_joints) and not fix_base,
        "tcp_prim_relative_path": tcp_rel,
        "tcp_prim_count": tcp_count,
        "base_prim_relative_path": base_rel,
        "total_mass_kg": masses["total"],
        "mass_by_group_kg": {k: v for k, v in masses.items() if k != "total"},
        "external_arcs": external_arcs(stage),
    }


def evaluate_checks(info: dict) -> tuple[bool, list[str]]:
    """(ok, failures): exactly one articulation, the 20 expected joints, eef_link present as a body, and the TCP/base prim paths matching the spec."""
    failures: list[str] = []
    if info["articulation_root_count"] != 1:
        failures.append(
            f"expected exactly 1 PhysicsArticulationRootAPI prim, found {info['articulation_root_count']}"
        )
    try:
        validate_joint_names(info["actuated_joint_names"])
    except ValueError as e:
        failures.append(str(e))
    if len(info["actuated_joint_names"]) != len(expected_joint_names()):
        failures.append(
            f"expected {len(expected_joint_names())} actuated joints, found {len(info['actuated_joint_names'])}"
        )
    if info["tcp_prim_relative_path"] is None:
        failures.append(f"{TCP_BODY_NAME} prim not found")
    elif TCP_BODY_NAME not in info["rigid_body_names"]:
        failures.append(f"{TCP_BODY_NAME} exists but is not a rigid body")
    # Downstream configs use these literal paths (spec single source of truth).
    for key, expected in (
        ("tcp_prim_relative_path", GO2_ARX5_TCP_PRIM_REL_PATH),
        ("base_prim_relative_path", GO2_ARX5_BASE_PRIM_REL_PATH),
    ):
        if info[key] is not None and info[key] != expected:
            failures.append(
                f"{key} is {info[key]!r} but go2_arx5_spec requires {expected!r}; downstream configs use that path"
            )
    return (not failures), failures


def flatten_in_place(usd_path: Path) -> None:
    """Flatten all references/payloads/sublayers into one layer and replace `usd_path` (default prim preserved)."""
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    default_name = default.GetName() if default and default.IsValid() else None
    flat = stage.Flatten()
    if default_name and not flat.defaultPrim:
        flat.defaultPrim = default_name
    tmp = usd_path.with_name(usd_path.stem + ".flat" + usd_path.suffix)
    flat.Export(str(tmp))
    del stage, flat
    os.replace(tmp, usd_path)
    # A layer still alive in the Sdf registry would keep serving the pre-flatten content.
    cached = Sdf.Layer.Find(str(usd_path))
    if cached:
        cached.Reload(force=True)


def strip_newton_root_props(usd_path: Path) -> list[str]:
    """Remove authored `newton:*` properties from articulation-root prims of a self-contained USD; returns their paths.

    Why: the Lab 3.0 line's `modify_articulation_root_properties(fix_root_link=True)` moves the articulation root to the
    parent prim and tries to `RemoveProperty` every authored Newton schema property from the old root; that raises
    "Failed to remove '.../base.newton:selfCollisionEnabled'" when the property comes from the referenced USD layer, which
    is always the case for a spawned asset. The converter authors `newton:selfCollisionEnabled`/`newton:inertia` on the
    root body; PhysX does not read them. Without this step `--fix_base` spawns of the converted USD crash.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    removed: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        for prop in list(prim.GetAuthoredProperties()):
            if prop.GetName().startswith("newton:") and prim.RemoveProperty(
                prop.GetName()
            ):
                removed.append(f"{prim.GetPath()}.{prop.GetName()}")
    if removed:
        stage.GetRootLayer().Save()
        cached = Sdf.Layer.Find(str(usd_path))
        if cached:
            cached.Reload(force=True)
    del stage
    return removed


def build_report(plan: dict, info: dict, extra: dict | None = None) -> dict:
    ok, failures = evaluate_checks(info)
    found = info["actuated_joint_names"]
    expected = list(expected_joint_names())
    report = {
        **plan,
        **info,
        "expected_joint_names": expected,
        "missing_joints": [n for n in expected if n not in set(found)],
        "unexpected_joints": sorted(set(found) - set(expected)),
        "validation": "OK" if ok else "FAIL",
        "validation_failures": failures,
        "self_contained": not info["external_arcs"],
    }
    report.update(extra or {})
    return report


def write_report(plan: dict, report: dict) -> Path:
    path = Path(plan["record"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------------------------------------------
# Everything below needs the Isaac app (imports lazily)
# ---------------------------------------------------------------------------------------------------------------


def converter_versions(module_name: str) -> dict:
    """Converter module name/version plus the urdf-usd-converter distribution version (None when absent)."""
    from importlib import metadata

    out: dict = {"converter_module": module_name}
    for dist in ("urdf-usd-converter", "urdf_usd_converter"):
        try:
            out["urdf_usd_converter"] = metadata.version(dist)
            break
        except metadata.PackageNotFoundError:
            out.setdefault("urdf_usd_converter", None)
    return out


def convert_urdf(plan: dict) -> tuple[Path, dict]:
    """merged URDF -> USD via UrdfConverter. Returns (usd path reported by the converter, converter info)."""
    converter_cls, cfg_cls, module_name = import_urdf_converter()
    candidates: dict = {
        "asset_path": plan["urdf"],
        "usd_dir": plan["usd_dir"],
        "usd_file_name": plan["usd_file"],
        "force_usd_conversion": True,
        "make_instanceable": plan["make_instanceable"],
        "fix_base": plan["fix_base"],
        "merge_fixed_joints": plan["merge_fixed_joints"],
        "self_collision": plan["self_collision"],
    }
    # UNVERIFIED: Lab 2.3.2 collider_type accepts only convex_hull/convex_decomposition, so 'none' omits the field
    # (converter default applies) and is recorded as such; the Lab 3.0 rewrite may have no collider_type at all.
    if plan["collider"] != "none":
        candidates["collider_type"] = plan["collider"]
    joint_drive_cls = getattr(cfg_cls, "JointDriveCfg", None)
    if joint_drive_cls is not None and hasattr(joint_drive_cls, "PDGainsCfg"):
        candidates["joint_drive"] = joint_drive_cls(
            drive_type="force",
            target_type="none",
            gains=joint_drive_cls.PDGainsCfg(stiffness=0.0, damping=0.0),
        )
    kwargs, dropped = filter_cfg_kwargs(cfg_cls, candidates)
    if "asset_path" not in kwargs:
        raise RuntimeError(
            f"{cfg_cls.__name__} from {module_name} has no `asset_path` field. STOP and report."
        )
    print(
        f"[INFO] UrdfConverterCfg from {module_name}: using {sorted(kwargs)}, dropped (no such field): {dropped}",
        flush=True,
    )
    # UNVERIFIED: on the Lab 3.0 line the converter may ignore usd_dir/usd_file_name; `usd_path` is authoritative.
    usd_path = Path(converter_cls(cfg_cls(**kwargs)).usd_path)
    info = {
        **converter_versions(module_name),
        "cfg_fields_used": sorted(kwargs),
        "cfg_fields_dropped": dropped,
    }
    return usd_path, info


def run_convert(plan: dict) -> int:
    from pxr import Usd

    urdf = Path(plan["urdf"])
    extra: dict = {"versions": framework_versions(), "input_sha256": sha256_file(urdf)}
    usd_path, conv_info = convert_urdf(plan)
    extra.update(conv_info)
    out = Path(plan["out"])
    print(f"[INFO] UrdfConverter done: {usd_path}", flush=True)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"could not open converted USD {usd_path}")
    arcs_before = external_arcs(stage)
    extra["external_arcs_before_flatten"] = arcs_before
    extra["converter_usd_path"] = str(usd_path)
    needs_copy = usd_path.resolve() != out.resolve()
    if arcs_before or needs_copy:
        # Flatten also covers the copy case: export the composed stage to the requested --out.
        # UNVERIFIED: Flatten keeps the default prim and physics schemas as authored (checked again below).
        if needs_copy:
            out.parent.mkdir(parents=True, exist_ok=True)
            default = stage.GetDefaultPrim()
            flat = stage.Flatten()
            if default and default.IsValid() and not flat.defaultPrim:
                flat.defaultPrim = default.GetName()
            flat.Export(str(out))
        else:
            flatten_in_place(out)
        extra["flattened"] = True
        print(
            f"[INFO] flattened {len(arcs_before)} external arc(s) into {out}",
            flush=True,
        )
    else:
        extra["flattened"] = False
    del stage

    extra["newton_root_props_stripped"] = strip_newton_root_props(out)
    final = Usd.Stage.Open(str(out))
    info = inspect_stage(final, fix_base=plan["fix_base"])
    extra["output_size_bytes"] = out.stat().st_size
    report = build_report(plan, info, extra)
    path = write_report(plan, report)
    print(
        f"[convert_go2_arx5] validation: {report['validation']} {report['validation_failures']}",
        flush=True,
    )
    print(
        f"[convert_go2_arx5] tcp_prim_relative_path={info['tcp_prim_relative_path']} "
        f"root_joints={info['root_joints']} self_contained={report['self_contained']}",
        flush=True,
    )
    print(f"[convert_go2_arx5] record: {path}", flush=True)
    return EXIT_OK if report["validation"] == "OK" else EXIT_VALIDATION_FAIL


def _write_error_record(plan: dict, tb: str) -> None:
    """Best-effort error record; never raises."""
    try:
        write_report(
            plan,
            {
                **plan,
                "validation": "ERROR",
                "error": tb,
                "versions": framework_versions(),
            },
        )
    except BaseException:  # noqa: BLE001
        traceback.print_exc()


def _finish(simulation_app, exit_code: int) -> None:
    """SimulationApp.close() terminates the process with code 0 (losing the real exit code and any traceback) and was observed to hang for >10 min at shutdown on the RunPod RTX 4090 pod (driver 595.91, Isaac Sim 5.1); everything we need (USD, convert.json record) is on disk before this point, so we hard-exit."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    try:
        plan = build_plan(pre_args)
        if not pre_args.dry_run:
            validate_plan(plan)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_ARGS
    if pre_args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return EXIT_OK

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True
    simulation_app = AppLauncher(args).app

    watchdog_s = float(args.watchdog_s)
    faulthandler.enable()
    if watchdog_s > 0:
        faulthandler.dump_traceback_later(watchdog_s, repeat=True, exit=False)
    exit_code = EXIT_VALIDATION_FAIL
    try:
        exit_code = run_convert(plan)
    except FileNotFoundError as e:
        traceback.print_exc()
        print(f"error: {e}", file=sys.stderr)
        exit_code = EXIT_BAD_ARGS
        _write_error_record(plan, traceback.format_exc())
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        exit_code = EXIT_VALIDATION_FAIL
        _write_error_record(plan, traceback.format_exc())
    finally:
        faulthandler.cancel_dump_traceback_later()
        # Report is on disk; SimulationApp.close() can hang/force exit 0, so hard-exit with the real code.
        _finish(simulation_app, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
