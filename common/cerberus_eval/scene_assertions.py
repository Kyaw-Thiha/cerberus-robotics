"""The free assertions the first simulated step must satisfy, plus the ``assertions.json`` writer.

Pure Python (stdlib + yaml). Everything here takes plain lists/arrays, so it is testable with mocked joint names and
synthetic observation vectors; the pod driver (``scripts/scene_eval.py``) only supplies the live values.

Checks, in the order the driver runs them:

1. **Joint order** -- the live articulation's joint names against the contract
   (``common/cerberus_policies/contracts/heightscan_ppo_v1.yaml``) AND against ``platforms.yaml``'s ``go2.joint_names``.
   A permutation is not cosmetic: the observation and the action targets are positional.
2. **Observation shape** -- exactly 235, with the contract's term order. The command slice is the one term whose
   value is known independently (the driver injected it), so it is checked by value, not only by offset.
3. **Height scan on flat ground** -- recorded, not enforced. The contract marks the convention ``unverified``; this
   reports the measured constant, whether it is constant, and which candidate conventions it matches, and never
   edits the contract.
4. **Finiteness** -- no NaN/inf in a zero-command stand.

Nothing here raises on a *soft* finding (the height-scan convention); :func:`assert_all` raises only on the hard
checks, which is why a mismatched joint order stops the run instead of producing a plausible-looking eval.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "common" / "cerberus_policies" / "contracts" / "heightscan_ppo_v1.yaml"
PLATFORMS_PATH = REPO_ROOT / "system" / "cerberus_bringup" / "config" / "platforms.yaml"
EXPECTED_OBS_DIM = 235
EXPECTED_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)
STAND_STEPS = 200


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None

    def __str__(self) -> str:
        return f"{self.name}: {'PASS' if self.passed else 'FAIL'}"


@dataclass
class AssertionReport:
    checks: list[Check] = field(default_factory=list)
    soft: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: Any = None) -> Check:
        check = Check(name=name, passed=bool(passed), detail=detail)
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "soft": self.soft,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------------------------------------
# contract / platform readers
# ---------------------------------------------------------------------------------------------------------
def load_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def contract_joint_order(contract: Mapping[str, Any] | None = None) -> list[str]:
    return list((contract or load_contract())["joints"]["order"])


def contract_default_joint_pos(contract: Mapping[str, Any] | None = None) -> dict[str, float]:
    return dict((contract or load_contract())["joints"]["default_joint_pos_rad"])


def contract_terms(contract: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return list((contract or load_contract())["observation"]["terms"])


def platform_joint_order(platforms_path: str | Path = PLATFORMS_PATH, platform: str = "go2") -> list[str]:
    return list(yaml.safe_load(Path(platforms_path).read_text(encoding="utf-8"))[platform]["joint_names"])


def term_offsets(terms: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    """``{term: (start, stop)}`` from the contract's declaration order."""
    offsets, start = {}, 0
    for term in terms:
        stop = start + int(term["dim"])
        offsets[str(term["name"])] = (start, stop)
        start = stop
    return offsets


# ---------------------------------------------------------------------------------------------------------
# 1. joint order
# ---------------------------------------------------------------------------------------------------------
def joint_order_problems(actual: Sequence[str], expected: Sequence[str], label: str = "expected") -> list[str]:
    """Every way ``actual`` differs from ``expected`` (missing, extra, or out of order)."""
    problems = []
    if len(actual) != len(expected):
        problems.append(f"{len(actual)} joints vs {len(expected)} in {label}: {list(actual)} vs {list(expected)}")
    missing = [j for j in expected if j not in actual]
    extra = [j for j in actual if j not in expected]
    if missing:
        problems.append(f"missing joints vs {label}: {missing}")
    if extra:
        problems.append(f"unexpected joints vs {label}: {extra}")
    if not problems:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                problems.append(f"joint {i} is {a!r} but {label} has {e!r}")
                break
    return problems


def check_joint_order(
    actual: Sequence[str],
    contract: Mapping[str, Any] | None = None,
    platforms_path: str | Path = PLATFORMS_PATH,
) -> list[str]:
    """Problems against the contract order and, when readable, ``platforms.yaml``'s order."""
    problems = joint_order_problems(actual, contract_joint_order(contract), "contract joints.order")
    try:
        platform = platform_joint_order(platforms_path)
    except (OSError, KeyError, yaml.YAMLError) as e:  # a missing config is a finding, not a crash
        problems.append(f"platforms.yaml unreadable ({type(e).__name__}: {e})")
    else:
        problems.extend(joint_order_problems(actual, platform, "platforms.yaml go2.joint_names"))
    return problems


def assert_joint_order(
    actual: Sequence[str],
    contract: Mapping[str, Any] | None = None,
    platforms_path: str | Path = PLATFORMS_PATH,
) -> None:
    """Raise ``AssertionError`` on any mismatch: positional joint data would silently corrupt the run."""
    problems = check_joint_order(actual, contract, platforms_path)
    if problems:
        raise AssertionError("articulation joint order does not match the contract:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------------------------------------
# 2. observation shape and term order
# ---------------------------------------------------------------------------------------------------------
def check_observation_dim(obs_dim: int, expected: int = EXPECTED_OBS_DIM) -> list[str]:
    if obs_dim == expected:
        return []
    return [f"observation is {obs_dim}-D, the policy needs {expected}-D (3+3+3+3+12+12+12+187)"]


def check_term_layout(terms: Sequence[Mapping[str, Any]], expected: Sequence[str] = EXPECTED_TERMS) -> list[str]:
    names = tuple(str(t["name"]) for t in terms)
    if names != tuple(expected):
        return [f"term order is {names}, contract expects {tuple(expected)}"]
    total = sum(int(t["dim"]) for t in terms)
    return [] if total == EXPECTED_OBS_DIM else [f"terms sum to {total}, not {EXPECTED_OBS_DIM}"]


def check_command_slice(obs_row: Sequence[float], command: Mapping[str, float], terms: Sequence[Mapping[str, Any]] | None = None,
                        tol: float = 1e-4) -> list[str]:
    """The injected command must appear in ``velocity_commands``: catches a silently reordered observation."""
    offsets = term_offsets(terms or contract_terms())
    start, stop = offsets["velocity_commands"]
    got = [float(v) for v in obs_row[start:stop]]
    want = [float(command[axis]) for axis in ("vx", "vy", "wz")]
    if len(got) != 3 or any(abs(a - b) > tol for a, b in zip(got, want)):
        return [f"obs[{start}:{stop}] is {got}, injected command was {want}"]
    return []


# ---------------------------------------------------------------------------------------------------------
# 3. height scan on flat ground (recorded, not enforced)
# ---------------------------------------------------------------------------------------------------------
def height_scan_report(
    scan: Iterable[float],
    *,
    base_height_m: float | None = None,
    offset_m: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Measured constant, clip hits and which candidate conventions it matches.

    The contract's formula is ``sensor_z - ray_hit_z - offset`` with ``offset`` defaulting to 0.5 and not overridden
    in the saved params. With the scanner mounted on the base and a flat floor, ``sensor_z - hit_z`` is the base
    height above the floor, so the candidates below are all plausible readings of the same physics. ``matches``
    lists them; an empty list means the scan follows a convention none of these describe -- report it, do not
    "fix" the contract.
    """
    values = [float(v) for v in scan]
    if not values:
        raise ValueError("height scan is empty")
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    report: dict[str, Any] = {
        "cells": len(values),
        "min": lo,
        "max": hi,
        "mean": mean,
        "spread": hi - lo,
        "constant": (hi - lo) <= 1e-6,
        "at_clip_low": sum(1 for v in values if v <= clip[0] + 1e-6),
        "at_clip_high": sum(1 for v in values if v >= clip[1] - 1e-6),
        "clip": list(clip),
        "offset_m": offset_m,
        "base_height_m": base_height_m,
        "matches": [],
        "convention_status": "unverified",
    }
    if base_height_m is not None:
        candidates = {
            "base_minus_offset": base_height_m - offset_m,
            "offset_minus_base": offset_m - base_height_m,
            "base_minus_offset_plus_20": base_height_m + 20.0 - offset_m,  # if sensor_z includes ray_offset z
            "zero": 0.0,
        }
        report["candidates"] = candidates
        report["matches"] = [name for name, value in candidates.items() if abs(mean - value) <= 1e-3]
    return report


# ---------------------------------------------------------------------------------------------------------
# 4. spawn layout (shared terrain: robots must not start on top of each other)
# ---------------------------------------------------------------------------------------------------------
def check_spawn_separation(poses: Sequence[Sequence[float]], min_separation_m: float = 0.8) -> list[str]:
    """Problems for any pair of per-env start poses closer than ``min_separation_m``.

    All envs share ONE scene prim, so nothing stops two robots from being spawned inside each other. Isaac Lab's
    per-env collision filtering is UNVERIFIED for a shared USD terrain, so this check is the cheap guard the design
    note promises: fail before the physics step rather than after an explosion.
    """
    problems = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            d = math.hypot(float(poses[i][0]) - float(poses[j][0]), float(poses[i][1]) - float(poses[j][1]))
            if d < min_separation_m:
                problems.append(f"spawns {i} and {j} are {d:.2f} m apart (< {min_separation_m} m)")
    return problems


# ---------------------------------------------------------------------------------------------------------
# 5. finiteness
# ---------------------------------------------------------------------------------------------------------
def nonfinite_indices(values: Iterable[float]) -> list[int]:
    return [i for i, v in enumerate(values) if not math.isfinite(float(v))]


def check_finite(series: Iterable[Iterable[float]], label: str = "observation") -> list[str]:
    """Problems for every row in ``series`` that holds a NaN/inf."""
    problems = []
    for row_index, row in enumerate(series):
        bad = nonfinite_indices(row)
        if bad:
            problems.append(f"{label} row {row_index} has non-finite values at indices {bad[:8]}{'...' if len(bad) > 8 else ''}")
    return problems


# ---------------------------------------------------------------------------------------------------------
# combined report
# ---------------------------------------------------------------------------------------------------------
def build_report(
    *,
    joint_names: Sequence[str],
    obs_dim: int,
    obs_row: Sequence[float] | None = None,
    command: Mapping[str, float] | None = None,
    scan: Iterable[float] | None = None,
    base_height_m: float | None = None,
    stand_rows: Sequence[Sequence[float]] | None = None,
    stand_steps_expected: int = STAND_STEPS,
    contract: Mapping[str, Any] | None = None,
) -> AssertionReport:
    """Run every check the driver has data for; missing inputs are reported as skipped soft entries."""
    contract = contract or load_contract()
    report = AssertionReport()

    report.add("joint_order", not check_joint_order(joint_names, contract), {
        "actual": list(joint_names),
        "problems": check_joint_order(joint_names, contract),
    })

    terms = contract_terms(contract)
    report.add("term_layout", not check_term_layout(terms), {"terms": [t["name"] for t in terms]})
    report.add("observation_dim", not check_observation_dim(obs_dim), {"obs_dim": obs_dim})

    if obs_row is not None and command is not None:
        problems = check_command_slice(obs_row, command, terms)
        report.add("command_slice", not problems, {"problems": problems})
    else:
        report.soft["command_slice"] = "skipped (no observation row / injected command supplied)"

    if scan is not None:
        report.soft["height_scan_flat_ground"] = height_scan_report(scan, base_height_m=base_height_m)
    else:
        report.soft["height_scan_flat_ground"] = "skipped (no scan captured)"

    if stand_rows is not None:
        problems = check_finite(stand_rows, "zero-command stand observation")
        report.add("stand_finite", not problems, {
            "steps": len(stand_rows),
            "expected_steps": stand_steps_expected,
            "problems": problems[:5],
        })
        report.add("stand_length", len(stand_rows) >= stand_steps_expected, {
            "steps": len(stand_rows), "expected_steps": stand_steps_expected,
        })
    else:
        report.soft["stand_finite"] = "skipped (no stand trace captured)"
    return report


def assert_all(report: AssertionReport) -> None:
    """Raise on any hard failure, with every failure in one message."""
    if report.passed:
        return
    lines = [f"- {c.name}: {c.detail}" for c in report.failures()]
    raise AssertionError("scene assertions failed:\n" + "\n".join(lines))
