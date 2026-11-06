"""Pure-python decision logic of the scene admission test (no Isaac, no numpy; stdlib + the admission manifest).

``scripts/validate_scenes.py`` measures raw facts in child processes (spawn candidates, LiDAR hits, a stepped Go2);
everything that turns those facts into a ``pass``/``fail`` verdict lives here so it is unit-tested locally:

* ``Thresholds``: every threshold as a named, documented, PROVISIONAL constant (none was calibrated on data yet;
  recalibrate from the first full sweep, then freeze).
* ``evaluate_*``: facts -> failure reasons; ``build_row``: facts -> ``AdmissionRow``.
* ``stratified_kitchen_subset``: deterministic spread of N kitchens over the Lightwheel 10 x 10 layout/style grid.
* ``scenes_to_run``: ``--resume`` / ``--limit`` selection against the existing manifest.

Child result dicts (``sample`` / ``env`` / ``render``) are plain JSON written by the child process; the parent adds
``timed_out`` and ``returncode``. Their keys are documented at the evaluators below.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_worlds.admission import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    AdmissionRow,
    SpawnPose,
)

KITCHEN_LAYOUTS = 10
KITCHEN_STYLES = 10


@dataclass(frozen=True)
class Thresholds:
    """Provisional thresholds. Geometry values size the Go2 (about 0.4 m tall, 0.7 m long, LiDAR 0.05 m above the
    base, SLAM LiDAR from ``common/cerberus_description/sensors/sensor_configs.py``); the arm variant adds headroom."""

    # --- spawn sampling (used by scene_spawn_sampling) ---
    grid_cell_m: float = 0.25  # xy sampling pitch; coarsened automatically above max_grid_cells
    max_grid_cells: int = 250_000
    max_hits_per_column: int = 16  # surfaces recorded per downward ray column (ceiling slab, furniture, floor, ...)
    clear_height_m: float = 0.8  # free vertical space above the floor: Go2 0.4 + LiDAR + margin for a stowed arm
    max_ceiling_m: float = 8.0  # a surface directly above the floor closer than this counts as a roof/ceiling
    min_floor_normal_z: float = 0.9  # n_z of the up-facing hit face: floors are flat (about 25 deg)
    max_step_m: float = 0.08  # floor height jump between neighbouring cells that blocks the cell (stairs, kerbs)
    level_tol_m: float = 0.15  # candidate floors this close to the chosen floor level count as the same level
    min_open_radius_m: float = 0.6  # free half-width around a spawn (Go2 bounding circle 0.35 m plus margin)
    max_open_radius_m: float = 5.0  # erosion limit for ranking spawns by open area
    spawn_separation_m: float = 2.0  # minimum distance between two stored spawn poses
    max_spawns: int = 5
    # --- LiDAR sweep: same pattern as LIDAR_CFG (100 channels, -25..15 deg, 360 deg at 1 deg) ---
    lidar_channels: int = 100
    lidar_vfov_deg: tuple[float, float] = (-25.0, 15.0)
    lidar_hres_deg: float = 1.0
    lidar_range_m: float = 50.0
    lidar_height_above_floor_m: float = 0.35  # default standing base height (~0.30) + the 0.05 m mount offset
    min_lidar_hit_fraction: float = 0.30  # an enclosed room hits well above this; an empty scene hits only the floor
    # --- physics check ---
    root_z_min_above_floor_m: float = 0.12  # below this the robot has sunk into / fallen through the floor (Galileo)
    root_z_max_above_floor_m: float = 0.50  # above this the robot is perched on something or the floor is wrong
    max_tilt_deg: float = 30.0  # stock Go2 PD sag leans ~12 deg (docs/arena_integration_log.md)
    max_drift_m: float = 1.5  # zero actions: the base must not slide or be ejected
    max_step_ms: float = 400.0  # per env.step at the scale env count; training would be unusable above this
    max_gpu_mib: float = 20_000.0  # must fit the 24 GB A5000 class pods with headroom
    # --- render pass (only when --render) ---
    render_min_std: float = 2.0  # pixel std of a non-blank RGB frame (a missing-MDL grey/black frame is ~0)


# ---------------------------------------------------------------------------------------------------------
# evaluators: raw child facts -> failure reasons
# ---------------------------------------------------------------------------------------------------------
def _short(text: object, limit: int = 200) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _phase_failure(res: dict[str, Any], label: str) -> str | None:
    """A reason when the child did not complete (timeout / crash / exception), else None."""
    if res.get("timed_out"):
        return f"{label}: timeout after {res.get('wall_s', '?')} s at stage {res.get('stage', 'launch')!r}"
    if not res.get("ok"):
        err = res.get("error") or f"child exited {res.get('returncode')} without a result"
        return f"{label}: error at stage {res.get('stage', 'launch')!r}: {_short(err)}"
    return None


def evaluate_sample(res: dict[str, Any], th: Thresholds) -> list[str]:
    """Reasons the geometry phase fails. Keys: ``up_axis``, ``spawns`` (dicts with x, y, yaw, floor_z),
    ``lidar_hits``, ``lidar_rays``, ``lidar_error``."""
    failure = _phase_failure(res, "sample")
    if failure:
        return [failure]
    reasons = []
    if res.get("up_axis") != "Z":
        reasons.append(f"scene up axis is {res.get('up_axis')!r}, only Z-up scenes are supported")
    if not res.get("spawns"):
        reasons.append("no valid spawn (no floor with clear torso/head volume found)")
    if res.get("lidar_error"):
        reasons.append(f"lidar check error: {_short(res['lidar_error'])}")
    elif res.get("spawns"):
        rays = res.get("lidar_rays") or 0
        frac = (res.get("lidar_hits") or 0) / rays if rays else 0.0
        if frac < th.min_lidar_hit_fraction:
            reasons.append(f"lidar hit fraction {frac:.2f} < {th.min_lidar_hit_fraction}")
    return reasons


def evaluate_env(
    res: dict[str, Any], th: Thresholds, label: str, *, check_scale: bool = False
) -> list[str]:
    """Reasons a stepped-env run fails. Keys: ``steps_done``, ``steps_requested``, ``obs_finite``,
    ``root_z_rel_min`` / ``root_z_rel_max`` (base z minus the measured floor_z), ``tilt_deg_max``, ``drift_m_max``,
    ``step_ms_mean``, ``gpu_mem_peak_mib``. `check_scale` adds the speed and memory limits (16-env run)."""
    failure = _phase_failure(res, label)
    if failure:
        return [failure]
    reasons = []
    if res.get("steps_done") != res.get("steps_requested"):
        reasons.append(f"{label}: only {res.get('steps_done')}/{res.get('steps_requested')} steps completed")
    if not res.get("obs_finite"):
        reasons.append(f"{label}: non-finite observation")
    z_min, z_max = res.get("root_z_rel_min"), res.get("root_z_rel_max")
    if z_min is None or z_max is None:
        reasons.append(f"{label}: no root height measured")
    else:
        if not z_min >= th.root_z_min_above_floor_m:
            reasons.append(
                f"{label}: base z {z_min:.2f} m above floor < {th.root_z_min_above_floor_m} (fell through / sank)"
            )
        if z_max > th.root_z_max_above_floor_m:
            reasons.append(f"{label}: base z {z_max:.2f} m above floor > {th.root_z_max_above_floor_m}")
    tilt = res.get("tilt_deg_max")
    if tilt is None or not tilt <= th.max_tilt_deg:
        reasons.append(f"{label}: tilt {tilt} deg > {th.max_tilt_deg}")
    drift = res.get("drift_m_max")
    if drift is None or not drift <= th.max_drift_m:
        reasons.append(f"{label}: drift {drift} m > {th.max_drift_m}")
    if check_scale:
        step_ms = res.get("step_ms_mean")
        if step_ms is None or not step_ms <= th.max_step_ms:
            reasons.append(f"{label}: step {step_ms} ms > {th.max_step_ms}")
        gpu = res.get("gpu_mem_peak_mib")
        if gpu is not None and gpu > th.max_gpu_mib:
            reasons.append(f"{label}: peak GPU {gpu:.0f} MiB > {th.max_gpu_mib:.0f}")
    return reasons


def evaluate_render(res: dict[str, Any], th: Thresholds) -> list[str]:
    """Reasons the optional RGB frame fails. Keys: ``frame_mean``, ``frame_std``."""
    failure = _phase_failure(res, "render")
    if failure:
        return [failure]
    std = res.get("frame_std")
    if std is None or not std >= th.render_min_std:
        return [f"render: frame is blank (std {std} < {th.render_min_std}); missing MDL materials?"]
    return []


def evaluate_scene(
    sample: dict[str, Any] | None,
    env_small: dict[str, Any] | None,
    env_scale: dict[str, Any] | None,
    render: dict[str, Any] | None,
    th: Thresholds,
    *,
    render_required: bool = False,
    env_combined: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """``(status, reasons)``. A phase that is ``None`` was not run because an earlier phase already failed.

    ``env_combined`` is the one-launch variant (`validate_scenes.py --combined-env-phases`): a single run at the
    scale env count with the long step count is checked against BOTH the stability rules and the scale rules, which
    replaces the `env_small` + `env_scale` pair and removes one Kit launch (and its `build_env`) per scene. Rows
    written before that option existed keep the split keys, and the labels in the reasons say which shape ran.
    """
    reasons: list[str] = []
    phases = (
        (
            (sample, lambda r: evaluate_sample(r, th)),
            (env_combined, lambda r: evaluate_env(r, th, "env")),
            (env_combined, lambda r: evaluate_env(r, th, "env_scale", check_scale=True)),
        )
        if env_combined is not None
        else (
            (sample, lambda r: evaluate_sample(r, th)),
            (env_small, lambda r: evaluate_env(r, th, "env4")),
            (env_scale, lambda r: evaluate_env(r, th, "env16", check_scale=True)),
        )
    )
    for res, evaluator in phases:
        if reasons:
            break
        reasons += evaluator(res) if res is not None else ["internal: phase not run"]
    if render_required and not reasons:
        reasons += evaluate_render(render, th) if render is not None else ["internal: render phase not run"]
    return (STATUS_FAIL if reasons else STATUS_PASS), reasons


def summarize_probe(res: dict[str, Any], th: Thresholds) -> dict[str, Any]:
    """The Arena shared-scene probe as recorded in ``metrics['arena_probe']`` (never affects the verdict)."""
    reasons = evaluate_env(res, th, "arena_probe")
    return {
        "status": STATUS_FAIL if reasons else STATUS_PASS,
        "reasons": reasons,
        "build_s": res.get("build_s"),
        "step_ms_mean": res.get("step_ms_mean"),
        "timed_out": bool(res.get("timed_out")),
    }


def collect_metrics(
    sample: dict[str, Any] | None,
    env_small: dict[str, Any] | None,
    env_scale: dict[str, Any] | None,
    render: dict[str, Any] | None,
    probe: dict[str, Any] | None,
    env_combined: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flat metrics dict for the manifest row (None-valued entries dropped).

    With ``env_combined`` the shared metrics come from that one run and the split-only keys (`build_s_4`/`build_s_16`,
    `step_ms_4`/`step_ms_16`) are left out, so a row never claims a 4-env build it did not do; `env_phase` says which
    shape it is.
    """
    combined = env_combined is not None
    sample = sample or {}
    if combined:
        env_small = env_scale = env_combined or {}
    else:
        env_small, env_scale = env_small or {}, env_scale or {}
    gpu = [v for v in (env_small.get("gpu_mem_peak_mib"), env_scale.get("gpu_mem_peak_mib")) if v is not None]
    metrics: dict[str, Any] = {
        "env_phase": "combined" if combined else "split",
        "build_s": env_small.get("build_s") if combined else None,
        "step_ms": env_small.get("step_ms_mean") if combined else None,
        "build_s_4": None if combined else env_small.get("build_s"),
        "build_s_16": None if combined else env_scale.get("build_s"),
        "step_ms_4": None if combined else env_small.get("step_ms_mean"),
        "step_ms_16": None if combined else env_scale.get("step_ms_mean"),
        "peak_gpu_mib": max(gpu) if gpu else None,
        "tilt_deg": env_small.get("tilt_deg_max"),
        "root_z_rel_min": env_small.get("root_z_rel_min"),
        "root_z_rel_max": env_small.get("root_z_rel_max"),
        "drift_m": env_small.get("drift_m_max"),
        "lidar_hits": sample.get("lidar_hits"),
        "lidar_rays": sample.get("lidar_rays"),
        "lidar_mesh_triangles": sample.get("triangles"),
        "lidar_mesh_build_s": sample.get("mesh_build_s"),
        "floor_level_z": sample.get("level_z"),
        "footprint_m": sample.get("footprint_m"),
        "spawn_open_radius_m": [s.get("open_radius_m") for s in sample.get("spawns", [])] or None,
        "sample_s": sample.get("wall_s"),
        "gpu_name": env_small.get("gpu_name") or env_scale.get("gpu_name"),
        "render_frame_std": (render or {}).get("frame_std"),
        "arena_probe": probe,
    }
    return {k: v for k, v in metrics.items() if v is not None}


def build_row(
    *,
    scene_id: str,
    source: str,
    category: str,
    backend: str,
    sample: dict[str, Any] | None,
    env_small: dict[str, Any] | None,
    env_scale: dict[str, Any] | None,
    render: dict[str, Any] | None,
    probe: dict[str, Any] | None,
    th: Thresholds,
    date_utc: str,
    render_required: bool = False,
    env_combined: dict[str, Any] | None = None,
) -> AdmissionRow:
    status, reasons = evaluate_scene(
        sample, env_small, env_scale, render, th, render_required=render_required, env_combined=env_combined
    )
    poses = tuple(
        SpawnPose(x=s["x"], y=s["y"], yaw=s["yaw"], floor_z=s["floor_z"]) for s in (sample or {}).get("spawns", [])
    )
    return AdmissionRow(
        scene_id=scene_id,
        source=source,
        category=category,
        backend=backend,
        status=status,
        reasons=tuple(reasons),
        spawn_poses=poses,
        shared_scene=bool((env_combined or env_small or {}).get("shared_scene", True)),
        metrics=collect_metrics(sample, env_small, env_scale, render, probe, env_combined),
        date_utc=date_utc,
    )


def skip_row(*, scene_id: str, source: str, category: str, backend: str, reason: str, date_utc: str) -> AdmissionRow:
    return AdmissionRow(
        scene_id=scene_id, source=source, category=category, backend=backend, status=STATUS_SKIP,
        reasons=(reason,), date_utc=date_utc,
    )


def thresholds_dict(th: Thresholds) -> dict[str, Any]:
    return asdict(th)


# ---------------------------------------------------------------------------------------------------------
# Lightwheel subset, resume
# ---------------------------------------------------------------------------------------------------------
def stratified_kitchen_subset(n: int) -> list[tuple[int, int]]:
    """`n` distinct 1-based ``(layout, style)`` pairs spread over the 10 x 10 grid, deterministic.

    Layouts cycle 1..10 and the style steps by 7 (a permutation of 1..10 per layout cycle) plus a 3 shift per
    extra cycle, so the first 10 pairs cover every layout and every style once and all 100 pairs are reachable.
    The result starts with (1, 1), the one kitchen known to download anonymously.
    """
    total = KITCHEN_LAYOUTS * KITCHEN_STYLES
    if not 1 <= n <= total:
        raise ValueError(f"lightwheel subset size must be 1..{total}, got {n}")
    return [(i % KITCHEN_LAYOUTS + 1, (7 * i + 3 * (i // KITCHEN_LAYOUTS)) % KITCHEN_STYLES + 1) for i in range(n)]


def scenes_to_run(
    scenes: list[tuple[str, str]],
    existing: list[AdmissionRow],
    *,
    resume: bool,
    retry_failed: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Scene ids this invocation should run, in the given order. `scenes` is ``(scene_id, category)`` pairs
    (callers sort smallest scene first). ``resume`` skips ids already in the manifest (with ``retry_failed`` only
    passed/skipped rows are kept); ``limit`` caps the scenes run PER CATEGORY, counted after the resume skips."""
    done = {r.scene_id for r in existing if resume and (not retry_failed or r.status != STATUS_FAIL)}
    taken: dict[str, int] = {}
    todo = []
    for scene_id, category in scenes:
        if scene_id in done or (limit is not None and taken.get(category, 0) >= limit):
            continue
        taken[category] = taken.get(category, 0) + 1
        todo.append(scene_id)
    return todo
