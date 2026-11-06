"""Scene admission test: which scenes can the Go2 stack actually use, and where may the robot start?

Spec: plans/arena_integration_handoff.md section 4c. One manifest per source (``data/scenes/<source>/admission.json``,
``common/cerberus_worlds/admission.py``) with one row per scene: pass/fail/skip, reasons, validated spawn poses
(best first, with the measured floor height) and metrics. Phase 1 navigation and the SLAM eval read it.

Sources: ``grscenes`` (plain_usd backend; ``home`` and ``commercial``, every scene on disk) and ``lightwheel``
(Arena backend; a deterministic subset of the 10 x 10 kitchen layout/style grid, ``--lightwheel-subset N``).

How it runs. The PARENT (this script without ``--child``; needs no Isaac, plain ``python3`` is fine) handles one
scene at a time and launches a FRESH CHILD process per phase through ``$ISAACLAB_PATH/isaaclab.sh -p`` with a hard
wall-clock timeout, killing the whole process group on expiry: a stalled PhysX warmup cannot be interrupted
in-process, so a hung scene becomes a recorded ``fail`` instead of a hung pod. After every scene the parent
rewrites the manifest and the progress report, so a killed pod keeps its partial results; ``--resume`` skips scenes
already in the manifest. Phases per scene (a later one only runs if the earlier one passed):

1. ``sample``  geometry only, no physics: gather the scene's triangle mesh, find floor with a clear torso/head
   volume on a grid (downward Warp ray layers, see ``scripts/scene_spawn_sampling.py`` for the method and why),
   store up to 5 spawn poses best-first, then one LiDAR-pattern sweep from LiDAR height at the first spawn.
2. ``env`` x 4 envs   backend ``build_env`` with the Go2 at the first spawn, ``--steps`` zero-action steps:
   observations finite, base z within a band above the measured floor (catches the Galileo fall-through), tilt,
   drift; records build s, step ms, peak GPU MiB.
3. ``env`` x 16 envs   the same at training scale (short), with step-time and GPU-memory limits.
4. ``render`` (only with ``--render``)   one RGB frame must be non-blank (catches missing MDL materials); taken
   with an Isaac Lab ``Camera`` sensor, not ``omni.replicator``.
5. ``arena probe`` (only with ``--arena-probe N``, first N admitted GRScenes)   Arena backend, ``shared_scene=True``,
   1 env: tests whether per-env cloning of the house caused the Arena warmup stall. Recorded in
   ``metrics['arena_probe']``; it never changes the verdict.

Verdict logic and thresholds (provisional) live in ``scripts/scene_admission_logic.py`` (unit tested).

Pod usage (new stack; data from ``scripts/run_scene_admission_pod.sh``)::

    python3 scripts/validate_scenes.py --source grscenes --limit 8 --resume        # 8 smallest scenes per category
    python3 scripts/validate_scenes.py --source grscenes --resume --arena-probe 3
    python3 scripts/validate_scenes.py --source lightwheel --lightwheel-subset 12 --resume
    python3 scripts/validate_scenes.py --source grscenes --dry-run                 # local, no Isaac

Exit codes: 0 sweep completed (per-scene failures are data, not errors), 1 nothing to run / internal error,
2 bad arguments. Children write their JSON first, flush and ``os._exit`` (``SimulationApp.close()`` swallows
failures and can hang); ``--watchdog_s`` makes each child dump all thread stacks to its log if a stage stalls.
Isaac-side phases have run on pods: the 18-scene admission sweep (2026-10-11/12) and, on the Lab 3.0 GA stack, the
`render` phase on one GRScenes home (data/reports/ga_upgrade/render_summary.txt, 2026-10-25). The sweep was NOT re-run
on GA. The parent logic and the dry run are pure python.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_admission_logic import (
    Thresholds,
    build_row,
    evaluate_env,
    evaluate_sample,
    evaluate_scene,
    scenes_to_run,
    skip_row,
    stratified_kitchen_subset,
    summarize_probe,
    thresholds_dict,
)

from common.cerberus_worlds import registry
from common.cerberus_worlds.admission import (
    STATUS_PASS,
    AdmissionRow,
    load_manifest,
    manifest_path,
    write_manifest,
)

VERSIONS_FILE = Path("/etc/cerberus_versions.json")
DEFAULT_REPORT_DIR = "data/reports/scene_admission"
SOURCES = {"grscenes": "plain_usd", "lightwheel": "arena"}
GRSCENES_CATEGORIES = ("home", "commercial")
STARTUP_ALLOWANCE_S = 180  # app launch + stepping, on top of --build-timeout-s, before a child is killed
# Per-child build budget. Calibrated 2026-10-07 on an A5000 running the same image: on a slow/cold host the
# `env` child spent 143-249 s in build_env and 120 s in reset (86 s / 0.7 s on a fast host), so the earlier
# 300 s default -- 480 s with STARTUP_ALLOWANCE_S -- killed scenes whose physics was fine. 900 s keeps about
# 2x headroom over the worst child observed; a phase that still times out is recorded as `fail` with a
# `timeout ... at stage ...` reason, and a re-run with --retry-failed redoes just those rows.
DEFAULT_BUILD_TIMEOUT_S = 900.0
MESH_BUDGET_S = 300.0  # gathering the scene triangles in the sample phase
SETTLE_FRACTION = 0.25  # steps before the first z/tilt/drift sample (the robot drops from z = 0.4 first)
WARMUP_STEPS = 5  # excluded from step-time statistics
RENDER_FRAMES = 8
RENDER_RESOLUTION = (320, 240)
CAMERA_EYE_HEIGHT_M = 1.0  # eye height above the measured floor for the admission frame (same framing as before)
CAMERA_INIT_MAX_STEPS = 100  # env steps to wait for the sensor's initialization callback (see _phase_render)
PHASES = ("sample", "env", "render")


def _frame_stats(frame: Any) -> dict[str, Any]:
    """Shape/mean/std of an ``(H, W, C)`` RGB frame, without importing numpy in the parent process.

    Duck-typed on purpose: ``validate_scenes.py`` is also run as the PARENT by a bare ``python3`` that may have
    no numpy, so the child passes a numpy array in and everything else here is plain python. A near-zero std is
    what a blank or untextured frame looks like (missing MDL materials), which the caller turns into a failure.
    """
    channels = frame[..., :3]
    return {"frame_shape": list(frame.shape), "frame_mean": float(channels.mean()), "frame_std": float(channels.std())}


@dataclass(frozen=True)
class ScenePlan:
    scene_id: str
    category: str
    backend: str


# ---------------------------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES), default="grscenes")
    p.add_argument("--categories", default=",".join(GRSCENES_CATEGORIES), help="grscenes only, comma separated")
    p.add_argument("--scene", action="append", default=None, help="explicit scene id/ref (repeatable); overrides discovery")
    p.add_argument("--lightwheel-subset", type=int, default=12, help="kitchens to sweep (spread over the 10x10 grid)")
    p.add_argument("--limit", type=int, default=None, help="run at most N scenes PER CATEGORY this invocation, smallest first")
    p.add_argument("--resume", action="store_true", help="skip scenes already in the manifest")
    p.add_argument("--retry-failed", action="store_true", help="with --resume: re-run rows whose status is fail")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--scale-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=200, help="zero-action steps at --num-envs")
    p.add_argument("--scale-steps", type=int, default=50, help="zero-action steps at --scale-envs")
    p.add_argument(
        "--build-timeout-s",
        type=float,
        default=DEFAULT_BUILD_TIMEOUT_S,
        help="per-child build timeout; the child is killed at this + the launch allowance (180 s)",
    )
    p.add_argument("--sample-timeout-s", type=float, default=600.0, help="timeout of the grscenes sample child")
    p.add_argument("--max-runtime-h", type=float, default=None, help="stop starting new scenes after this many hours")
    p.add_argument("--render", action="store_true", help="also require one non-blank RGB frame (needs a rendering driver)")
    p.add_argument(
        "--combined-env-phases",
        action="store_true",
        help="run ONE env launch (scale envs x the long step count) instead of the 4-env + 16-env pair; each launch "
        "pays the full build_env cost, so this cuts roughly a third of the per-scene time. Same checks, both applied "
        "to the one run (see scene_admission_logic.evaluate_scene)",
    )
    p.add_argument("--arena-probe", type=int, default=0, metavar="N", help="Arena shared-scene probe for the first N admitted GRScenes")
    p.add_argument("--isaaclab-sh", default=None, help="default: $ISAACLAB_PATH/isaaclab.sh")
    p.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="progress report, child logs and child JSON")
    p.add_argument("--watchdog_s", type=float, default=240.0, help="children dump all thread stacks every N s (0 disables)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the plan; launch nothing")
    # internal: one child process = one phase
    p.add_argument("--child", choices=PHASES, default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--scene-id", default=None, help=argparse.SUPPRESS)
    p.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    p.add_argument("--spawn", type=float, nargs=4, default=None, metavar=("X", "Y", "YAW", "FLOOR_Z"), help=argparse.SUPPRESS)
    p.add_argument("--shared-scene", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    return p


def check_args(args: argparse.Namespace) -> str | None:
    """An error message for invalid argument values, else None."""
    if args.num_envs < 1 or args.scale_envs < 1 or args.steps < 1 or args.scale_steps < 1:
        return "--num-envs, --scale-envs, --steps and --scale-steps must be >= 1"
    if args.build_timeout_s <= 0 or args.sample_timeout_s <= 0:
        return "timeouts must be > 0"
    if args.limit is not None and args.limit < 1:
        return "--limit must be >= 1"
    if args.arena_probe < 0:
        return "--arena-probe must be >= 0"
    if args.source == "lightwheel" and not 1 <= args.lightwheel_subset <= 100:
        return "--lightwheel-subset must be 1..100"
    unknown = sorted(set(args.categories.split(",")) - set(GRSCENES_CATEGORIES))
    if unknown:
        return f"--categories: unknown {unknown}, valid: {list(GRSCENES_CATEGORIES)}"
    if args.retry_failed and not args.resume:
        return "--retry-failed needs --resume"
    if args.child and not (args.child_out and args.scene_id and args.backend):
        return "--child needs --child-out, --scene-id and --backend"
    return None


# ---------------------------------------------------------------------------------------------------------
# scene discovery (pure python, through the registry)
# ---------------------------------------------------------------------------------------------------------
def plan_scenes(args: argparse.Namespace) -> list[ScenePlan]:
    """The scenes this source offers (all categories / the kitchen subset), or the explicit ``--scene`` refs."""
    backend_name = SOURCES[args.source]
    backend = registry.get_backend(backend_name)
    infos = [i for i in backend.list_scenes() if i.source == args.source]
    by_id = {i.scene_id: i for i in infos}
    if args.scene:
        plans = []
        for ref in args.scene:
            if args.source == "grscenes" and ":" not in ref:
                ref = f"grscenes:{ref}"
            scene_id = backend.normalize_scene_ref(ref)
            info = by_id.get(scene_id)
            plans.append(ScenePlan(scene_id, info.category if info else "", backend_name))
        return plans
    if args.source == "lightwheel":
        by_pair = {(i.metadata["layout_id"], i.metadata["style_id"]): i for i in infos}
        chosen = [by_pair[pair] for pair in stratified_kitchen_subset(args.lightwheel_subset)]
        return [ScenePlan(i.scene_id, i.category, backend_name) for i in chosen]
    wanted = set(args.categories.split(","))
    chosen = [i for i in infos if (i.category or "home") in wanted]
    chosen.sort(key=lambda i: (Path(i.metadata["usd_path"]).stat().st_size, i.scene_id))  # smallest scene first
    return [ScenePlan(i.scene_id, i.category or "home", backend_name) for i in chosen]


def read_tool_versions() -> dict[str, Any]:
    """``/etc/cerberus_versions.json`` (written by the image build) or {} when absent (local runs)."""
    try:
        return json.loads(VERSIONS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(path)
        return True
    except Exception:  # noqa: BLE001 - best effort; callers decide
        traceback.print_exc()
        return False


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


# ---------------------------------------------------------------------------------------------------------
# parent: child processes
# ---------------------------------------------------------------------------------------------------------
def isaaclab_sh(args: argparse.Namespace) -> str:
    if args.isaaclab_sh:
        return args.isaaclab_sh
    return str(Path(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab")) / "isaaclab.sh")


def child_command(
    args: argparse.Namespace, phase: str, scene_id: str, backend: str, out: Path, extra: list[str]
) -> list[str]:
    return [
        isaaclab_sh(args), "-p", str(Path(__file__).resolve()),
        "--child", phase, "--child-out", str(out), "--scene-id", scene_id, "--backend", backend,
        "--watchdog_s", str(args.watchdog_s), *extra,
    ]  # fmt: skip


def run_child(cmd: list[str], timeout_s: float, out: Path, log: Path) -> dict[str, Any]:
    """Run one child to completion or kill its whole process group after `timeout_s`. Returns the child's JSON
    (partial if it was killed) plus ``timed_out``, ``returncode`` and ``wall_s`` measured here."""
    out.unlink(missing_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TERM": "xterm"}
    t0 = time.monotonic()
    timed_out = False
    with open(log, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=60)
    try:
        data = json.loads(out.read_text())
    except (OSError, ValueError):
        data = {}
    data.update(timed_out=timed_out, returncode=proc.returncode, wall_s=round(time.monotonic() - t0, 1))
    return data


PhaseRunner = Callable[..., dict[str, Any]]


def make_phase_runner(args: argparse.Namespace, plan: ScenePlan, work_dir: Path) -> PhaseRunner:
    stem = plan.scene_id.replace(":", "_").replace("/", "_")

    def run_phase(phase: str, extra: list[str] | None = None, *, backend: str | None = None, tag: str = "") -> dict[str, Any]:
        name = f"{stem}_{phase}{tag}"
        timeout = args.build_timeout_s + STARTUP_ALLOWANCE_S
        if phase == "sample" and (backend or plan.backend) == "plain_usd":
            timeout = args.sample_timeout_s
        cmd = child_command(args, phase, plan.scene_id, backend or plan.backend, work_dir / f"{name}.json", extra or [])
        return run_child(cmd, timeout, work_dir / f"{name}.json", work_dir / f"{name}.log")

    return run_phase


def spawn_args(spawn: dict[str, Any]) -> list[str]:
    return ["--spawn", *(repr(float(spawn[k])) for k in ("x", "y", "yaw", "floor_z"))]


def run_scene(
    plan: ScenePlan,
    args: argparse.Namespace,
    th: Thresholds,
    run_phase: PhaseRunner,
    *,
    want_probe: bool,
) -> AdmissionRow:
    """All phases of one scene -> its manifest row. `run_phase(phase, extra, backend=, tag=)` returns a child result."""
    source = args.source
    sample = run_phase("sample")
    env_small = env_scale = render = probe = None
    env_combined = None
    if not evaluate_sample(sample, th):
        spawn = spawn_args(sample["spawns"][0])
        if args.combined_env_phases:
            # One launch at the scale env count for the LONG step count: the stability rules and the scale rules are
            # both applied to it, so the 4-env launch (and its build_env) is not paid for. Strictly stronger than the
            # 16-env/50-step run it replaces, and it removes a third of the per-scene Kit launches.
            env_combined = run_phase(
                "env", [*spawn, "--num-envs", str(args.scale_envs), "--steps", str(args.steps)], tag="_combined"
            )
            stable = not evaluate_env(env_combined, th, "env")
            scaled = not evaluate_env(env_combined, th, "env_scale", check_scale=True)
            if args.render and stable and scaled:
                render = run_phase("render", [*spawn, "--num-envs", "1"])
        else:
            env_small = run_phase("env", [*spawn, "--num-envs", str(args.num_envs), "--steps", str(args.steps)], tag="_4")
            if not evaluate_env(env_small, th, "env4"):
                env_scale = run_phase(
                    "env", [*spawn, "--num-envs", str(args.scale_envs), "--steps", str(args.scale_steps)], tag="_16"
                )
                if args.render and not evaluate_env(env_scale, th, "env16", check_scale=True):
                    render = run_phase("render", [*spawn, "--num-envs", "1"])
    status, _ = evaluate_scene(
        sample, env_small, env_scale, render, th, render_required=args.render, env_combined=env_combined
    )
    if status == STATUS_PASS and want_probe:
        res = run_phase(
            "env", [*spawn, "--num-envs", "1", "--steps", str(args.scale_steps), "--shared-scene"],
            backend="arena", tag="_arena_probe",
        )  # fmt: skip
        probe = summarize_probe(res, th)
    return build_row(
        scene_id=plan.scene_id, source=source, category=plan.category, backend=plan.backend,
        sample=sample, env_small=env_small, env_scale=env_scale, render=render, probe=probe,
        th=th, date_utc=now_utc(), render_required=args.render, env_combined=env_combined,
    )  # fmt: skip


def count_probes(rows: dict[str, AdmissionRow]) -> int:
    return sum(1 for r in rows.values() if "arena_probe" in r.metrics)


def sweep(args: argparse.Namespace, plans: list[ScenePlan], report_path: Path) -> int:
    th = Thresholds()
    versions = {**read_tool_versions(), "admission_thresholds": thresholds_dict(th)}
    try:
        rows = {r.scene_id: r for r in load_manifest(args.source)}
    except FileNotFoundError:
        rows = {}
    plan_by_id = {p.scene_id: p for p in plans}
    todo = scenes_to_run(
        [(p.scene_id, p.category) for p in plans], list(rows.values()),
        resume=args.resume, retry_failed=args.retry_failed, limit=args.limit,
    )  # fmt: skip
    work_dir = _abs(args.report_dir) / args.source
    report: dict[str, Any] = {
        "source": args.source, "started_utc": now_utc(), "manifest": str(manifest_path(args.source)),
        "planned": len(plans), "to_run": len(todo), "done": False, "current": None, "ran": [],
    }  # fmt: skip
    _write_json(report_path, report)
    t_start = time.monotonic()
    for i, scene_id in enumerate(todo, 1):
        if args.max_runtime_h is not None and (time.monotonic() - t_start) / 3600 > args.max_runtime_h:
            print(f"[INFO] --max-runtime-h {args.max_runtime_h} reached; stopping before {scene_id}", flush=True)
            break
        plan = plan_by_id[scene_id]
        report["current"] = scene_id
        _write_json(report_path, report)
        print(f"[INFO] scene {i}/{len(todo)}: {scene_id}", flush=True)
        t0 = time.monotonic()
        info = registry.get_backend(plan.backend).describe_scene(scene_id)
        if info.get("error"):
            row = skip_row(
                scene_id=scene_id, source=args.source, category=plan.category, backend=plan.backend,
                reason=f"scene data missing: {info['error']}"[:300], date_utc=now_utc(),
            )  # fmt: skip
        else:
            want_probe = args.source == "grscenes" and count_probes(rows) < args.arena_probe
            row = run_scene(plan, args, th, make_phase_runner(args, plan, work_dir), want_probe=want_probe)
        rows[scene_id] = row
        write_manifest(args.source, list(rows.values()), tool_versions=versions, generated_utc=now_utc())
        report["ran"].append({"scene_id": scene_id, "status": row.status, "reasons": list(row.reasons), "wall_s": round(time.monotonic() - t0, 1)})
        _write_json(report_path, report)
        print(f"[INFO] {scene_id}: {row.status.upper()} {list(row.reasons)}", flush=True)
    counts = {s: sum(1 for r in rows.values() if r.status == s) for s in ("pass", "fail", "skip")}
    report.update(done=True, current=None, counts=counts)
    _write_json(report_path, report)
    print(f"[INFO] manifest {manifest_path(args.source)}: {counts}", flush=True)
    return 0


def dry_run(args: argparse.Namespace, plans: list[ScenePlan]) -> int:
    th = Thresholds()
    try:
        existing = load_manifest(args.source)
    except FileNotFoundError:
        existing = []
    todo = scenes_to_run(
        [(p.scene_id, p.category) for p in plans], existing,
        resume=args.resume, retry_failed=args.retry_failed, limit=args.limit,
    )  # fmt: skip
    sample_cmd = (
        child_command(args, "sample", todo[0], SOURCES[args.source], Path("<out>.json"), []) if todo else None
    )
    warnings = []
    if not plans:
        warnings.append(f"no {args.source} scenes found; run scripts/run_scene_admission_pod.sh first")
    plan = {
        "source": args.source, "backend": SOURCES[args.source], "manifest": str(manifest_path(args.source)),
        "scenes_planned": len(plans), "scenes_to_run": len(todo), "first_scenes": todo[:5],
        "phases": ["sample", f"env x{args.num_envs}", f"env x{args.scale_envs}"]
        + (["render"] if args.render else [])
        + ([f"arena probe (first {args.arena_probe} admitted)"] if args.arena_probe else []),
        "timeouts_s": {"env": args.build_timeout_s + STARTUP_ALLOWANCE_S, "sample_grscenes": args.sample_timeout_s},
        "isaaclab_sh": isaaclab_sh(args), "example_sample_cmd": sample_cmd,
        "tool_versions": read_tool_versions(), "thresholds": thresholds_dict(th), "warnings": warnings,
    }  # fmt: skip
    print(json.dumps(plan, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------
# child: Isaac side (every isaaclab / pxr / omni / torch import is inside a function)
# ---------------------------------------------------------------------------------------------------------
def _launch_app(enable_cameras: bool):
    # AppLauncher defaults come from a throwaway parser so this works on the Lab 2.3 and 3.0 launchers (3.0 no
    # longer registers --headless / --enable_cameras, but reads them from the namespace; common/cerberus_compat/launcher.py).
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args([])
    app_args.headless = True
    app_args.enable_cameras = enable_cameras  # UNVERIFIED: Lab 3.0 launch_simulation reads args['enable_cameras']
    if hasattr(app_args, "device"):
        app_args.device = "cuda:0"
    return AppLauncher(app_args).app


def _gpu_used_mib() -> float:
    import torch

    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20  # device level: PhysX/Kit allocate outside torch's allocator


def _all_finite(obs: Any) -> bool:
    import torch

    if isinstance(obs, torch.Tensor):
        return bool(torch.isfinite(obs).all()) if obs.is_floating_point() else True
    if isinstance(obs, dict):
        return all(_all_finite(v) for v in obs.values())
    if isinstance(obs, (list, tuple)):
        return all(_all_finite(v) for v in obs)
    return True


def _gather_triangles(usd_stage: Any, root_path: str, skip_names: frozenset[str], budget_s: float):
    """World-frame triangles of every Mesh prim under `root_path` (instance proxies included, subtrees named in
    `skip_names` pruned): ``(points (N, 3) float32, triangles (M, 3) int32)``."""
    import numpy as np
    from pxr import Usd, UsdGeom
    from scene_spawn_sampling import triangulate_faces

    start = usd_stage.GetPseudoRoot() if root_path == "/" else usd_stage.GetPrimAtPath(root_path)
    if not start or not start.IsValid():
        raise RuntimeError(f"scene root {root_path!r} not found on the stage")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    t0 = time.perf_counter()
    points, faces, offset = [], [], 0
    prims = iter(Usd.PrimRange(start, Usd.TraverseInstanceProxies()))  # UNVERIFIED: instance proxies of GRScenes objects
    for prim in prims:
        if prim.GetName() in skip_names:
            prims.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if time.perf_counter() - t0 > budget_s:
            raise TimeoutError(f"gathering scene triangles exceeded {budget_s:.0f} s ({offset} vertices so far)")
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = (a.Get() for a in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()))
        if pts is None or counts is None or idx is None or len(pts) == 0:
            continue
        m = np.array(xforms.GetLocalToWorldTransform(prim)).reshape(4, 4)  # Gf row-vector convention: p' = p M
        world = np.asarray(pts, dtype=np.float64) @ m[:3, :3] + m[3, :3]
        tris = triangulate_faces(np.asarray(counts), np.asarray(idx))
        # winding is counter-clockwise-outward for rightHanded meshes; leftHanded or a mirroring transform flips it
        if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != (np.linalg.det(m[:3, :3]) < 0):
            tris = tris[:, [0, 2, 1]]
        points.append(world.astype(np.float32))
        faces.append(tris + offset)
        offset += len(pts)
    if not points:
        raise RuntimeError(f"no Mesh prims under {root_path!r}")
    return np.concatenate(points), np.concatenate(faces)


def _phase_sample(args: argparse.Namespace, rec: dict, stage: Callable[[str], None]) -> None:
    import numpy as np
    import torch
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom
    from scene_spawn_sampling import (
        lidar_ray_dirs,
        next_cast_z,
        plan_grid,
        plan_spawns,
        robust_xy_bounds,
    )

    th = Thresholds()
    backend = registry.get_backend(args.backend)
    device = "cuda:0"
    origin = np.zeros(3)
    if args.backend == "plain_usd":
        from pxr import Usd

        stage("open_stage")
        usd_stage = Usd.Stage.Open(backend.describe_scene(args.scene_id)["usd_path"])
        root, skip = "/", frozenset()
    else:  # Arena owns the scene USD, so build a 1-env env and read its live stage (the robot subtree is skipped)
        from common.cerberus_worlds.arena_assets.go2_embodiment import GO2_EMBODIMENT

        stage("build_env")
        backend.register_embodiment(GO2_EMBODIMENT)
        env = backend.build_env(backend.load_scene(args.scene_id), GO2_EMBODIMENT, None, 1)
        usd_stage = env.unwrapped.sim.stage
        origin = env.unwrapped.scene.env_origins[0].cpu().numpy().astype(np.float64)
        root, skip = "/World/envs/env_0", frozenset({"Robot"})  # UNVERIFIED: Arena places the scene under env_0
    rec["up_axis"] = str(UsdGeom.GetStageUpAxis(usd_stage))
    rec["meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(usd_stage)

    stage("gather_meshes")
    t0 = time.perf_counter()
    points, triangles = _gather_triangles(usd_stage, root, skip, MESH_BUDGET_S)
    stage("build_bvh")
    mesh = convert_to_warp_mesh(points, triangles, device=device)
    rec["triangles"] = len(triangles)
    rec["mesh_build_s"] = round(time.perf_counter() - t0, 2)

    stage("cast_floor")
    lo, hi = robust_xy_bounds(points)
    spec = plan_grid(lo, hi, th)
    z_top, z_bottom = float(points[:, 2].max()) + 0.5, float(points[:, 2].min()) - 0.5
    xy = torch.as_tensor(spec.xy(), dtype=torch.float32, device=device)
    n = xy.shape[0]
    down = torch.tensor([0.0, 0.0, -1.0], device=device).expand(n, 3).contiguous()
    hit_z = np.full((th.max_hits_per_column, n), np.nan, dtype=np.float32)
    hit_nz = np.full_like(hit_z, np.nan)
    start_z = np.full(n, z_top, dtype=np.float32)
    for k in range(th.max_hits_per_column):
        starts = torch.cat([xy, torch.as_tensor(start_z, device=device)[:, None]], dim=1)
        # warp's mesh_query_ray returns both faces and the unit winding-based face normal (warp/native/intersect.h);
        # the layered casts and the signed floor test rely on that
        hits, dist, normal, _ = raycast_mesh(
            starts, down, mesh, max_dist=z_top - z_bottom + 1.0, return_distance=True, return_normal=True
        )
        found = torch.isfinite(dist)
        if not bool(found.any()):
            break
        nan = torch.full_like(dist, float("nan"))
        hit_z[k] = torch.where(found, hits[:, 2], nan).cpu().numpy()
        hit_nz[k] = torch.where(found, normal[:, 2], nan).cpu().numpy()
        start_z = next_cast_z(hit_z[k], z_bottom - 1.0).astype(np.float32)
    plan = plan_spawns(hit_z, hit_nz, spec, th)
    rec.update(
        level_z=None if plan.level_z is None else round(plan.level_z - float(origin[2]), 3),
        footprint_m=[round(float(hi[0] - lo[0]), 1), round(float(hi[1] - lo[1]), 1)],
        grid_cells=int(n), cell_m=round(spec.cell, 3), free_cells=plan.free_cells, notes=plan.notes,
        spawns=[
            {
                "x": round(s.x - float(origin[0]), 3), "y": round(s.y - float(origin[1]), 3), "yaw": round(s.yaw, 4),
                "floor_z": round(s.floor_z - float(origin[2]), 3), "open_radius_m": round(s.open_radius_m, 2),
                "roofed": s.roofed,
            }
            for s in plan.spawns
        ],
    )  # fmt: skip
    if not plan.spawns:
        return
    stage("lidar_sweep")
    try:
        first = plan.spawns[0]
        dirs = torch.as_tensor(lidar_ray_dirs(th), dtype=torch.float32, device=device)
        sensor = torch.tensor([first.x, first.y, first.floor_z + th.lidar_height_above_floor_m], device=device)
        _, dist, _, _ = raycast_mesh(sensor.expand(len(dirs), 3).contiguous(), dirs, mesh, max_dist=th.lidar_range_m, return_distance=True)
        rec["lidar_rays"] = len(dirs)
        rec["lidar_hits"] = int(torch.isfinite(dist).sum())
    except Exception as e:  # noqa: BLE001 - recorded as a lidar failure, the spawns above stay valid
        traceback.print_exc()
        rec["lidar_error"] = repr(e)


def _build_go2_env(args: argparse.Namespace, rec: dict, stage: Callable[[str], None]):
    """Backend env with the Go2 at the spawn pose; returns ``(env, backend)``."""
    from common.cerberus_worlds.arena_assets.go2_embodiment import GO2_EMBODIMENT

    backend = registry.get_backend(args.backend)
    x, y, yaw, _ = args.spawn
    backend.configure(shared_scene=args.shared_scene, robot_pose=(x, y, yaw))
    rec["shared_scene"] = bool(backend.shared_scene)
    stage("register_embodiment")
    backend.register_embodiment(GO2_EMBODIMENT)
    stage("load_scene")
    scene = backend.load_scene(args.scene_id)
    stage("build_env")
    t0 = time.perf_counter()
    env = backend.build_env(scene, GO2_EMBODIMENT, None, args.num_envs)
    rec["build_s"] = round(time.perf_counter() - t0, 2)
    rec["num_envs"] = args.num_envs
    return env


def _phase_env(args: argparse.Namespace, rec: dict, stage: Callable[[str], None]) -> None:
    import torch

    from common.cerberus_compat.isaac_data import as_tensor

    rec["steps_requested"] = args.steps
    env = _build_go2_env(args, rec, stage)
    rec["gpu_name"] = torch.cuda.get_device_name(0)
    peak_mib = _gpu_used_mib()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    stage("reset")
    t0 = time.perf_counter()
    obs, _ = env.reset()
    rec["reset_s"] = round(time.perf_counter() - t0, 2)
    finite = _all_finite(obs)
    x, y, _, floor_z = args.spawn
    spawn_xy = torch.tensor([x, y], device=unwrapped.device)
    actions = torch.zeros((unwrapped.num_envs, int(unwrapped.action_manager.total_action_dim)), device=unwrapped.device)
    settle = int(args.steps * SETTLE_FRACTION)
    step_ms: list[float] = []
    z_rel: list[float] = []
    tilt: list[float] = []
    drift: list[float] = []
    stage("step")
    done = 0
    with torch.inference_mode():
        for i in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            obs, _, _, _, _ = env.step(actions)
            torch.cuda.synchronize()
            if i >= WARMUP_STEPS:
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            done += 1
            finite = finite and _all_finite(obs)
            if not finite:
                break
            if i % 10 == 0:
                peak_mib = max(peak_mib, _gpu_used_mib())
            if i >= settle and (i % 5 == 0 or i == args.steps - 1):
                pos = as_tensor(robot.data.root_pos_w) - unwrapped.scene.env_origins
                z_rel += (pos[:, 2] - floor_z).tolist()
                drift += torch.linalg.norm(pos[:, :2] - spawn_xy, dim=-1).tolist()
                gravity_z = as_tensor(robot.data.projected_gravity_b)[:, 2]
                tilt += torch.rad2deg(torch.acos((-gravity_z).clamp(-1.0, 1.0))).tolist()
    rec.update(steps_done=done, obs_finite=bool(finite), gpu_mem_peak_mib=round(peak_mib, 1))
    if step_ms:
        ordered = sorted(step_ms)
        rec["step_ms_mean"] = round(sum(step_ms) / len(step_ms), 2)
        rec["step_ms_p95"] = round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 2)
    if z_rel:
        rec.update(
            root_z_rel_min=round(min(z_rel), 3), root_z_rel_max=round(max(z_rel), 3),
            tilt_deg_max=round(max(tilt), 2), drift_m_max=round(max(drift), 3),
        )  # fmt: skip


def _admission_camera_cfg():
    """A pinhole camera prim for the admission frame -- the repo's own pattern (sensor_configs.py's SHOWCASE_CAMERA_CFG)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="/World/admission_camera",  # standalone: no {ENV_REGEX_NS}, one camera for one env (--num-envs 1)
        update_period=0.0,  # take a frame on every update() call
        height=RENDER_RESOLUTION[1],
        width=RENDER_RESOLUTION[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
        ),
    )  # fmt: skip


def _phase_render(args: argparse.Namespace, rec: dict, stage: Callable[[str], None]) -> None:
    """One RGB frame from an Isaac Lab ``Camera`` sensor.

    Replaces the earlier ``omni.replicator`` implementation, whose ``rep.orchestrator.step()`` does not work in
    this image ("Orchestrator failed to schedule a capture after 1000 app updates", 2026-10-06 admission run).
    Isaac Lab's own ``Camera`` sensor is proven here (the SLAM loop's rigs render), so this uses it:

    * ``Camera(cfg)`` authors its prim in ``__init__`` (camera.py spawns when ``cfg.spawn`` is set);
    * the sensor's render product is built by a physics-manager callback on the next sim step, so the first
      ``env.step()`` must happen before the pose is set (``set_world_poses_from_view`` needs the view);
    * the first RTX frames are empty, so ``RENDER_FRAMES`` frames are taken and the last is used.

    VERIFIED on a pod (2026-10-11, A5000, release/3.0.0): exit 0, ok=True, frame (240, 320, 3), frame_std 49.98
    against a threshold of 2.0, and the PNG is a properly lit, textured room -- not the flat red of a missing-MDL
    frame. Three things this cost to learn, all now handled below: a hand-written child run must pass a fully
    qualified scene id (``grscenes:<folder>``); the sensor only initializes on ``PhysicsEvent.PHYSICS_READY``, so
    physics is reset after the camera is created; and the single camera prim requires a single env.
    """
    import torch
    from isaaclab.sensors import Camera

    from common.cerberus_compat.isaac_data import as_tensor

    # One camera prim at one pose means one env: this phase always renders a single env. (The parent passes
    # --num-envs 1 too; the env phases are the ones that scale.) Without forcing it, the default of 4 makes Camera's
    # initialization fail with "Number of camera prims in the view (1) does not match the number of environments (4)"
    # -- hit on the 2026-10-11 pod run.
    args.num_envs = 1
    env = _build_go2_env(args, rec, stage)
    unwrapped = env.unwrapped
    actions = torch.zeros((unwrapped.num_envs, int(unwrapped.action_manager.total_action_dim)), device=unwrapped.device)
    frame = None
    with torch.inference_mode():
        stage("camera")
        camera = Camera(_admission_camera_cfg())
        # A sensor initializes only on PhysicsEvent.PHYSICS_READY, which is emitted once when physics becomes ready.
        # The env was already built (and its scene sensors initialized) by this point, so a camera created now would
        # NEVER initialize -- confirmed on release/3.0.0: 100 env.step()s left is_initialized False, and
        # set_world_poses_from_view then raised `AttributeError: 'Camera' object has no attribute '_device'`
        # (_device is set in _initialize_impl). SimulationContext.reset() re-runs the physics manager's reset+play,
        # re-emitting PHYSICS_READY, which initializes this sensor. No Isaac Lab file is patched.
        stage("physics_reset")
        unwrapped.sim.reset()
        if not camera.is_initialized:
            for _ in range(CAMERA_INIT_MAX_STEPS):
                env.step(actions)
                if camera.is_initialized:
                    break
        if not camera.is_initialized:
            raise RuntimeError(
                "camera still uninitialized after SimulationContext.reset() and "
                f"{CAMERA_INIT_MAX_STEPS} env steps: the sensor's PHYSICS_READY callback never ran"
            )
        stage("reset")
        env.reset()
        x, y, yaw, floor_z = args.spawn
        eye = torch.tensor([[x, y, floor_z + CAMERA_EYE_HEIGHT_M]], device=unwrapped.device)
        target = torch.tensor([[x + math.cos(yaw), y + math.sin(yaw), eye[0, 2].item()]], device=unwrapped.device)
        camera.set_world_poses_from_view(eye, target)
        stage("render")
        for _ in range(RENDER_FRAMES):
            env.step(actions)
            camera.update(unwrapped.step_dt)
            output = camera.data.output or {}
            frame = as_tensor(output.get("rgb")) if "rgb" in output else None
    if frame is None or frame.numel() == 0:
        raise RuntimeError("Camera sensor returned no RGB data")
    arr = frame[0, ..., :3].to(torch.uint8).cpu().numpy()
    rec.update(_frame_stats(arr))
    # Kept next to the child JSON so a human can tell textured from flat-red (the statistics cannot).
    import imageio.v2 as imageio

    png = Path(args.child_out).with_suffix(".png")
    imageio.imwrite(png, arr)
    rec["frame_png"] = str(png)


CHILD_PHASES: dict[str, Callable[[argparse.Namespace, dict, Callable[[str], None]], None]] = {
    "sample": _phase_sample,
    "env": _phase_env,
    "render": _phase_render,
}


def child_main(args: argparse.Namespace) -> int:
    """One phase in this process. Writes the result JSON first (also at every stage), flushes, ``os._exit``s."""
    out = Path(args.child_out)
    rec: dict[str, Any] = {"phase": args.child, "scene_id": args.scene_id, "backend": args.backend, "ok": False, "error": None, "stage": "launch"}
    t0 = time.perf_counter()

    def stage(name: str) -> None:
        rec["stage"] = name
        rec[f"t_{name}"] = round(time.perf_counter() - t0, 1)
        _write_json(out, rec)
        print(f"[INFO] stage: {name}", flush=True)

    code = 1
    try:
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        _write_json(out, rec)
        registry.get_backend(args.backend).prepare_scene_env(args.scene_id)  # MDL_SYSTEM_PATH before Kit starts
        # Always the camera-enabled (RTX) Kit experience, even for the physics-only phases. On the v6-rel image every
        # Isaac script launched WITHOUT it died within ~1 s ("[crash] appState = startup" dumps, no output, parent
        # stuck on scene 1/8 with the GPU idle), while the SLAM loop, which sets enable_cameras, ran its whole
        # mission on the same pod. UNVERIFIED: that this is the cause (inferred from 5 crashing no-camera launches
        # vs 1 working camera launch; the crash dumps were lost with the pod).
        _launch_app(enable_cameras=True)
        stage("launched")
        CHILD_PHASES[args.child](args, rec, stage)
        rec["ok"] = True
        code = 0
    except BaseException as e:  # noqa: BLE001 - includes SystemExit/KeyboardInterrupt from Kit
        traceback.print_exc()
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        faulthandler.cancel_dump_traceback_later()
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        _write_json(out, rec)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # never SimulationApp.close(): exits 0 on failure and has hung >10 min
    return code  # unreachable; keeps the type checker honest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)  # exits 2 on bad args
    problem = check_args(args)
    if problem:
        print(f"[ERROR] {problem}", file=sys.stderr)
        return 2
    if args.child:
        return child_main(args)
    try:
        plans = plan_scenes(args)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        return dry_run(args, plans)
    if not plans:
        print(f"[ERROR] no {args.source} scenes found (run scripts/run_scene_admission_pod.sh first)", file=sys.stderr)
        return 1
    report_path = _abs(args.report_dir) / f"validate_scenes_{args.source}.json"
    try:
        return sweep(args, plans, report_path)
    except Exception:  # noqa: BLE001 - the report and manifest on disk already hold every finished scene
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
