"""W1 spike: stock Go2 in a scene, built through a world backend, stepped with zero actions at N parallel envs.

Gate A: Arena hosts a Go2 in a GRScenes home (`--scene grscenes:<folder>`) at 16+ envs with GPU physics and no
patching. Gate B: the same with a Lightwheel kitchen (`--scene lightwheel:<layout>:<style>`). `--scene stock` is the
control run (an Arena library background, or a ground plane on the plain_usd backend). Run `--backend plain_usd`
for the no-Arena fallback.

Run on a pod (new stack: Isaac Sim 6.1.0 container, Python 3.12, Lab 3.0 line, Arena `main` installed, see
scripts/fetch_arena.py):

    $ISAACLAB_PATH/isaaclab.sh -p scripts/arena_spike_go2.py --scene stock --headless
    $ISAACLAB_PATH/isaaclab.sh -p scripts/arena_spike_go2.py --scene grscenes:<id>_usd --num_envs 16 --headless
    python scripts/arena_spike_go2.py --scene lightwheel:1:1 --dry-run        # local: prints the plan, no Isaac

Flow: launch app -> register embodiment -> load scene -> build env -> `env.reset()` -> N zero-action steps, checking
observations are finite and timing each step -> JSON report written FIRST -> flush -> `os._exit(code)`.
`SimulationApp.close()` is never called: it exits 0 whatever happened and has hung >10 min at shutdown (see
docs/runpod.md "Operational gotchas" and scripts/spawn_hold_go2_arx5.py). A faulthandler watchdog (`--watchdog_s`)
dumps all thread stacks if a stage stalls, and a progress report is written at every stage so a hang or kill
leaves evidence on disk.

Backends are only reached through `common.cerberus_worlds.registry.get_backend` (architecture rule, enforced by
common/cerberus_worlds/test_no_direct_backend_imports.py).

Exit codes: 0 pass, 1 fail, 2 bad args. `--dry-run`: 0 if the plan resolves (scene data present), 1 if not.
# UNVERIFIED: the Isaac-side flow has not been run (needs a GPU pod); the dry-run path is pure python.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_compat.reset import reset_all
from common.cerberus_description.go2_arx5_spec import GO2_ARX5_EMBODIMENT
from common.cerberus_worlds import registry
from common.cerberus_worlds.arena_assets.go2_embodiment import (
    GO2_EMBODIMENT,
)

DEFAULT_REPORT = "data/reports/arena_spike_go2.json"
EMBODIMENTS = {"go2": GO2_EMBODIMENT, "go2_arx5": GO2_ARX5_EMBODIMENT}
WARMUP_STEPS = 5  # excluded from the step-time statistics (kernel/JIT warm-up)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scene",
        default="stock",
        help="stock | grscenes:<folder> | lightwheel:<layout 1-10>:<style 1-10>",
    )
    parser.add_argument("--backend", default="arena", choices=["arena", "plain_usd"])
    parser.add_argument("--embodiment", default="go2", choices=sorted(EMBODIMENTS))
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument(
        "--headless", action="store_true", help="no GUI (always use on a pod)"
    )
    parser.add_argument(
        "--watchdog_s",
        type=float,
        default=900.0,
        help="dump all thread stacks every N s if hung (0 disables)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="print the resolved plan; import no Isaac",
    )
    parser.add_argument(
        "--env_spacing",
        type=float,
        default=None,
        help="metres between envs (default: backend choice)",
    )
    parser.add_argument(
        "--shared_scene",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="spawn the GRScenes USD once for all envs (default: backend choice; plain_usd=True, arena=False)",
    )
    parser.add_argument(
        "--robot_pose",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "YAW"),
        help="robot start in the env frame",
    )
    return parser


def resolve_plan(args: argparse.Namespace) -> dict:
    """Pure python: resolve the scene and backend options. Returns the plan dict (never imports Isaac)."""
    backend = registry.get_backend(args.backend)
    scene_id = backend.normalize_scene_ref(args.scene)
    backend.configure(
        env_spacing=args.env_spacing,
        shared_scene=args.shared_scene,
        robot_pose=args.robot_pose,
    )
    info = backend.describe_scene(scene_id)
    problems = []
    if info.get("error"):
        problems.append(info["error"])
    if args.embodiment == "go2_arx5" and args.backend == "arena":
        problems.append("go2_arx5 Arena embodiment is W2 (use --backend plain_usd)")
    return {
        "backend": args.backend,
        "backend_class": type(backend).__name__,
        "embodiment": args.embodiment,
        "joint_groups": {
            k: list(v) for k, v in EMBODIMENTS[args.embodiment].joint_groups.items()
        },
        "scene_ref": args.scene,
        "scene_id": scene_id,
        "scene": info,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "headless": args.headless,
        "report": args.report,
        "env_spacing": getattr(backend, "env_spacing", None),
        "shared_scene": getattr(backend, "shared_scene", None),
        "robot_pose": getattr(backend, "robot_pose", None),
        "problems": problems,
    }


def _versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for pkg in ("isaaclab", "isaacsim", "torch", "isaaclab_arena"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    meta = _REPO_ROOT / "third_party" / "IsaacLab-Arena" / ".cerberus_fetch.json"
    if meta.is_file():
        try:
            out["arena_fetch"] = json.loads(meta.read_text())
        except json.JSONDecodeError:
            pass
    return out


def _write_report(path: Path, report: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        return True
    except Exception:  # noqa: BLE001 - best effort, caller decides
        traceback.print_exc()
        return False


def _report_path(args: argparse.Namespace) -> Path:
    p = Path(args.report)
    return p if p.is_absolute() else _REPO_ROOT / p


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on bad args
    if args.num_envs < 1 or args.steps < 1:
        print("[ERROR] --num_envs and --steps must be >= 1", file=sys.stderr)
        return 2
    try:
        plan = resolve_plan(args)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        for problem in plan["problems"]:
            print(f"[PROBLEM] {problem}", file=sys.stderr)
        return 1 if plan["problems"] else 0

    # MDL_SYSTEM_PATH must be in the environment before Kit starts.
    registry.get_backend(args.backend).prepare_scene_env(plan["scene_id"])

    # Isaac imports below this line only. AppLauncher defaults come from a throwaway parser so the same code
    # works on the Lab 2.3 and 3.0 launchers (3.0 deprecates --headless but still accepts it).
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args([])
    app_args.headless = bool(args.headless)
    if hasattr(app_args, "device"):
        app_args.device = "cuda:0"
    simulation_app = AppLauncher(app_args).app

    report_path = _report_path(args)
    progress: dict = {"stage": "launched"}
    report = {
        "gate": "W1 Arena spike (Go2)",
        "passed": False,
        "failure_reasons": ["incomplete (still running or killed)"],
        "plan": plan,
        "versions": _versions(),
        "progress": progress,
        "metrics": {},
    }
    exit_code = 1
    try:
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        _write_report(report_path, report)
        try:
            _run(args, plan, report, report_path)
        except BaseException as e:  # noqa: BLE001 - includes SystemExit/KeyboardInterrupt
            traceback.print_exc()
            report["passed"] = False
            report["failure_reasons"] = [
                f"exception at stage {progress.get('stage')!r}: {e!r}"
            ]
            report["error"] = traceback.format_exc()
        wrote = _write_report(report_path, report)  # report FIRST, before any teardown
        if not wrote:
            report["passed"] = False
        print(f"[INFO] report: {report_path}", flush=True)
        print(
            f"[INFO] {'PASS' if report['passed'] else 'FAIL'}: {report['failure_reasons']}",
            flush=True,
        )
        exit_code = 0 if report["passed"] else 1
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        _finish(simulation_app, exit_code)
    return exit_code  # only reached if _finish returns (tests)


def _finish(simulation_app, exit_code: int) -> None:
    """Hard-exit; never calls simulation_app.close() (exits 0 on failure, hangs >10 min at shutdown)."""
    del simulation_app
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001, S110 - nothing sensible to do if flushing fails while exiting
        pass
    os._exit(exit_code)


def _iter_tensors(obj, path: str = "obs"):
    """Yield (path, tensor) for every tensor in a nested dict/tuple/list observation."""
    import torch

    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_tensors(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{path}/{i}")


def _obs_finite(obs) -> tuple[bool, dict]:
    """(all finite, {path: shape}) for a nested observation."""
    import torch

    shapes, ok = {}, True
    for path, tensor in _iter_tensors(obs):
        shapes[path] = list(tensor.shape)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            ok = False
    return ok, shapes


def _run(args: argparse.Namespace, plan: dict, report: dict, report_path: Path) -> None:
    """Fill `report` in place. Raises on failure (main records the exception)."""
    import torch

    from common.cerberus_worlds.arena_assets.go2_embodiment import _to_torch

    progress, metrics = report["progress"], report["metrics"]
    reasons: list[str] = []

    def stage(name: str) -> None:
        progress["stage"] = name
        progress[f"t_{name}"] = round(time.perf_counter() - t_start, 2)
        _write_report(report_path, report)
        print(f"[INFO] stage: {name}", flush=True)

    t_start = time.perf_counter()
    embodiment = EMBODIMENTS[args.embodiment]
    backend = registry.get_backend(args.backend)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stage("register_embodiment")
    backend.register_embodiment(embodiment)
    stage("load_scene")
    scene = backend.load_scene(plan["scene_id"])
    stage("build_env")
    t0 = time.perf_counter()
    env = backend.build_env(scene, embodiment, None, args.num_envs)
    metrics["build_env_s"] = round(time.perf_counter() - t0, 2)
    unwrapped = env.unwrapped
    stage("reset")
    t0 = time.perf_counter()
    obs, _ = reset_all(env)
    metrics["reset_s"] = round(time.perf_counter() - t0, 2)

    # --- device / physics -------------------------------------------------------------------------
    robot = unwrapped.scene["robot"]
    joint_pos = _to_torch(robot.data.joint_pos)
    sim_device = str(getattr(unwrapped.sim, "device", unwrapped.device))
    gpu_dynamics = None
    physx_scene: dict = {}
    try:  # Lab 2.3.2 accessor
        gpu_dynamics = bool(
            unwrapped.sim.get_physics_context().is_gpu_dynamics_enabled()
        )
    except Exception:  # noqa: BLE001
        pass
    try:  # Lab 3.0 line: PhysxManager writes these on the physics scene prim (physx_manager.py, "gpu dynamics")
        prim = unwrapped.sim.stage.GetPrimAtPath(unwrapped.sim.cfg.physics_prim_path)
        for key in ("physxScene:enableGPUDynamics", "physxScene:broadphaseType"):
            attr = prim.GetAttribute(key)
            physx_scene[key] = attr.Get() if attr and attr.IsValid() else None
        if gpu_dynamics is None and physx_scene.get("physxScene:enableGPUDynamics") is not None:
            gpu_dynamics = bool(physx_scene["physxScene:enableGPUDynamics"])
    except Exception as e:  # noqa: BLE001
        physx_scene["error"] = repr(e)
    metrics.update(
        {
            "num_envs": int(unwrapped.num_envs),
            "env_device": str(unwrapped.device),
            "sim_device": sim_device,
            "robot_data_device": str(joint_pos.device),
            "gpu_dynamics_enabled": gpu_dynamics,
            "physx_scene": physx_scene,
            "step_dt_s": float(getattr(unwrapped, "step_dt", float("nan"))),
        }
    )
    on_gpu = "cuda" in str(unwrapped.device) and joint_pos.device.type == "cuda"
    metrics["gpu_physics_device_confirmed"] = bool(on_gpu and gpu_dynamics is not False)
    if not metrics["gpu_physics_device_confirmed"]:
        reasons.append(
            f"physics not confirmed on GPU (env={unwrapped.device}, data={joint_pos.device}, gpu_dynamics={gpu_dynamics})"
        )
    if metrics["num_envs"] != args.num_envs:
        reasons.append(f"num_envs {metrics['num_envs']} != requested {args.num_envs}")

    # --- name addressing ----------------------------------------------------------------------------
    try:
        legs = list(embodiment.joint_groups["legs"])
        _, found = robot.find_joints(legs, preserve_order=True)
        metrics["robot_num_joints"] = len(robot.joint_names)
        metrics["legs_found_by_name_in_order"] = list(found) == legs
        if list(found) != legs:
            reasons.append("12 leg joints not found by name / in order")
    except Exception as e:  # noqa: BLE001
        reasons.append(f"find_joints failed: {e!r}")

    # --- observations and stepping ----------------------------------------------------------------
    ok, shapes = _obs_finite(obs)
    metrics["obs_shapes"] = shapes
    metrics["obs_finite_at_reset"] = ok
    if not ok:
        reasons.append("non-finite observation after reset")
    if not shapes:
        reasons.append("no tensor observations returned")

    action_dim = int(unwrapped.action_manager.total_action_dim)
    metrics["action_dim"] = action_dim
    actions = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)
    stage("step")
    step_ms: list[float] = []
    steps_done = 0
    all_finite = True
    peak_dev_mb = 0.0
    z_trace: list[float] = []
    with torch.inference_mode():
        for i in range(args.steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            obs, _, _, _, _ = env.step(actions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            steps_done += 1
            if i >= WARMUP_STEPS:
                step_ms.append(dt_ms)
            finite, _ = _obs_finite(obs)
            if torch.cuda.is_available() and i % 10 == 0:
                free_b, total_b = torch.cuda.mem_get_info()
                peak_dev_mb = max(peak_dev_mb, (total_b - free_b) / 2**20)
            if i % 20 == 0 or i == args.steps - 1:
                z_trace.append(
                    round(float(_to_torch(robot.data.root_pos_w)[:, 2].mean()), 4)
                )
            if not finite:
                all_finite = False
                reasons.append(f"non-finite observation at step {i}")
                break
    metrics["steps_done"] = steps_done
    metrics["obs_finite_all_steps"] = all_finite
    metrics["gpu_mem_peak_device_sampled_mb"] = round(peak_dev_mb, 1)
    metrics["robot_z_mean_trace_every_20_steps"] = z_trace
    if steps_done != args.steps:
        reasons.append(f"only {steps_done}/{args.steps} steps completed")
    if step_ms:
        metrics["step_ms_mean"] = round(sum(step_ms) / len(step_ms), 3)
        metrics["step_ms_p50"] = round(_percentile(step_ms, 0.5), 3)
        metrics["step_ms_p95"] = round(_percentile(step_ms, 0.95), 3)
        metrics["env_steps_per_s"] = round(
            unwrapped.num_envs * 1000.0 / metrics["step_ms_mean"], 1
        )
    robot_z = _to_torch(robot.data.root_pos_w)[:, 2]
    metrics["robot_z_min_max"] = [float(robot_z.min()), float(robot_z.max())]
    if not bool(torch.isfinite(robot_z).all()):
        reasons.append("non-finite robot height")
    try:  # standing check: base height above the feet (a standing Go2 is ~0.25-0.35 m, a collapsed one ~0.1 m)
        foot_ids = [i for i, n in enumerate(robot.body_names) if n.endswith("_foot")]
        if foot_ids:
            foot_z = _to_torch(robot.data.body_pos_w)[:, foot_ids, 2]
            metrics["foot_z_mean"] = round(float(foot_z.mean()), 4)
            metrics["base_minus_foot_z_mean"] = round(
                float((robot_z - foot_z.mean(dim=1)).mean()), 4
            )
            # Use the measurement above instead of only reporting it: a Go2 standing on a floor keeps its base
            # 0.25-0.35 m above its feet, a collapsed one ~0.1 m. Both 2026-10-05 EA and 2026-11-06 reported
            # passed=true with a base BELOW the feet (the robot sunk through the Galileo background floor), so a
            # PASS that ignores this is a rubber stamp. Scene-relative on purpose: absolute root z is meaningless
            # on stock backgrounds whose floor sits ~0.5-0.8 m below the origin.
            if metrics["base_minus_foot_z_mean"] <= 0.15:
                reasons.append(
                    f"base is {metrics['base_minus_foot_z_mean']} m above the feet (standing Go2: 0.25-0.35)"
                )
    except Exception as e:  # noqa: BLE001
        metrics["foot_z_error"] = repr(e)

    # --- memory ----------------------------------------------------------------------------------------
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        metrics["gpu_mem_peak_torch_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 2**20, 1
        )
        metrics["gpu_mem_peak_torch_reserved_mb"] = round(
            torch.cuda.max_memory_reserved() / 2**20, 1
        )
        # PhysX/Kit allocate outside torch's allocator: device-level used memory is the honest number.
        metrics["gpu_mem_used_device_mb"] = round((total - free) / 2**20, 1)
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        reasons.append("CUDA not available")

    metrics["total_s"] = round(time.perf_counter() - t_start, 2)
    report["failure_reasons"] = reasons
    report["passed"] = not reasons
    progress["stage"] = "done"


if __name__ == "__main__":
    sys.exit(main())
