"""W1.5a gate: GO2-ARX5 (20 DoF: 12 legs, joint1..6, joint7/joint8 fingers) spawns, stands with a stowed arm,
joints addressable by name, gripper opens and closes.

Headless physics-only smoke test (no training, no policy). Holds the default pose
at N parallel envs for --duration_s and checks height / tilt / joint error / drift /
name addressing / arm mass; then a gripper phase commands both fingers closed, settles, records the positions
against the targets, commands them open, settles and records again. `--fix_base` fixes the Go2 base to the world
(fixed-base demos). Runs on a GPU pod: OLD stack (Sim 5.1.0 / Lab 2.3.2) or NEW stack (Sim 6.1 / Lab 3.0 line).

Everything at module level is pure python (import-safe without Isaac); isaaclab / omni
are imported only inside main(), after AppLauncher has started.

Exit codes: 0 pass, 1 fail, 2 bad args.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_compat.isaac_data import as_tensor, physx_view
from common.cerberus_description.go2_arx5_spec import (
    ARX5_BODY_NAMES,
    ARX5_EEF_BODY,
    GO2_ARX5_EMBODIMENT,
    GO2_ARX5_USD_ENV,
    GRIPPER_CLOSED_M,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_M,
    fused_usd_path,
)

SIM_DT = 0.005
SETTLE_WINDOW_S = 2.0
EXPECTED_DOF = 20
DEFAULT_GRIPPER_SETTLE_S = 1.0
DEFAULT_REPORT = "data/reports/go2_arx5_hold.json"


@dataclass(frozen=True)
class HoldThresholds:
    min_base_height: float = 0.15
    max_base_height: float = 0.45
    max_tilt_deg: float = 25.0
    # Arm bound is the real "stowed arm holds its pose" check. Leg bound is deliberately loose:
    # stock Go2 PD gains (stiffness 25) sag under body weight with no policy (observed 0.42-0.66 rad).
    # Both are provisional pending calibration.
    max_arm_joint_err_rad: float = 0.10
    max_leg_joint_err_rad: float = 0.80
    max_settle_drift_m: float = 0.02
    arm_mass_range: tuple[float, float] = (3.2, 3.8)
    # PROVISIONAL gripper tolerances (m, max over envs and fingers of |position - target| after the settle time).
    # Not calibrated: the gripper gains are provisional too. Closed target 0.0 is the joint limit, open is 0.044.
    max_gripper_closed_err_m: float = 0.005
    max_gripper_open_err_m: float = 0.003


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def evaluate_hold(metrics: dict, thresholds: HoldThresholds) -> tuple[bool, list[str]]:
    """Returns (passed, failure reasons).

    Expected metrics keys: all_finite (bool), base_height_end_min/_max (m, over envs),
    tilt_deg_max (deg, from projected gravity z), arm_joint_err_max / leg_joint_err_max (rad),
    settle_drift_m,
    joint_names_ok (bool), eef_present (bool), arm_mass_kg, num_dof (int, optional, must be 20 when present).
    When `gripper_tested` is true, gripper_closed_err_max / gripper_open_err_max (m) are required too.
    """
    t = thresholds
    reasons: list[str] = []

    numeric = (
        "base_height_end_min",
        "base_height_end_max",
        "tilt_deg_max",
        "arm_joint_err_max",
        "leg_joint_err_max",
        "settle_drift_m",
        "arm_mass_kg",
    )
    if not metrics.get("all_finite", False) or not all(_finite(metrics.get(k)) for k in numeric):
        reasons.append("non-finite (NaN/inf) or missing values in simulation state/metrics")
        return False, reasons

    if metrics["base_height_end_min"] < t.min_base_height:
        reasons.append(f"base height too low: min {metrics['base_height_end_min']:.3f} m < {t.min_base_height} m")
    if metrics["base_height_end_max"] > t.max_base_height:
        reasons.append(f"base height too high: max {metrics['base_height_end_max']:.3f} m > {t.max_base_height} m")
    if metrics["tilt_deg_max"] > t.max_tilt_deg:
        reasons.append(f"tilt too large: {metrics['tilt_deg_max']:.1f} deg > {t.max_tilt_deg} deg")
    if metrics["arm_joint_err_max"] > t.max_arm_joint_err_rad:
        reasons.append(
            f"arm joint error too large: {metrics['arm_joint_err_max']:.3f} rad > {t.max_arm_joint_err_rad} rad"
            f" (worst: {metrics.get('arm_worst_joint', '?')})"
        )
    if metrics["leg_joint_err_max"] > t.max_leg_joint_err_rad:
        reasons.append(
            f"leg joint error too large: {metrics['leg_joint_err_max']:.3f} rad > {t.max_leg_joint_err_rad} rad"
        )
    if metrics["settle_drift_m"] > t.max_settle_drift_m:
        reasons.append(f"still settling: drift {metrics['settle_drift_m']:.4f} m > {t.max_settle_drift_m} m")
    if not metrics.get("joint_names_ok", False):
        reasons.append(f"joint-name addressing check failed: {metrics.get('joint_names_detail', 'see report')}")
    if not metrics.get("eef_present", False):
        reasons.append(f"EEF body '{ARX5_EEF_BODY}' missing from robot bodies")
    lo, hi = t.arm_mass_range
    if not (lo <= metrics["arm_mass_kg"] <= hi):
        reasons.append(f"arm mass {metrics['arm_mass_kg']:.3f} kg outside [{lo}, {hi}] kg")

    if "num_dof" in metrics and metrics["num_dof"] != EXPECTED_DOF:
        reasons.append(f"robot has {metrics['num_dof']} DoF, expected {EXPECTED_DOF}")

    if metrics.get("gripper_tested", False):
        g_keys = ("gripper_closed_err_max", "gripper_open_err_max")
        if not all(_finite(metrics.get(k)) for k in g_keys):
            reasons.append("gripper test: non-finite or missing gripper metrics")
        else:
            if metrics["gripper_closed_err_max"] > t.max_gripper_closed_err_m:
                reasons.append(
                    f"gripper did not close: error {metrics['gripper_closed_err_max']:.4f} m > "
                    f"{t.max_gripper_closed_err_m} m (target {GRIPPER_CLOSED_M} m)"
                )
            if metrics["gripper_open_err_max"] > t.max_gripper_open_err_m:
                reasons.append(
                    f"gripper did not open: error {metrics['gripper_open_err_max']:.4f} m > "
                    f"{t.max_gripper_open_err_m} m (target {GRIPPER_OPEN_M} m)"
                )

    return not reasons, reasons


def build_report(
    metrics: dict,
    thresholds: HoldThresholds,
    passed: bool,
    reasons: list[str],
    usd_path: str,
    versions: dict,
    args: dict,
) -> dict:
    return {
        "gate": "W1.5a go2_arx5 spawn+hold+gripper",
        "passed": passed,
        "failure_reasons": reasons,
        "usd_path": usd_path,
        "versions": versions,
        "args": args,
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }


def build_parser(with_app_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W1.5a: spawn GO2-ARX5 and hold the default pose.")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument(
        "--usd", type=str, default=None, help=f"override fused USD (sets {GO2_ARX5_USD_ENV} before cfg import)"
    )
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument("--fix_base", action="store_true", help="fix the Go2 base to the world (GO2_ARX5_FIXED_BASE_CFG)")
    parser.add_argument(
        "--gripper_settle_s",
        type=float,
        default=DEFAULT_GRIPPER_SETTLE_S,
        help="settle time after each gripper command, seconds (default 1.0, provisional)",
    )
    parser.add_argument("--no_gripper_test", action="store_true", help="skip the gripper close/open phase")
    d = HoldThresholds()
    parser.add_argument("--min_base_height", type=float, default=d.min_base_height)
    parser.add_argument("--max_base_height", type=float, default=d.max_base_height)
    parser.add_argument("--max_tilt_deg", type=float, default=d.max_tilt_deg)
    parser.add_argument("--max_arm_joint_err_rad", type=float, default=d.max_arm_joint_err_rad)
    parser.add_argument("--max_leg_joint_err_rad", type=float, default=d.max_leg_joint_err_rad)
    parser.add_argument("--max_settle_drift_m", type=float, default=d.max_settle_drift_m)
    parser.add_argument("--max_gripper_closed_err_m", type=float, default=d.max_gripper_closed_err_m)
    parser.add_argument("--max_gripper_open_err_m", type=float, default=d.max_gripper_open_err_m)
    parser.add_argument(
        "--watchdog_s", type=float, default=420.0, help="dump all thread stacks every N s if hung (0 disables)"
    )
    parser.add_argument("--arm_mass_range", type=float, nargs=2, default=list(d.arm_mass_range), metavar=("LO", "HI"))
    if with_app_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def thresholds_from_args(args: argparse.Namespace) -> HoldThresholds:
    return HoldThresholds(
        min_base_height=args.min_base_height,
        max_base_height=args.max_base_height,
        max_tilt_deg=args.max_tilt_deg,
        max_arm_joint_err_rad=args.max_arm_joint_err_rad,
        max_leg_joint_err_rad=args.max_leg_joint_err_rad,
        max_settle_drift_m=args.max_settle_drift_m,
        max_gripper_closed_err_m=args.max_gripper_closed_err_m,
        max_gripper_open_err_m=args.max_gripper_open_err_m,
        arm_mass_range=(float(args.arm_mass_range[0]), float(args.arm_mass_range[1])),
    )


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for pkg in ("isaaclab", "isaacsim", "torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:  # best effort
            out[pkg] = None
    return out


def main(argv: list[str] | None = None) -> int:
    # parse first (before app launch) so bad args exit 2 quickly; AppLauncher args are
    # added here, which needs isaaclab.app only (safe before app start, as in play.py).
    parser = build_parser(with_app_args=True)
    args = parser.parse_args(argv)  # argparse exits with code 2 on bad args
    if (
        args.num_envs < 1
        or args.duration_s <= 0
        or args.gripper_settle_s <= 0
        or args.arm_mass_range[0] > args.arm_mass_range[1]
    ):
        print("[ERROR] invalid --num_envs/--duration_s/--gripper_settle_s/--arm_mass_range", file=sys.stderr)
        return 2
    args.headless = True
    thresholds = thresholds_from_args(args)

    if args.usd:
        os.environ[GO2_ARX5_USD_ENV] = str(Path(args.usd).resolve())

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    exit_code = 1
    try:
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        partial: dict = {}
        try:
            report = _run(args, thresholds, partial)
        except BaseException as e:  # noqa: BLE001 - includes SystemExit/KeyboardInterrupt
            tb = traceback.format_exc()
            traceback.print_exc()
            report = build_report(
                partial,
                thresholds,
                False,
                [f"exception: {e!r}"],
                usd_path="",
                versions={},
                args={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, type(None)))},
            )
            report["error"] = tb
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = _REPO_ROOT / report_path
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            print(f"[INFO] report: {report_path}", flush=True)
        except Exception:  # noqa: BLE001 - best effort
            traceback.print_exc()
            if report["passed"]:
                report["passed"] = False
        print(f"[INFO] {'PASS' if report['passed'] else 'FAIL'}: {report['failure_reasons']}", flush=True)
        exit_code = 0 if report["passed"] else 1
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        sys.stdout.flush()
        sys.stderr.flush()
        _finish(simulation_app, exit_code)
    return exit_code  # only reached if _finish returns (tests)


def _finish(simulation_app, exit_code: int) -> None:
    """Hard-exit the process; never calls simulation_app.close().

    SimulationApp.close() terminates the process with code 0 (losing the real exit code and any
    traceback) and was observed to hang for >10 min at shutdown on the RunPod RTX 4090 pod
    (driver 595.91, Isaac Sim 5.1) after the work finished and the report was written; everything
    we need is on disk before this point, so we hard-exit.
    """
    del simulation_app  # kept for call-site compatibility
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - nothing sensible to do if flushing fails while exiting
        pass
    os._exit(exit_code)


def _gripper_phase(args: argparse.Namespace, robot, scene, sim) -> dict:
    """Command both fingers closed, settle, record positions vs targets; then open, settle, record.

    Everything else is held at the default pose. Tolerances live in HoldThresholds (PROVISIONAL).
    """
    import torch

    g_idx, _ = robot.find_joints(list(GRIPPER_JOINT_NAMES), preserve_order=True)
    n_steps = max(1, int(args.gripper_settle_s / SIM_DT))

    def command_and_settle(target_m: float) -> tuple[list[float], float, bool]:
        target = as_tensor(robot.data.default_joint_pos).clone()
        target[:, g_idx] = target_m
        for _ in range(n_steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(SIM_DT)
        pos = as_tensor(robot.data.joint_pos)[:, g_idx].clone()  # (envs, fingers)
        err = (pos - target_m).abs()
        finite = bool(torch.isfinite(pos).all())
        return [float(x) for x in pos.mean(dim=0)], float(err.max()), finite

    closed_pos, closed_err, closed_finite = command_and_settle(GRIPPER_CLOSED_M)
    open_pos, open_err, open_finite = command_and_settle(GRIPPER_OPEN_M)
    return {
        "gripper_tested": True,
        "gripper_joint_names": list(GRIPPER_JOINT_NAMES),
        "gripper_settle_s": float(args.gripper_settle_s),
        "gripper_closed_target_m": GRIPPER_CLOSED_M,
        "gripper_open_target_m": GRIPPER_OPEN_M,
        "gripper_closed_pos_mean_m": closed_pos,
        "gripper_open_pos_mean_m": open_pos,
        "gripper_closed_err_max": closed_err if closed_finite else float("nan"),
        "gripper_open_err_max": open_err if open_finite else float("nan"),
    }


def _run(args: argparse.Namespace, thresholds: HoldThresholds, progress: dict | None = None) -> dict:
    """Everything needing Isaac; only called after the app is launched."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from common.cerberus_description.go2_arx5_cfg import (
        GO2_ARX5_CFG,
        GO2_ARX5_FIXED_BASE_CFG,
    )

    robot_cfg = GO2_ARX5_FIXED_BASE_CFG if args.fix_base else GO2_ARX5_CFG

    @configclass
    class HoldSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )
        robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args.device))
    scene = InteractiveScene(HoldSceneCfg(num_envs=args.num_envs, env_spacing=3.0))
    print("[INFO] scene built", flush=True)
    sim.reset()
    print("[INFO] sim.reset() done", flush=True)
    robot = scene["robot"]

    # name-based addressing
    expected = list(GO2_ARX5_EMBODIMENT.joint_names)
    detail = []
    try:
        idx, _ = robot.find_joints(expected, preserve_order=True)
        if len(idx) != len(expected):
            detail.append(f"find_joints returned {len(idx)} != {len(expected)}")
    except Exception as e:  # noqa: BLE001
        detail.append(f"find_joints raised: {e}")
    if set(robot.joint_names) != set(expected):
        detail.append(
            f"missing={sorted(set(expected) - set(robot.joint_names))} "
            f"unexpected={sorted(set(robot.joint_names) - set(expected))}"
        )
    if GO2_ARX5_EMBODIMENT.resolve() is not GO2_ARX5_CFG:
        detail.append("GO2_ARX5_EMBODIMENT.resolve() is not GO2_ARX5_CFG")
    num_dof = len(robot.joint_names)
    if num_dof != EXPECTED_DOF:
        detail.append(f"robot reports {num_dof} DoF, expected {EXPECTED_DOF}")
    eef_present = ARX5_EEF_BODY in robot.body_names

    n_steps = int(args.duration_s / SIM_DT)
    heights, gz, jpos = [], [], []
    default_q = as_tensor(robot.data.default_joint_pos).clone()
    print(f"[INFO] stepping started: {n_steps} steps", flush=True)
    for _ in range(n_steps):
        robot.set_joint_position_target(as_tensor(robot.data.default_joint_pos))
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)
        heights.append((as_tensor(robot.data.root_pos_w)[:, 2] - scene.env_origins[:, 2]).clone())
        gz.append(as_tensor(robot.data.projected_gravity_b).clone())
        jpos.append(as_tensor(robot.data.joint_pos).clone())

    print("[INFO] stepping done", flush=True)
    h = torch.stack(heights)  # (T, N)
    g = torch.stack(gz)
    q = torch.stack(jpos)  # (T, N, J)
    win = min(int(SETTLE_WINDOW_S / SIM_DT), h.shape[0] - 1)
    all_finite = bool(torch.isfinite(h).all() and torch.isfinite(g).all() and torch.isfinite(q).all())

    # gravity z in body frame is -cos(tilt) when upright
    tilt = torch.rad2deg(torch.acos(torch.clamp(-g[-1][:, 2], -1.0, 1.0)))
    # Convention-safe: derived from projected_gravity_b (a vector), never from root_quat_w, so it is
    # identical for Lab 2.x (wxyz) and 3.x (xyzw). as_tensor() unwraps the 3.x ProxyArray.
    # Signed pitch/roll from projected gravity g_b = R^T (0,0,-1) (x fwd, y left, z up):
    #   pitch (about +y, positive = nose DOWN): gx = sin(pitch) -> pitch = asin(gx)
    #   roll  (about +x, positive = left side UP / right side down): gy = -sin(roll) -> roll = asin(-gy)
    # reported as the per-env value with the largest magnitude.
    pitch_e = torch.rad2deg(torch.asin(torch.clamp(g[-1][:, 0], -1.0, 1.0)))
    roll_e = torch.rad2deg(torch.asin(torch.clamp(-g[-1][:, 1], -1.0, 1.0)))
    pitch_deg = float(pitch_e[pitch_e.abs().argmax()])
    roll_deg = float(roll_e[roll_e.abs().argmax()])
    err_j = (q[-1] - default_q).abs().max(dim=0).values  # (J,) max over envs
    dev_j = (q[-1] - default_q).mean(dim=0)  # (J,) signed mean over envs
    jn = list(robot.joint_names)
    err_by_name = {n: float(err_j[i]) for i, n in enumerate(jn)}
    dev_by_name = {n: float(dev_j[i]) for i, n in enumerate(jn)}
    leg_names = [n for n in GO2_ARX5_EMBODIMENT.joint_groups["legs"] if n in err_by_name]
    arm_names = [n for n in GO2_ARX5_EMBODIMENT.joint_groups["arm"] if n in err_by_name]
    leg_errs = [err_by_name[n] for n in leg_names]
    arm_errs = [err_by_name[n] for n in arm_names]
    leg_err_max = max(leg_errs) if leg_errs else float("nan")
    leg_err_mean = sum(leg_errs) / len(leg_errs) if leg_errs else float("nan")
    arm_err_max = max(arm_errs) if arm_errs else float("nan")
    worst_arm = max(arm_names, key=lambda n: err_by_name[n]) if arm_names else ""
    drift = (h[-1] - h[-1 - win]).abs().max()

    gripper = {}
    if not args.no_gripper_test:
        gripper = _gripper_phase(args, robot, scene, sim)
        print(
            f"[INFO] gripper: closed err {gripper['gripper_closed_err_max']:.4f} m, "
            f"open err {gripper['gripper_open_err_max']:.4f} m",
            flush=True,
        )

    # Lab 2.3.2: get_masses() is a (num_envs, num_bodies) CPU torch tensor; Lab 3.0.0 (this pod): root_view is
    # a PhysX ArticulationView and get_masses() returns a Warp array, converted below before indexed torch ops.
    # physx_view() selects Lab 3.0 root_view and falls back to root_physx_view on Lab 2.x.
    masses = as_tensor(physx_view(robot).get_masses())
    if not isinstance(masses, torch.Tensor):
        import warp as wp

        masses = wp.to_torch(masses)
    body_names = list(robot.body_names)
    arm_cols = [body_names.index(b) for b in ARX5_BODY_NAMES if b in body_names]
    arm_mass = float(masses[0, arm_cols].sum())
    total_mass = float(masses[0].sum())

    metrics = {
        "all_finite": all_finite,
        "num_envs": int(args.num_envs),
        "steps": n_steps,
        "base_height_end_min": float(h[-1].min()),
        "base_height_end_max": float(h[-1].max()),
        "base_height_per_env_min": [float(x) for x in h.min(dim=0).values],
        "base_height_per_env_max": [float(x) for x in h.max(dim=0).values],
        "tilt_deg_max": float(tilt.max()),
        "leg_joint_err_max": leg_err_max,
        "leg_joint_err_mean": leg_err_mean,
        "arm_joint_err_max": arm_err_max,
        "arm_worst_joint": worst_arm,
        "joint_err_by_name": err_by_name,
        "joint_dev_by_name": dev_by_name,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "settle_drift_m": float(drift),
        "joint_names_ok": not detail,
        "joint_names_detail": "; ".join(detail),
        "eef_present": bool(eef_present),
        "num_dof": num_dof,
        "fix_base": bool(args.fix_base),
        **gripper,
        "arm_mass_kg": arm_mass,
        "total_mass_kg": total_mass,
        "joint_names": list(robot.joint_names),
        "body_names": body_names,
    }
    print(
        f"[INFO] joint err: legs max {leg_err_max:.3f} rad (mean {leg_err_mean:.3f}), "
        f"arm max {arm_err_max:.3f} rad (worst {worst_arm}); pitch {pitch_deg:+.1f} deg, roll {roll_deg:+.1f} deg",
        flush=True,
    )
    if progress is not None:
        progress.update(metrics)
    print("[INFO] metrics computed", flush=True)
    passed, reasons = evaluate_hold(metrics, thresholds)
    return build_report(
        metrics,
        thresholds,
        passed,
        reasons,
        usd_path=str(fused_usd_path()),
        versions=_versions(),
        args={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, type(None)))},
    )


if __name__ == "__main__":
    sys.exit(main())
