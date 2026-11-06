"""W2a: scripted demo recorder for the fixed-base Go2 + ARX5 stack task (headless pod).

Isaac Lab's `scripts/tools/record_demos.py` needs a keyboard / spacemouse / XR device, which does not exist on a
headless pod. This script builds the env the same way `record_demos.py` does (Isaac Lab `v3.0.0-EA`:
`ActionStateRecorderManagerCfg`, `DatasetExportMode.EXPORT_SUCCEEDED_ONLY`, `terminations.success` replaced by an inert
term and evaluated by hand, `time_out` removed, `observations.policy.concatenate_terms = False`, `env.sim.reset()` +
`recorder_manager.reset()` + `env.reset()` between attempts, success after N consecutive successful steps, then
`record_pre_reset` / `set_success_to_episodes` / `export_episodes`), so the HDF5 output feeds `annotate_demos.py` and
`generate_dataset.py` directly. Instead of a teleop device a waypoint state machine (`waypoints.py`) drives the env:
above cube_2 -> descend -> close (pause) -> lift -> above cube_1 -> descend -> open (pause) -> retreat. Cube poses are
randomized by the env's reset event, so every episode differs.

    $ISAACLAB_PATH/isaaclab.sh -p scripts/record_scripted_demos_go2_arx5.py --num_demos 10 --headless \\
        --output data/demos/go2_arx5_stack_scripted.hdf5 --seed 1 --report data/reports/go2_arx5_scripted_demos.json

Then (repo root on PYTHONPATH so the task ids can be registered):

    PYTHONPATH=$PWD $ISAACLAB_PATH/isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \\
        --task Cerberus-Go2Arx5-Stack-Cube-IK-Rel-Mimic-v0 --auto --input_file <demos.hdf5> \\
        --output_file <annotated.hdf5> \\
        --external_callback policy.manipulation.go2_arx5_stack.register.register_all --headless

Everything at module level is pure python (import-safe without Isaac); isaaclab is imported only inside main()/_run(),
after the app launched. Writes the JSON report first, then hard-exits with the real code (no SimulationApp.close(),
see docs/runpod.md).

Exit codes: 0 recorded --num_demos successful demos, 1 fewer / exception, 2 bad args.

UNVERIFIED (never run; nothing here could be executed without a GPU pod): the Isaac Lab 3.0 calls copied from
`record_demos.py` (`parse_env_cfg(task, device=, num_envs=)`, `terminations.success.replace(...)`,
`recorder_manager.{reset,record_pre_reset,set_success_to_episodes,export_episodes,close}`,
`exported_successful_episode_count`), that `env.sim.reset()` between attempts is clean for the fixed-base fused robot,
the waypoint tolerances / `--grasp_z_offset` against the real finger geometry, and the HDF5 read-back in
`summarize_hdf5` (h5py was not available locally).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from policy.manipulation.go2_arx5_stack import layout
from policy.manipulation.go2_arx5_stack.register import TASK_ID_IK_REL
from policy.manipulation.go2_arx5_stack.waypoints import (
    PickPlaceConfig,
    ScriptedPickPlace,
)

DEFAULT_OUTPUT = "data/demos/go2_arx5_stack_scripted.hdf5"
DEFAULT_REPORT = "data/reports/go2_arx5_scripted_demos.json"


def build_parser(with_app_args: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="W2a: record scripted pick-and-place demos for Isaac Lab Mimic."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_ID_IK_REL,
        help="registered task id (default: %(default)s)",
    )
    parser.add_argument(
        "--num_demos", type=int, default=10, help="successful demos to record"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="HDF5 dataset file (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace --output if it exists"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=10,
        help="consecutive steps of task success that conclude a demo (as record_demos.py)",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=0,
        help="stop after this many episodes (default 0: 3 x num_demos + 5)",
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=600, help="per-episode step cap"
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=30,
        help="extra steps to wait for success after the script has finished",
    )
    parser.add_argument(
        "--grasp_pitch_deg",
        type=float,
        default=math.degrees(layout.GRASP_PITCH_RAD),
        help="rotate the stowed TCP frame about the world y axis by this angle (90 = gripper pointing down)",
    )
    parser.add_argument(
        "--hover_height",
        type=float,
        default=layout.HOVER_HEIGHT,
        help="approach/transport height (m)",
    )
    parser.add_argument(
        "--grasp_z_offset",
        type=float,
        default=0.0,
        help="TCP height above the cube centre at grasp (m); tune on pod",
    )
    parser.add_argument(
        "--close_pause_steps", type=int, default=PickPlaceConfig.close_pause_steps
    )
    parser.add_argument(
        "--open_pause_steps", type=int, default=PickPlaceConfig.open_pause_steps
    )
    parser.add_argument(
        "--max_phase_steps", type=int, default=PickPlaceConfig.max_phase_steps
    )
    parser.add_argument(
        "--watchdog_s",
        type=float,
        default=900.0,
        help="dump all thread stacks every N s if hung (0 disables)",
    )
    if with_app_args:
        # AppLauncher args (--device, ...) plus a --headless flag even where Isaac Lab 3.0 dropped it.
        from common.cerberus_compat.launcher import add_launcher_args

        add_launcher_args(parser)
    else:
        parser.add_argument("--headless", action="store_true", default=False)
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    problems = []
    if args.num_demos < 1:
        problems.append("--num_demos must be >= 1")
    if args.num_success_steps < 1:
        problems.append("--num_success_steps must be >= 1")
    if args.max_attempts < 0:
        problems.append("--max_attempts must be >= 0")
    if args.max_episode_steps < 1 or args.settle_steps < 0:
        problems.append("--max_episode_steps must be >= 1 and --settle_steps >= 0")
    if args.hover_height <= 0:
        problems.append("--hover_height must be > 0")
    if (
        args.close_pause_steps < 1
        or args.open_pause_steps < 1
        or args.max_phase_steps < 1
    ):
        problems.append(
            "--close_pause_steps/--open_pause_steps/--max_phase_steps must be >= 1"
        )
    if not str(args.output).endswith((".hdf5", ".h5")):
        problems.append("--output must end in .hdf5")
    return problems


def attempt_budget(args: argparse.Namespace) -> int:
    return args.max_attempts if args.max_attempts > 0 else 3 * args.num_demos + 5


def pick_place_config(args: argparse.Namespace) -> PickPlaceConfig:
    return PickPlaceConfig(
        cube_size=layout.CUBE_SIZE,
        hover_height=args.hover_height,
        grasp_z_offset=args.grasp_z_offset,
        ik_scale=layout.IK_ACTION_SCALE,
        close_pause_steps=args.close_pause_steps,
        open_pause_steps=args.open_pause_steps,
        max_phase_steps=args.max_phase_steps,
    )


def evaluate_run(
    requested: int, recorded: int, hdf5: dict | None, error: str | None
) -> tuple[bool, list[str]]:
    """Pass/fail of a whole run: all demos recorded, and the file on disk agrees. Returns (passed, reasons)."""
    reasons = []
    if error:
        reasons.append(f"exception: {error}")
    if recorded < requested:
        reasons.append(f"recorded {recorded} of {requested} requested successful demos")
    if recorded > 0:
        if not hdf5 or hdf5.get("error"):
            reasons.append(
                f"could not read the output file back: {(hdf5 or {}).get('error', 'missing summary')}"
            )
        elif hdf5.get("num_demos") != recorded:
            reasons.append(
                f"output file has {hdf5.get('num_demos')} demos but {recorded} were exported"
            )
        elif not all(hdf5.get("success", [])):
            reasons.append("output file contains demos not flagged successful")
    return not reasons, reasons


def build_report(
    args: dict,
    attempts: list[dict],
    recorded: int,
    hdf5: dict | None,
    versions: dict,
    error: str | None = None,
) -> dict:
    passed, reasons = evaluate_run(args.get("num_demos", 0), recorded, hdf5, error)
    statuses: dict[str, int] = {}
    for a in attempts:
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    report = {
        "gate": "W2a scripted demo recording go2_arx5 stack",
        "passed": passed,
        "failure_reasons": reasons,
        "num_demos_requested": args.get("num_demos"),
        "num_demos_recorded": recorded,
        "attempt_status_counts": statuses,
        "attempts": attempts,
        "output": args.get("output"),
        "hdf5": hdf5,
        "versions": versions,
        "args": args,
    }
    if error:
        report["error"] = error
    return report


def summarize_hdf5(path: str | Path) -> dict:
    """Read the dataset back (h5py imported lazily): demo count, per-demo samples/success flags, env name."""
    try:
        import h5py

        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = sorted(
                data.keys(),
                key=lambda k: (
                    int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k
                ),
            )
            env_args = data.attrs.get("env_args")
            return {
                "path": str(path),
                "num_demos": len(keys),
                "demo_keys": keys,
                "num_samples": [int(data[k].attrs.get("num_samples", 0)) for k in keys],
                "success": [bool(data[k].attrs.get("success", False)) for k in keys],
                "env_args": json.loads(env_args) if env_args else None,
            }
    except Exception as e:  # noqa: BLE001 - report, never raise
        return {"path": str(path), "error": repr(e)}


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for pkg in (
        "isaaclab",
        "isaaclab_mimic",
        "isaaclab_tasks",
        "isaacsim",
        "torch",
        "h5py",
    ):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:  # best effort
            out[pkg] = None
    return out


def _jsonable_args(args: argparse.Namespace) -> dict:
    return {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (int, float, str, bool, list, type(None)))
    }


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def main(argv: list[str] | None = None) -> int:
    # Parse first (before the app launches) so bad args exit 2 quickly; AppLauncher args are added here, which only
    # needs isaaclab.app (safe before app start).
    parser = build_parser(with_app_args=True)
    args = parser.parse_args(argv)  # argparse exits with code 2 on bad args
    problems = validate_args(args)
    if problems:
        print("[ERROR] " + "; ".join(problems), file=sys.stderr)
        return 2
    output = _resolve(args.output)
    if output.exists() and not args.overwrite:
        print(
            f"[ERROR] {output} exists; pass --overwrite to replace it", file=sys.stderr
        )
        return 2
    args.headless = True  # pods have no display

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    exit_code = 1
    try:
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        progress: dict = {"attempts": [], "recorded": 0}
        error = None
        try:
            _run(args, output, progress)
        except BaseException as e:  # noqa: BLE001 - includes SystemExit/KeyboardInterrupt
            error = f"{e!r}\n{traceback.format_exc()}"
            traceback.print_exc()
        hdf5 = (
            summarize_hdf5(output)
            if progress["recorded"] > 0 and output.exists()
            else None
        )
        report = build_report(
            _jsonable_args(args),
            progress["attempts"],
            progress["recorded"],
            hdf5,
            progress.get("versions", {}),
            error,
        )
        report_path = _resolve(args.report)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            print(f"[INFO] report: {report_path}", flush=True)
        except Exception:  # noqa: BLE001 - best effort
            traceback.print_exc()
            report["passed"] = False
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
        sys.stdout.flush()
        sys.stderr.flush()
        _finish(simulation_app, exit_code)
    return exit_code  # only reached if _finish returns (tests)


def _finish(simulation_app, exit_code: int) -> None:
    """Hard-exit; never calls simulation_app.close() (it exits 0 and has hung for >10 min, docs/runpod.md)."""
    del simulation_app  # kept for call-site compatibility
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001, S110 - nothing sensible to do if flushing fails while exiting
        pass
    os._exit(exit_code)


def _never_terminate(env):
    """Inert replacement for the success termination (as record_demos.py): success is evaluated by hand."""
    import torch

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _run(args: argparse.Namespace, output: Path, progress: dict) -> None:
    """Everything needing Isaac; only called after the app is launched. Fills `progress` as it goes."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from policy.manipulation.go2_arx5_stack.register import register_all

    progress["versions"] = _versions()
    random.seed(args.seed)  # the cube-pose reset event samples with python's `random`
    torch.manual_seed(args.seed)
    register_all()

    output.parent.mkdir(parents=True, exist_ok=True)
    # UNVERIFIED: every Isaac Lab call below mirrors record_demos.py at v3.0.0-EA but has not run for this task.
    # --- env cfg, as record_demos.py's create_environment_config ---------------------------------------------------
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.env_name = args.task.split(":")[-1]
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise RuntimeError(
            f"task {args.task} has no terminations.success; cannot label demos as successful"
        )
    env_cfg.terminations.success = success_term.replace(
        func=_never_terminate, params={}
    )
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output.parent)
    env_cfg.recorders.dataset_filename = output.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print("[INFO] env created", flush=True)

    machine = ScriptedPickPlace(pick_place_config(args))
    pitch = torch.tensor(
        [math.radians(args.grasp_pitch_deg)], dtype=torch.float32, device=env.device
    )
    pitch_axis = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=env.device)
    q_pitch = math_utils.quat_from_angle_axis(
        pitch, pitch_axis
    )  # rotation about the world y axis

    def reset_env():
        # as record_demos.py handle_reset with --reset_sim_buffer_each_episode (its default)
        env.sim.reset()
        env.recorder_manager.reset()
        obs, _ = env.reset()
        return obs

    def run_attempt(index: int) -> dict:
        obs = reset_env()
        pol = obs["policy"]
        cubes = pol["cube_positions"][0].tolist()
        cube_1, cube_2 = tuple(cubes[0:3]), tuple(cubes[3:6])
        machine.reset(pick_pos=cube_2, place_pos=cube_1)
        # Desired TCP orientation: the post-reset (stowed) orientation rotated by the grasp pitch about world y.
        q_des = math_utils.quat_mul(q_pitch, pol["eef_quat"])
        record = {
            "index": index,
            "cube_1": list(cube_1),
            "cube_2": list(cube_2),
            "status": "max_episode_steps",
        }
        success_steps = 0
        settle = 0
        step = 0
        for step in range(1, args.max_episode_steps + 1):
            pol = obs["policy"]
            eef = tuple(pol["eef_pos"][0].tolist())
            rot_err = math_utils.axis_angle_from_quat(
                math_utils.quat_mul(q_des, math_utils.quat_inv(pol["eef_quat"]))
            )
            act = machine.step(eef, tuple(rot_err[0].tolist()))
            if machine.failed:
                record["status"] = "script_failed"
                record["failure_reason"] = machine.failure_reason
                break
            action = torch.tensor(
                [act.as_list()], dtype=torch.float32, device=env.device
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                # e.g. a cube fell off the table; the env already reset itself
                record["status"] = "env_terminated"
                break
            if bool(success_term.func(env, **success_term.params)[0]):
                success_steps += 1
            else:
                success_steps = 0
            if success_steps >= args.num_success_steps:
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                record["status"] = "success"
                break
            if machine.done:
                settle += 1
                if settle > args.settle_steps:
                    record["status"] = "done_without_success"
                    break
        record["steps"] = step
        record["phase"] = machine.phase.value
        return record

    budget = attempt_budget(args)
    with torch.inference_mode():
        while (
            progress["recorded"] < args.num_demos and len(progress["attempts"]) < budget
        ):
            record = run_attempt(len(progress["attempts"]))
            progress["attempts"].append(record)
            progress["recorded"] = int(
                env.recorder_manager.exported_successful_episode_count
            )
            print(
                f"[INFO] attempt {record['index']}: {record['status']} in {record['steps']} steps "
                f"(phase {record['phase']}); recorded {progress['recorded']}/{args.num_demos}",
                flush=True,
            )

    # UNVERIFIED: recorder_manager.close() flushes and closes the stream without needing env.close().
    # Close the HDF5 stream (not the env / app) so the file can be read back and is complete on disk.
    try:
        env.recorder_manager.close()
    except Exception:  # noqa: BLE001 - the read-back in the report will say if the file is unusable
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
