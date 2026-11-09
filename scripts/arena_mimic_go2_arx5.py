"""W2b pod script: fixed-base Go2 + ARX5 pick-and-place Mimic flow in a Lightwheel RoboCasa kitchen, composed by Arena.

Subcommands (all take --num_envs --report --headless --watchdog_s --dry-run, plus the kitchen environment flags):

  smoke     build the env (Lightwheel kitchen + go2_arx5 embodiment + pick-and-place task), reset, step zero actions,
            report observation/action structure, fixed-base check, arm hold drift and the Mimic API calls.
  record    scripted demos (pure-python waypoint state machine, policy/manipulation/go2_arx5_stack/waypoints.py) recorded
            through Isaac Lab's ActionStateRecorderManagerCfg to HDF5, the same approach as
            scripts/record_scripted_demos_go2_arx5.py but on the Arena-composed (mimic) env.
  annotate  add datagen_info + subtask_term_signals (auto mode, needs the `grasp_1` observation of the embodiment).
  generate  Isaac Lab Mimic data generation from the annotated file.

Which entry points are correct (read from third_party/IsaacLab-Arena at 311d0f0fc / Lab bb0c8e1b9, see
common/cerberus_worlds/arena_assets/go2_arx5_embodiment.py for the full list):
  * Arena has NO annotate/generate script of its own. Its docs run Isaac Lab's
    submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/{annotate_demos,generate_dataset}.py with
    `--task <arena env name> --mimic --external_callback
    isaaclab_arena.environments.isaaclab_interop.environment_registration_callback` (+ `--external_environment_class_path
    module:Class` for an out-of-tree environment). `generate` uses exactly that, as a subprocess, so that a hang in
    SimulationApp.close() (docs/runpod.md) cannot swallow the report.
  * `annotate` runs an in-process port of annotate_demos.py auto mode by default (`--engine inprocess`): the Lab
    script reads `args_cli.headless`, which the pinned AppLauncher does not register any more (AttributeError before it
    does anything; fixing it would mean patching Lab). `--engine lab` runs the Lab script anyway, for a Lab commit where
    it works.
  * Recording: Arena's docs use Lab's record_demos.py (teleop devices only). On a headless pod we record in-process.

The kitchen environment (riser block, plate, Lightwheel-SDK pick object) is Go2Arx5KitchenPickAndPlaceEnvironment in
common/cerberus_worlds/arena_assets/go2_arx5_embodiment.py; ALL stages build it from the same options, so pass the same
kitchen flags to record, annotate and generate. Scenes and the embodiment come from registry.get_backend('arena').

Pod usage (repo root on PYTHONPATH, Arena installed, Lightwheel SDK reachable):

    python scripts/arena_mimic_go2_arx5.py smoke --headless
    python scripts/arena_mimic_go2_arx5.py record --num_demos 10 --headless
    python scripts/arena_mimic_go2_arx5.py annotate --headless
    python scripts/arena_mimic_go2_arx5.py generate --generation_num_trials 20 --num_envs 4

Every subcommand writes its JSON report FIRST, then flushes and hard-exits with the real code (no
SimulationApp.close(), see docs/runpod.md), with a faulthandler watchdog. `--dry-run` prints the resolved plan and
touches no Isaac. Exit codes: 0 pass, 1 fail, 2 bad args.

Verified on a pod (Sim 6.2 / Lab v3.0.0 GA, 2026-10-25): `smoke` passed in a Lightwheel kitchen
(data/reports/ga_upgrade/mimic_smoke.json). NOT verified on GA: `annotate` / `generate` -- Lab Mimic's own
`env.reset()` hits the upstream Arena `slice(None)` bug (docs/arena_integration_log.md, "Lab 3.0 GA upgrade", item 5).
The in-process annotate/record calls mirror Lab's annotate_demos.py / record_demos.py at the pinned commit; the
module docstring of go2_arx5_embodiment.py lists the remaining assumptions.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from record_scripted_demos_go2_arx5 import summarize_hdf5

from common.cerberus_compat.reset import reset_all
from common.cerberus_description.go2_arx5_spec import GO2_ARX5_EMBODIMENT
from common.cerberus_worlds import registry
from common.cerberus_worlds.arena_assets import go2_arx5_embodiment as emb
from policy.manipulation.go2_arx5_stack import layout
from policy.manipulation.go2_arx5_stack.waypoints import (
    PickPlaceConfig,
    ScriptedPickPlace,
)

COMMANDS = ("smoke", "record", "annotate", "generate")
BACKEND = "arena"
INTEROP_CALLBACK = "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback"
LAB_MIMIC_SCRIPTS_REL = "submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic"
DEFAULT_ARENA_ROOT = "third_party/IsaacLab-Arena"
DEFAULT_RECORDED = "data/demos/arena_go2_arx5_kitchen_scripted.hdf5"
DEFAULT_ANNOTATED = "data/demos/arena_go2_arx5_kitchen_annotated.hdf5"
DEFAULT_GENERATED = "data/demos/arena_go2_arx5_kitchen_generated.hdf5"
DEFAULT_REPORTS = {c: f"data/reports/arena_mimic_go2_arx5_{c}.json" for c in COMMANDS}
DEFAULT_NUM_ENVS = {"smoke": 1, "record": 1, "annotate": 1, "generate": 4}
SMOKE_BASE_Z_TOL_M = 0.01  # fixed base: the root must not move
SMOKE_ARM_DRIFT_TOL_RAD = 0.05  # zero relative-IK command must hold the stowed arm
SMOKE_OBJECT_Z_BAND_M = (-0.02, 0.15)  # pick/plate z relative to the support top after settling


# ---------------------------------------------------------------------------------------------------------
# Parsing and planning (pure python)
# ---------------------------------------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser, command: str) -> None:
    p.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS[command])
    p.add_argument("--report", type=str, default=DEFAULT_REPORTS[command])
    p.add_argument("--headless", action="store_true", default=False, help="no Kit viewport (pods have no display)")
    p.add_argument(
        "--watchdog_s", type=float, default=900.0, help="dump all thread stacks every N s if hung (0 disables)"
    )
    p.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="print the resolved plan and exit; imports no Isaac (exit 0 if the plan has no problems)",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    emb.add_environment_cli_args(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="build, reset, step, report")
    _add_common(p, "smoke")
    p.add_argument("--steps", type=int, default=30, help="zero-action steps after the reset")
    p.add_argument(
        "--env_kind",
        choices=("mimic", "rl"),
        default="mimic",
        help="mimic: ArenaEnvBuilderCfg(mimic=True) (ManagerBasedRLMimicEnv, what Mimic uses); rl: the plain Arena env",
    )

    p = sub.add_parser("record", help="scripted demos -> HDF5 (in-process, mimic env)")
    _add_common(p, "record")
    p.add_argument("--num_demos", type=int, default=10, help="successful demos to record")
    p.add_argument("--output", type=str, default=DEFAULT_RECORDED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num_success_steps", type=int, default=10, help="consecutive success steps that conclude a demo")
    p.add_argument("--max_attempts", type=int, default=0, help="stop after this many episodes (0: 3 x num_demos + 5)")
    p.add_argument("--max_episode_steps", type=int, default=600)
    p.add_argument("--pre_settle_steps", type=int, default=10, help="zero-action steps after reset before planning")
    p.add_argument("--settle_steps", type=int, default=30, help="extra steps to wait for success after the script ends")
    p.add_argument("--grasp_pitch_deg", type=float, default=math.degrees(layout.GRASP_PITCH_RAD))
    p.add_argument("--grasp_yaw_deg", type=float, default=90.0, help="offset from object yaw for the world-Z grasp orientation")
    p.add_argument("--hover_height", type=float, default=layout.HOVER_HEIGHT)
    p.add_argument("--grasp_z_offset", type=float, default=0.0, help="TCP height above the object root at grasp (m)")
    p.add_argument("--pick_center_dz", type=float, default=0.0, help="object root -> grasp centre z offset (m); tune")
    p.add_argument("--close_pause_steps", type=int, default=PickPlaceConfig.close_pause_steps)
    p.add_argument("--open_pause_steps", type=int, default=PickPlaceConfig.open_pause_steps)
    p.add_argument("--max_phase_steps", type=int, default=PickPlaceConfig.max_phase_steps)

    p = sub.add_parser("annotate", help="add datagen_info and subtask signals to recorded demos")
    _add_common(p, "annotate")
    p.add_argument("--input_file", type=str, default=DEFAULT_RECORDED)
    p.add_argument("--output_file", type=str, default=DEFAULT_ANNOTATED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--engine", choices=("inprocess", "lab"), default="inprocess")
    _add_lab_args(p)

    p = sub.add_parser("generate", help="Isaac Lab Mimic data generation (Lab generate_dataset.py via Arena interop)")
    _add_common(p, "generate")
    p.add_argument("--input_file", type=str, default=DEFAULT_ANNOTATED)
    p.add_argument("--output_file", type=str, default=DEFAULT_GENERATED)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--generation_num_trials", type=int, default=10)
    p.add_argument("--min_generated", type=int, default=1, help="demos the output must contain to pass")
    p.add_argument(
        "--accept_timeout",
        action="store_true",
        help="pass when the Lab process had to be killed after --timeout_s but the output has >= --min_generated demos",
    )
    _add_lab_args(p)
    return parser


def _add_lab_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arena_root", type=str, default=DEFAULT_ARENA_ROOT, help="Arena checkout with its Lab submodule")
    p.add_argument("--lab_python", type=str, default=sys.executable, help="interpreter for the Lab scripts")
    p.add_argument("--timeout_s", type=float, default=3600.0, help="kill the Lab subprocess after this long")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def validate_args(args: argparse.Namespace) -> list[str]:
    """Hard argument problems (exit 2)."""
    problems = []
    if args.num_envs < 1:
        problems.append("--num_envs must be >= 1")
    cmd = args.command
    if cmd in ("record", "annotate") and args.num_envs != 1:
        problems.append(f"{cmd} runs a single env (--num_envs 1); got {args.num_envs}")
    if cmd == "smoke" and args.steps < 1:
        problems.append("--steps must be >= 1")
    if cmd == "record":
        if args.num_demos < 1:
            problems.append("--num_demos must be >= 1")
        if args.num_success_steps < 1 or args.max_attempts < 0 or args.pre_settle_steps < 0 or args.settle_steps < 0:
            problems.append("--num_success_steps >= 1, --max_attempts/--pre_settle_steps/--settle_steps >= 0")
        if not math.isfinite(args.grasp_yaw_deg):
            problems.append("--grasp_yaw_deg must be finite")
        if args.max_episode_steps < 1 or args.hover_height <= 0:
            problems.append("--max_episode_steps >= 1 and --hover_height > 0")
        if min(args.close_pause_steps, args.open_pause_steps, args.max_phase_steps) < 1:
            problems.append("--close_pause_steps/--open_pause_steps/--max_phase_steps must be >= 1")
    for attr in ("output", "input_file", "output_file"):
        value = getattr(args, attr, None)
        if value is not None and not str(value).endswith((".hdf5", ".h5")):
            problems.append(f"--{attr} must end in .hdf5")
    if cmd == "generate":
        if args.generation_num_trials < 1 or args.min_generated < 1:
            problems.append("--generation_num_trials and --min_generated must be >= 1")
        if str(args.seed) == str(args.generation_num_trials):
            # Lab's unrecognised-argument check intersects the *tokens* both parsers left over (list_intersection),
            # so equal values of two flags that only one parser knows are reported as unrecognised arguments.
            problems.append("--seed and --generation_num_trials must differ (Lab token-set argument check)")
    if getattr(args, "timeout_s", 1.0) <= 0:
        problems.append("--timeout_s must be > 0")
    return problems


def attempt_budget(args: argparse.Namespace) -> int:
    return args.max_attempts if args.max_attempts > 0 else 3 * args.num_demos + 5


def pick_place_config(args: argparse.Namespace) -> PickPlaceConfig:
    opts = emb.options_from_namespace(args)
    return PickPlaceConfig(
        cube_size=opts.object_height_m,
        hover_height=args.hover_height,
        grasp_z_offset=args.grasp_z_offset,
        ik_scale=layout.IK_ACTION_SCALE,
        close_pause_steps=args.close_pause_steps,
        open_pause_steps=args.open_pause_steps,
        max_phase_steps=args.max_phase_steps,
    )


def env_cli_args(opts: emb.KitchenOptions) -> list[str]:
    """The kitchen flags as an argv fragment (passed to Lab scripts, consumed by the Arena interop callback)."""
    return [
        "--kitchen_scene",
        opts.kitchen_scene,
        "--pick_object",
        opts.pick_object,
        "--robot_xy",
        *[repr(float(v)) for v in opts.robot_xy],
        "--riser_height_m",
        repr(float(opts.riser_height_m)),
        "--object_height_m",
        repr(float(opts.object_height_m)),
        "--plate_size_m",
        *[repr(float(v)) for v in opts.plate_size_m],
        "--plate_thickness_m",
        repr(float(opts.plate_thickness_m)),
        "--episode_length_s",
        repr(float(opts.episode_length_s)),
    ]


def lab_script_path(arena_root: str | Path, name: str) -> Path:
    return _resolve(arena_root) / LAB_MIMIC_SCRIPTS_REL / name


def lab_command(args: argparse.Namespace, stage: str) -> list[str]:
    """Isaac Lab Mimic script command line through Arena's interop callback (`stage`: annotate or generate)."""
    opts = emb.options_from_namespace(args)
    script = {"annotate": "annotate_demos.py", "generate": "generate_dataset.py"}[stage]
    cmd = [
        args.lab_python,
        str(lab_script_path(args.arena_root, script)),
        "--task",
        emb.ENV_NAME,
        "--mimic",
        "--external_callback",
        INTEROP_CALLBACK,
        "--external_environment_class_path",
        emb.ENV_FACTORY_PATH,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--num_envs",
        str(args.num_envs),
        "--no_solve_relations",  # the interop callback would otherwise solve relations (default True); we place by pose
        "--input_file",
        str(_resolve(args.input_file)),
        "--output_file",
        str(_resolve(args.output_file)),
    ]
    if stage == "annotate":
        cmd.append("--auto")
    else:
        cmd += ["--generation_num_trials", str(args.generation_num_trials)]
    return cmd + env_cli_args(opts)


def lab_env(headless: bool = True) -> dict[str, str]:
    """Environment for the Lab subprocess: repo root importable (our env factory) and Kit headless."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    if headless:
        env["HEADLESS"] = "1"
    return env


def resolve_plan(args: argparse.Namespace) -> dict:
    """Resolve scene, embodiment, geometry and commands. Pure python, never imports Isaac. Raises ValueError (exit 2)."""
    backend = registry.get_backend(BACKEND)
    opts = emb.options_from_namespace(args)
    scene_id = backend.normalize_scene_ref(opts.kitchen_scene)  # ValueError on a bad ref
    info = backend.describe_scene(scene_id)
    problems = list(emb.validate_options(opts))
    warnings = list(emb.option_warnings(opts))
    if info.get("kind") != "lightwheel":
        problems.append(f"the kitchen environment needs a Lightwheel kitchen, got {info.get('kind')!r} ({scene_id})")
    plan: dict = {
        "command": args.command,
        "backend": BACKEND,
        "backend_class": type(backend).__name__,
        "scene_id": scene_id,
        "scene": info,
        "embodiment": {
            "name": GO2_ARX5_EMBODIMENT.name,
            "arena_asset": emb.GO2_ARX5_ARENA_NAME,
            "joint_groups": {k: list(v) for k, v in GO2_ARX5_EMBODIMENT.joint_groups.items()},
            "fixed_base": True,
            "action_dim": emb.ACTION_DIM,
            "arm_action": emb.ARM_ACTION_SPEC,
            "gripper_action": emb.GRIPPER_ACTION_SPEC,
            "leg_action_term": None,
            "observation_terms": list(emb.OBSERVATION_TERMS),
            "subtask_terms": list(emb.SUBTASK_TERM_NAMES),
            "ee_frame": {"name": emb.EE_FRAME_NAME, "source": emb.ROBOT_BASE_PRIM_PATH, "target": emb.EE_TARGET_PRIM_PATH},
            "control": {"sim_dt": emb.SIM_DT, "decimation": emb.DECIMATION},
        },
        "environment": {
            "name": emb.ENV_NAME,
            "factory": emb.ENV_FACTORY_PATH,
            "options": opts.to_dict(),
            "geometry": emb.kitchen_geometry(opts),
        },
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "headless": args.headless,
        "report": args.report,
        "watchdog_s": args.watchdog_s,
        "problems": problems,
        "warnings": warnings,
    }
    if args.command == "smoke":
        plan["steps"] = args.steps
        plan["env_kind"] = args.env_kind
    elif args.command == "record":
        plan["output"] = args.output
        plan["attempt_budget"] = attempt_budget(args)
        plan["num_demos"] = args.num_demos
        plan["waypoints"] = asdict(pick_place_config(args))
        plan["grasp_pitch_deg"] = args.grasp_pitch_deg
        plan["grasp_yaw_offset_deg"] = args.grasp_yaw_deg
        plan["grasp_yaw_mode"] = "object_yaw_plus_offset"
        if _resolve(args.output).exists() and not args.overwrite:
            problems.append(f"{args.output} exists; pass --overwrite to replace it")
    elif args.command in ("annotate", "generate"):
        plan["input_file"] = args.input_file
        plan["output_file"] = args.output_file
        plan["engine"] = getattr(args, "engine", "lab")
        if not _resolve(args.input_file).exists():
            warnings.append(f"input file {args.input_file} does not exist yet (record/annotate first)")
        if _resolve(args.output_file).exists() and not args.overwrite:
            problems.append(f"{args.output_file} exists; pass --overwrite to replace it")
        if plan["engine"] == "lab" or args.command == "generate":
            script = "annotate_demos.py" if args.command == "annotate" else "generate_dataset.py"
            path = lab_script_path(args.arena_root, script)
            plan["lab_command"] = lab_command(args, args.command)
            plan["lab_command_line"] = shlex.join(plan["lab_command"])
            if not path.is_file():
                warnings.append(f"Lab script not found at {path} (set --arena_root)")
            if args.command == "annotate":
                warnings.append(
                    "Lab annotate_demos.py reads args_cli.headless, which the pinned AppLauncher no longer registers; "
                    "expect AttributeError unless the Lab commit differs (default --engine inprocess avoids it)"
                )
    return plan


def _yaw_from_quaternion(math_utils, quaternion):
    """Extract world yaw through Lab's XYZW-aware Euler conversion API."""
    from common.cerberus_compat.isaac_data import as_tensor

    return math_utils.euler_xyz_from_quat(as_tensor(quaternion))[2]


def _compose_grasp_orientation(math_utils, q_yaw, q_pitch, eef_quat):
    """Apply world-Z yaw after world-Y pitch and before the observed end-effector orientation."""
    return math_utils.quat_mul(math_utils.quat_mul(q_yaw, q_pitch), eef_quat)


# ---------------------------------------------------------------------------------------------------------
# Pass/fail evaluation (pure python)
# ---------------------------------------------------------------------------------------------------------
def evaluate_smoke(m: dict) -> list[str]:
    """Failure reasons of a smoke run from its measured metrics (empty list = pass)."""
    reasons = []
    if m.get("action_dim") != emb.ACTION_DIM:
        reasons.append(f"action dim {m.get('action_dim')} != {emb.ACTION_DIM}")
    if list(m.get("action_terms", [])) != [emb.ARM_ACTION_NAME, emb.GRIPPER_ACTION_NAME]:
        reasons.append(f"action terms {m.get('action_terms')} != arm_action, gripper_action (no leg term)")
    missing = [k for k in emb.OBSERVATION_TERMS if k not in m.get("policy_obs_keys", [])]
    if missing:
        reasons.append(f"policy observation lacks {missing}")
    missing = [k for k in emb.SUBTASK_TERM_NAMES if k not in m.get("subtask_obs_keys", [])]
    if missing:
        reasons.append(f"subtask_terms observation lacks {missing}")
    for key in ("robot", emb.EE_FRAME_NAME, m.get("pick_name"), emb.PLATE_NAME):
        if key not in m.get("scene_keys", []):
            reasons.append(f"scene has no entity {key!r}")
    if m.get("is_fixed_base") is not True:
        reasons.append(f"robot is_fixed_base is {m.get('is_fixed_base')}, expected True")
    if not m.get("finite", False):
        reasons.append("non-finite values in observations/state")
    if m.get("base_z_drift", math.inf) > SMOKE_BASE_Z_TOL_M:
        reasons.append(f"base z drifted {m.get('base_z_drift')} m on a fixed base")
    if m.get("arm_drift_rad", math.inf) > SMOKE_ARM_DRIFT_TOL_RAD:
        reasons.append(f"arm joints drifted {m.get('arm_drift_rad')} rad under a zero command")
    lo, hi = SMOKE_OBJECT_Z_BAND_M
    for name in ("pick", "plate"):
        dz = m.get(f"{name}_dz_to_support")
        if dz is None or not lo <= dz <= hi:
            reasons.append(f"{name} z relative to the support top is {dz}, expected within [{lo}, {hi}] m")
    if m.get("env_kind") == "mimic":
        if m.get("mimic_eef_pose_shape") != [m.get("num_envs"), 4, 4]:
            reasons.append(f"get_robot_eef_pose shape {m.get('mimic_eef_pose_shape')}")
        if emb.PLATE_NAME not in m.get("mimic_object_pose_keys", []) or m.get("pick_name") not in m.get(
            "mimic_object_pose_keys", []
        ):
            reasons.append(f"get_object_poses keys {m.get('mimic_object_pose_keys')} lack the pick object / plate")
        if not m.get("mimic_signal_keys"):
            reasons.append("get_subtask_term_signals returned nothing")
    return reasons


def evaluate_record(requested: int, recorded: int, hdf5: dict | None, error: str | None) -> list[str]:
    reasons = []
    if error:
        reasons.append(f"exception: {error}")
    if recorded < requested:
        reasons.append(f"recorded {recorded} of {requested} requested successful demos")
    if recorded > 0:
        if not hdf5 or hdf5.get("error"):
            reasons.append(f"could not read the output file back: {(hdf5 or {}).get('error', 'missing summary')}")
        elif hdf5.get("num_demos") != recorded:
            reasons.append(f"output file has {hdf5.get('num_demos')} demos but {recorded} were exported")
        elif not all(hdf5.get("success", [])):
            reasons.append("output file contains demos not flagged successful")
    return reasons


def evaluate_annotate(processed: int, exported: int, hdf5: dict | None) -> list[str]:
    reasons = []
    if processed == 0:
        reasons.append("the input file has no episodes")
    elif exported < processed:
        reasons.append(f"annotated {exported} of {processed} episodes (the rest failed replay or lack a subtask signal)")
    if exported > 0 and (not hdf5 or hdf5.get("error") or hdf5.get("num_demos") != exported):
        reasons.append(f"output file read-back disagrees with the {exported} exported episodes: {hdf5}")
    return reasons


def evaluate_generate(
    returncode: int | None, timed_out: bool, hdf5: dict | None, min_generated: int, accept_timeout: bool
) -> list[str]:
    reasons = []
    if timed_out and not accept_timeout:
        reasons.append("Lab process exceeded --timeout_s and was killed")
    elif not timed_out and returncode != 0:
        reasons.append(f"Lab process exited with code {returncode}")
    if not hdf5 or hdf5.get("error"):
        reasons.append(f"no readable output file: {(hdf5 or {}).get('error', 'missing')}")
    else:
        if hdf5.get("num_demos", 0) < min_generated:
            reasons.append(f"output has {hdf5.get('num_demos')} demos, wanted >= {min_generated}")
        if not all(hdf5.get("success", [])):
            reasons.append("output contains demos not flagged successful")
    return reasons


# ---------------------------------------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------------------------------------
def _versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for pkg in ("isaaclab", "isaaclab_mimic", "isaaclab_tasks", "isaacsim", "isaaclab_arena", "lightwheel-sdk", "torch", "h5py"):
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
    return _resolve(args.report)


def _finish(exit_code: int) -> None:
    """Hard-exit; never calls simulation_app.close() (it exits 0 on failure and hangs >10 min, docs/runpod.md)."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - nothing sensible to do if flushing fails while exiting
        pass
    os._exit(exit_code)


def _new_report(args: argparse.Namespace, plan: dict) -> dict:
    return {
        "gate": f"W2b arena kitchen go2_arx5: {args.command}",
        "command": args.command,
        "passed": False,
        "failure_reasons": ["incomplete (still running or killed)"],
        "plan": plan,
        "versions": {},
        "progress": {"stage": "start"},
        "metrics": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on bad args
    problems = validate_args(args)
    if problems:
        print("[ERROR] " + "; ".join(problems), file=sys.stderr)
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
        for warning in plan["warnings"]:
            print(f"[WARNING] {warning}", file=sys.stderr)
        return 1 if plan["problems"] else 0
    if plan["problems"]:
        print("[ERROR] " + "; ".join(plan["problems"]), file=sys.stderr)
        return 2

    report = _new_report(args, plan)
    report_path = _report_path(args)
    simulation_app = None
    exit_code = 1
    try:
        report["versions"] = _versions()
        _write_report(report_path, report)  # skeleton first: a crash while Kit starts still leaves a report
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        if args.command != "generate":
            registry.get_backend(BACKEND).prepare_scene_env(plan["scene_id"])  # before Kit starts (no-op for kitchens)
            simulation_app = _launch_app(args)
            _purge_leaked_isaaclab_assets()
        try:
            RUNNERS[args.command](args, plan, report, report_path)
        except BaseException as e:  # noqa: BLE001 - includes SystemExit/KeyboardInterrupt
            traceback.print_exc()
            report["passed"] = False
            report["failure_reasons"] = [
                *report.get("failure_reasons", []),
                f"exception at stage {report['progress'].get('stage')!r}: {e!r}",
            ]
            report["error"] = traceback.format_exc()
        if not report["passed"] and report["failure_reasons"] == ["incomplete (still running or killed)"]:
            report["failure_reasons"] = ["runner returned without a verdict"]
        wrote = _write_report(report_path, report)  # report FIRST, before any teardown
        if not wrote:
            report["passed"] = False
        print(f"[INFO] report: {report_path}", flush=True)
        print(f"[INFO] {'PASS' if report['passed'] else 'FAIL'}: {report['failure_reasons']}", flush=True)
        exit_code = 0 if report["passed"] else 1
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        report["passed"] = False
        report["failure_reasons"] = [f"exception outside the runner: {e!r}"]
        report["error"] = traceback.format_exc()
        _write_report(report_path, report)
        exit_code = 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        del simulation_app  # never closed
        _finish(exit_code)
    return exit_code  # only reached if _finish returns (tests)


def _launch_app(args: argparse.Namespace):
    """Start Kit through AppLauncher with launcher defaults from a throwaway parser (works on Lab 2.3 and 3.0 lines)."""
    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    app_args = launcher_parser.parse_args([])
    app_args.headless = bool(args.headless)
    if hasattr(app_args, "device"):
        app_args.device = args.device
    return AppLauncher(app_args).app


def _purge_leaked_isaaclab_assets() -> None:
    """Same workaround as Arena's isaaclab_interop._purge_leaked_isaaclab_assets_presets (Lab issue #6514).

    Kit auto-loads the isaaclab_assets extension while the app starts and leaves preset modules bound to a duplicate
    ``ArticulationCfg`` class in ``sys.modules``; the Go2 preset we build the robot from would then be rejected by
    ``InteractiveScene`` ("Unknown asset config type for robot"). Not a patch of Lab: it only drops our own cached imports.
    """
    for name in [n for n in sys.modules if n == "isaaclab_assets" or n.startswith("isaaclab_assets.")]:
        del sys.modules[name]


# ---------------------------------------------------------------------------------------------------------
# Isaac-side helpers (lazy imports; only after the app launched)
# ---------------------------------------------------------------------------------------------------------
def _compose_env(args: argparse.Namespace, mimic: bool):
    """Build the Arena environment through Arena's builder and register it with gym.

    Returns ``(gym_name, parsed_env_cfg, env_kwargs)``. ArenaBackend.build_env is not used: it has no ``mimic`` switch
    (and fixes ``solve_relations=False``, which is what we want anyway). Scene, embodiment and registration still go
    through ``registry.get_backend('arena')`` inside the environment factory.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    factory = emb.get_arena_classes()[emb.ENV_FACTORY_CLASS]()
    arena_env = factory.get_env(args)
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(
            num_envs=args.num_envs, seed=args.seed, solve_relations=False, mimic=mimic, device=args.device
        ),
    )
    return builder.build_registered()


def _never_terminate(env):
    """Inert replacement for the success termination (as record_demos.py): success is evaluated by hand."""
    import torch

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _env_positions(env, key: str) -> list[float]:
    """Env-local position of scene entity `key` in env 0."""
    from common.cerberus_compat.isaac_data import as_tensor

    pos = as_tensor(env.scene[key].data.root_pos_w) - env.scene.env_origins
    return [float(v) for v in pos[0].tolist()]


def _env_pose(env, key: str) -> dict[str, list[float]]:
    """Env-local object root pose, including the simulator's reported world quaternion."""
    from common.cerberus_compat.isaac_data import as_tensor

    data = env.scene[key].data
    pos = as_tensor(data.root_pos_w) - env.scene.env_origins
    quat = as_tensor(data.root_quat_w)
    return {
        "position_m": [float(v) for v in pos[0].tolist()],
        "root_quaternion_wxyz": [float(v) for v in quat[0].tolist()],
    }


def _env0_value(value):
    """JSON-friendly value for a single-env observation or tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _failure_diagnostics(obs, target_pos, rot_error, pick_pose, phase: str, joint_state=None) -> dict:
    """Build additive terminal evidence without changing the existing attempt/report schema."""
    policy = obs.get("policy", {})
    eef_pos = _env0_value(policy.get("eef_pos"))
    fingers = _env0_value(policy.get("gripper_pos"))
    target = None if target_pos is None else [float(v) for v in target_pos]
    actual = None if eef_pos is None else [float(v) for v in eef_pos]
    error_vec = None if target is None or actual is None else [t - a for t, a in zip(target, actual)]
    pos_error = None if error_vec is None else math.sqrt(sum(v * v for v in error_vec))
    subtask = obs.get("subtask_terms", {})
    grasp = _env0_value(subtask.get("grasp_1")) if isinstance(subtask, dict) else None
    if isinstance(grasp, list) and len(grasp) == 1:
        grasp = grasp[0]
    if hasattr(grasp, "item"):
        grasp = grasp.item()
    if isinstance(grasp, (bool, int, float, str)) or grasp is None:
        grasp_signal = grasp
    else:
        grasp_signal = str(grasp)
    return {
        "phase": phase,
        "eef_position_m": actual,
        "target_eef_position_m": target,
        "position_error_vector_m": error_vec,
        "position_error_m": pos_error,
        "rotation_error_vector_rad": None if rot_error is None else [float(v) for v in rot_error],
        "finger_positions_m": None if fingers is None else [float(v) for v in fingers],
        "pick_object_root_pose": pick_pose,
        "grasp_1_signal": grasp_signal,
        "arm_joint_state": joint_state,
    }


def _arm_joint_state(env) -> list[dict] | None:
    """Best-effort arm joint positions and soft limits for diagnosing IK saturation."""
    from common.cerberus_compat.isaac_data import as_tensor
    from common.cerberus_description.go2_arx5_spec import ARX5_ARM_JOINTS

    try:
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(list(ARX5_ARM_JOINTS), preserve_order=True)
        positions = as_tensor(robot.data.joint_pos)[0, ids].tolist()
        limits_data = getattr(robot.data, "soft_joint_pos_limits", None)
        limits = as_tensor(limits_data)[0, ids].tolist() if limits_data is not None else None
    except Exception:  # noqa: BLE001 - diagnostics must never turn a task failure into a recorder exception
        return None
    return [
        {
            "name": name,
            "position_rad": float(position),
            "soft_limits_rad": None if limits is None else [float(v) for v in limits[i]],
        }
        for i, (name, position) in enumerate(zip(ARX5_ARM_JOINTS, positions))
    ]


def _tensors(obj, path="obs"):
    import torch

    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{path}/{k}")


# ---------------------------------------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------------------------------------
def _run_smoke(args: argparse.Namespace, plan: dict, report: dict, report_path: Path) -> None:
    import gymnasium as gym
    import torch

    from common.cerberus_compat.isaac_data import as_tensor
    from common.cerberus_description.go2_arx5_spec import ARX5_ARM_JOINTS

    progress, metrics = report["progress"], report["metrics"]
    t_start = time.perf_counter()

    def stage(name: str) -> None:
        progress["stage"] = name
        progress[f"t_{name}"] = round(time.perf_counter() - t_start, 2)
        _write_report(report_path, report)
        print(f"[INFO] stage: {name}", flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    opts = emb.options_from_namespace(args)
    mimic = args.env_kind == "mimic"
    stage("register_embodiment")
    registry.get_backend(BACKEND).register_embodiment(GO2_ARX5_EMBODIMENT)
    stage("compose_env")
    name, cfg, env_kwargs = _compose_env(args, mimic=mimic)
    stage("make_env")
    env = (gym.make(name, cfg=cfg) if mimic else gym.make(name, cfg=cfg, **env_kwargs)).unwrapped
    metrics.update(env_kind=args.env_kind, num_envs=env.num_envs, env_class=type(env).__name__, pick_name=opts.pick_object)

    stage("reset")
    obs, _ = reset_all(env)
    robot = env.scene["robot"]
    metrics["scene_keys"] = sorted(env.scene.keys())
    metrics["is_fixed_base"] = bool(getattr(robot, "is_fixed_base", None))
    metrics["action_dim"] = int(env.action_manager.total_action_dim)
    metrics["action_terms"] = list(env.action_manager.active_terms)
    metrics["obs_groups"] = sorted(obs.keys())
    metrics["policy_obs_keys"] = sorted(obs["policy"].keys())
    metrics["policy_obs_shapes"] = {k: list(v.shape) for k, v in obs["policy"].items()}
    metrics["subtask_obs_keys"] = sorted(obs.get("subtask_terms", {}).keys())
    metrics["joint_names"] = list(robot.joint_names)
    arm_ids, _ = robot.find_joints(list(ARX5_ARM_JOINTS), preserve_order=True)
    arm0 = as_tensor(robot.data.joint_pos)[:, arm_ids].clone()
    base_z0 = as_tensor(robot.data.root_pos_w)[:, 2].clone()

    stage("step")
    action = torch.zeros(env.num_envs, emb.ACTION_DIM, device=env.device)  # zero pose delta, gripper >= 0 = open
    step_times = []
    finite = True
    with torch.inference_mode():
        for _ in range(args.steps):
            t0 = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)
            step_times.append(time.perf_counter() - t0)
            for _, tensor in _tensors(obs):
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    finite = False
        arm1 = as_tensor(robot.data.joint_pos)[:, arm_ids]
        metrics["arm_drift_rad"] = float((arm1 - arm0).abs().max())
        metrics["base_z_drift"] = float((as_tensor(robot.data.root_pos_w)[:, 2] - base_z0).abs().max())
    metrics["finite"] = finite
    metrics["step_s_mean"] = round(sum(step_times) / len(step_times), 4)
    metrics["eef_pos"] = [float(v) for v in obs["policy"]["eef_pos"][0].tolist()]
    metrics["gripper_pos"] = [float(v) for v in obs["policy"]["gripper_pos"][0].tolist()]
    geometry = plan["environment"]["geometry"]
    top = float(geometry["support_top_z"])
    pick_pos = _env_positions(env, opts.pick_object)
    plate_pos = _env_positions(env, emb.PLATE_NAME)
    metrics["pick_pos"], metrics["plate_pos"] = pick_pos, plate_pos
    metrics["pick_dz_to_support"] = pick_pos[2] - top
    metrics["plate_dz_to_support"] = plate_pos[2] - top
    if mimic:
        with torch.inference_mode():
            metrics["mimic_eef_pose_shape"] = list(env.get_robot_eef_pose(next(iter(env.cfg.subtask_configs))).shape)
            metrics["mimic_object_pose_keys"] = sorted(env.get_object_poses().keys())
            metrics["mimic_signal_keys"] = sorted(env.get_subtask_term_signals().keys())
            metrics["mimic_eef_names"] = list(env.cfg.subtask_configs.keys())
    reasons = evaluate_smoke(metrics)
    report["passed"] = not reasons
    report["failure_reasons"] = reasons
    stage("done")


# ---------------------------------------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------------------------------------
def _run_record(args: argparse.Namespace, plan: dict, report: dict, report_path: Path) -> None:
    import gymnasium as gym
    import torch
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    from isaaclab.utils import math as math_utils

    progress = report["progress"]
    attempts: list[dict] = []
    progress["attempts"] = attempts
    progress["recorded"] = 0
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opts = emb.options_from_namespace(args)
    random.seed(args.seed)  # the pose reset events sample with python's `random`
    torch.manual_seed(args.seed)

    progress["stage"] = "compose_env"
    registry.get_backend(BACKEND).register_embodiment(GO2_ARX5_EMBODIMENT)
    # Recording on the mimic env: Arena excludes metrics/recorders from it, so replacing `recorders` is safe, and it is
    # the env class annotate/generate replay on.  UNVERIFIED: dynamics are the same as the plain RL env.
    name, env_cfg, _ = _compose_env(args, mimic=True)
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise RuntimeError(f"env {name} has no terminations.success; cannot label demos as successful")
    env_cfg.terminations.success = success_term.replace(func=_never_terminate, params={})
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output.parent)
    env_cfg.recorders.dataset_filename = output.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.env_name = name

    progress["stage"] = "make_env"
    env = gym.make(name, cfg=env_cfg).unwrapped
    machine = ScriptedPickPlace(pick_place_config(args))
    pitch = torch.tensor([math.radians(args.grasp_pitch_deg)], dtype=torch.float32, device=env.device)
    pitch_axis = torch.tensor([[0.0, 1.0, 0.0]], device=env.device)
    yaw_axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
    q_pitch = math_utils.quat_from_angle_axis(pitch, pitch_axis)  # rotation about world Y
    # UNVERIFIED: local X5A finger axis and fetched SweetPotato003 bbox motivated a 90° offset; bbox and random-yaw projection remain empirical until the pod run.
    hold = torch.zeros(1, emb.ACTION_DIM, device=env.device)
    hold[:, -1] = 1.0  # gripper open

    def reset_env():
        env.sim.reset()  # as record_demos.py handle_reset with --reset_sim_buffer_each_episode
        env.recorder_manager.reset()
        obs, _ = reset_all(env)
        return obs

    def run_attempt(index: int) -> dict:
        obs = reset_env()
        for _ in range(args.pre_settle_steps):  # let the object settle on the riser before planning
            obs, *_ = env.step(hold)
        pick = _env_positions(env, opts.pick_object)
        pick_yaw = _yaw_from_quaternion(math_utils, env.scene[opts.pick_object].data.root_quat_w)
        grasp_yaw = pick_yaw + math.radians(args.grasp_yaw_deg)
        q_yaw = math_utils.quat_from_angle_axis(grasp_yaw, yaw_axis)  # world-Z yaw = object yaw + offset
        plate = _env_positions(env, emb.PLATE_NAME)
        pick_pos = (pick[0], pick[1], pick[2] + args.pick_center_dz)
        machine.reset(
            pick_pos=pick_pos, place_pos=emb.place_reference_pos(plate, opts.plate_thickness_m, opts.object_height_m)
        )
        q_des = _compose_grasp_orientation(math_utils, q_yaw, q_pitch, obs["policy"]["eef_quat"])
        pick_yaw_deg = math.degrees(float(pick_yaw[0].item()))
        record = {
            "index": index,
            "pick": pick,
            "plate": plate,
            "object_yaw_deg": pick_yaw_deg,
            "grasp_yaw_offset_deg": args.grasp_yaw_deg,
            "grasp_yaw_deg": pick_yaw_deg + args.grasp_yaw_deg,
            "status": "max_episode_steps",
        }
        success_steps = settle = step = 0
        terminal_target = terminal_phase = terminal_rot_error = None
        for step in range(1, args.max_episode_steps + 1):
            pol = obs["policy"]
            eef = tuple(pol["eef_pos"][0].tolist())
            rot_err = math_utils.axis_angle_from_quat(math_utils.quat_mul(q_des, math_utils.quat_inv(pol["eef_quat"])))
            active_target = machine.target
            if active_target is not None:
                terminal_target = active_target
                terminal_phase = machine.phase.value
            terminal_rot_error = tuple(rot_err[0].tolist())
            act = machine.step(eef, terminal_rot_error)
            if machine.failed:
                record["status"], record["failure_reason"] = "script_failed", machine.failure_reason
                break
            action = torch.tensor([act.as_list()], dtype=torch.float32, device=env.device)
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):  # e.g. the object fell off the riser
                record["status"] = "env_terminated"
                break
            success_steps = success_steps + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
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
        record["steps"], record["phase"] = step, machine.phase.value
        if record["status"] in ("script_failed", "done_without_success"):
            record["diagnostics"] = _failure_diagnostics(
                obs=obs,
                target_pos=terminal_target,
                rot_error=terminal_rot_error,
                pick_pose=_env_pose(env, opts.pick_object),
                phase=terminal_phase or machine.phase.value,
                joint_state=_arm_joint_state(env),
            )
        return record

    budget = attempt_budget(args)
    error = None
    try:
        with torch.inference_mode():
            while progress["recorded"] < args.num_demos and len(attempts) < budget:
                progress["stage"] = f"attempt_{len(attempts)}"
                record = run_attempt(len(attempts))
                attempts.append(record)
                progress["recorded"] = int(env.recorder_manager.exported_successful_episode_count)
                print(
                    f"[INFO] attempt {record['index']}: {record['status']} in {record['steps']} steps "
                    f"(phase {record['phase']}); recorded {progress['recorded']}/{args.num_demos}",
                    flush=True,
                )
                _write_report(report_path, report)
    except BaseException as e:  # noqa: BLE001 - the partial file is still summarised below
        error = f"{e!r}\n{traceback.format_exc()}"
        traceback.print_exc()
    try:  # flush/close the HDF5 stream (not the env / app) so the file is complete on disk
        env.recorder_manager.close()
    except Exception:  # noqa: BLE001 - the read-back below says whether the file is usable
        traceback.print_exc()
    hdf5 = summarize_hdf5(output) if progress["recorded"] > 0 and output.exists() else None
    statuses: dict[str, int] = {}
    for a in attempts:
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    report["metrics"].update(attempt_status_counts=statuses, hdf5=hdf5, output=str(output))
    reasons = evaluate_record(args.num_demos, progress["recorded"], hdf5, error)
    report["passed"] = not reasons
    report["failure_reasons"] = reasons
    if error:
        report["error"] = error


# ---------------------------------------------------------------------------------------------------------
# annotate (in-process port of Isaac Lab's annotate_demos.py auto mode)
# ---------------------------------------------------------------------------------------------------------
def _run_annotate(args: argparse.Namespace, plan: dict, report: dict, report_path: Path) -> None:
    if args.engine == "lab":
        _run_lab_subprocess(args, plan, report, report_path, "annotate")
        return
    import gymnasium as gym
    import torch
    from isaaclab.envs import ManagerBasedRLMimicEnv
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    progress, metrics = report["progress"], report["metrics"]
    input_file, output_file = _resolve(args.input_file), _resolve(args.output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"input dataset {input_file} does not exist")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- recorder terms, as in annotate_demos.py (they are defined inside that script, not importable) ---------------
    class PreStepDatagenInfoRecorder(RecorderTerm):
        def record_pre_step(self):
            eef_pose_dict = {n: self._env.get_robot_eef_pose(eef_name=n) for n in self._env.cfg.subtask_configs.keys()}
            datagen_info = {
                "object_pose": self._env.get_object_poses(),
                "eef_pose": eef_pose_dict,
                "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
            }
            return "obs/datagen_info", datagen_info

    @configclass
    class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder

    class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
        def record_pre_step(self):
            return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()

    @configclass
    class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder

    @configclass
    class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
        record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
        record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    progress["stage"] = "compose_env"
    registry.get_backend(BACKEND).register_embodiment(GO2_ARX5_EMBODIMENT)
    name, env_cfg, _ = _compose_env(args, mimic=True)
    env_cfg.env_name = name
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is None:
        raise RuntimeError("no success termination term found in the environment")
    env_cfg.terminations.success = None
    env_cfg.terminations = None  # as annotate_demos.py: all termination terms off, success checked by hand
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = MimicRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output_file.parent)
    env_cfg.recorders.dataset_filename = output_file.stem

    progress["stage"] = "make_env"
    env = gym.make(name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("the environment must derive from ManagerBasedRLMimicEnv")
    if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
        raise NotImplementedError("the mimic env does not implement get_subtask_term_signals (needed for --auto)")
    reset_all(env)

    handler = HDF5DatasetFileHandler()
    handler.open(str(input_file))
    names = list(handler.get_episode_names())
    processed = exported = 0
    episodes: list[dict] = []
    progress["episodes"] = episodes
    try:
        with torch.inference_mode():
            for episode_index, episode_name in enumerate(names):
                processed += 1
                progress["stage"] = f"episode_{episode_index}"
                episode = handler.load_episode(episode_name, env.device)
                info = {"index": episode_index, "name": episode_name, "replay_success": False, "signals": {}}
                # replay_episode() of annotate_demos.py
                env.sim.reset()
                env.recorder_manager.reset()
                env.reset_to(episode.data["initial_state"], None, is_relative=True)
                for action in episode.data["actions"]:
                    env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1))
                info["replay_success"] = bool(success_term.func(env, **success_term.params)[0])
                ok = info["replay_success"]
                if ok:
                    annotated = env.recorder_manager.get_episode(0)
                    signals = annotated.data["obs"]["datagen_info"]["subtask_term_signals"]
                    for signal_name, flags in signals.items():
                        fired = bool(torch.any(torch.as_tensor(flags, device=env.device)))
                        info["signals"][signal_name] = fired
                        ok = ok and fired
                if ok:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported += 1
                info["exported"] = ok
                episodes.append(info)
                print(f"[INFO] episode {episode_index} ({episode_name}): {info}", flush=True)
                _write_report(report_path, report)
    finally:
        try:
            env.recorder_manager.close()  # flush the HDF5 stream; not env.close()/app close
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    hdf5 = summarize_hdf5(output_file) if exported > 0 and output_file.exists() else None
    metrics.update(processed=processed, exported=exported, hdf5=hdf5, output=str(output_file))
    reasons = evaluate_annotate(processed, exported, hdf5)
    report["passed"] = not reasons
    report["failure_reasons"] = reasons


# ---------------------------------------------------------------------------------------------------------
# generate (and annotate --engine lab): Lab scripts through Arena's interop callback, as a subprocess
# ---------------------------------------------------------------------------------------------------------
def _run_lab_subprocess(args: argparse.Namespace, plan: dict, report: dict, report_path: Path, stage: str) -> None:
    cmd = plan["lab_command"]
    output_file = _resolve(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_path = report_path.with_suffix(".log")
    metrics = report["metrics"]
    metrics.update(lab_command=plan["lab_command_line"], log=str(log_path), output=str(output_file))
    report["progress"]["stage"] = f"lab_{stage}"
    _write_report(report_path, report)
    timed_out = False
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, env=lab_env(headless=True), cwd=str(_REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )
        try:
            returncode = proc.wait(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            returncode = proc.wait()
    metrics.update(returncode=returncode, timed_out=timed_out, duration_s=round(time.perf_counter() - t0, 1))
    hdf5 = summarize_hdf5(output_file) if output_file.exists() else None
    metrics["hdf5"] = hdf5
    if stage == "generate":
        reasons = evaluate_generate(returncode, timed_out, hdf5, args.min_generated, args.accept_timeout)
    else:
        reasons = evaluate_generate(returncode, timed_out, hdf5, 1, False)
    report["passed"] = not reasons
    report["failure_reasons"] = reasons


def _run_generate(args: argparse.Namespace, plan: dict, report: dict, report_path: Path) -> None:
    _run_lab_subprocess(args, plan, report, report_path, "generate")


RUNNERS = {
    "smoke": _run_smoke,
    "record": _run_record,
    "annotate": _run_annotate,
    "generate": _run_generate,
}


if __name__ == "__main__":
    sys.exit(main())
