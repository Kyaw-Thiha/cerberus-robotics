"""Shared plumbing for the Newton spike's stage scripts (kit-less launch, env creation, reporting).

Everything here assumes the kit-less Isaac Lab install described in ``spikes/newton/README.md``:
there is no Kit app to start, ``launch_simulation(cfg, args_cli)`` decides from the config whether any
runtime is needed, and ``gym.make(task, cfg=cfg)`` builds the environment inside that context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def build_parser(description: str) -> argparse.ArgumentParser:
    """A parser with Isaac Lab's launcher arguments (``--device``, ``--viz``, ...) and stage flags."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", required=True, help="output directory for this stage's artifacts")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    return parser


def out_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"[out] wrote {path}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[out] wrote {path}", flush=True)


class Timer:
    """Wall-clock marks, so the report can quote real stage timings."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.marks: list[tuple[str, float]] = []

    def mark(self, label: str) -> float:
        elapsed = time.time() - self.t0
        self.marks.append((label, round(elapsed, 2)))
        print(f"[time] {label} {elapsed:.1f}s", flush=True)
        return elapsed

    def as_dict(self) -> dict:
        return {"marks_s": dict(self.marks), "total_s": round(time.time() - self.t0, 2)}


def obs_vector(obs):
    """The policy observation as one tensor, whatever container the env hands back."""
    import torch

    from policy.locomotion.eval.policy_io import select_policy_obs

    value = select_policy_obs(obs)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def terminate_flags(env) -> dict:
    """Per-env termination/timeout flags for the step that just ran, across Lab naming variants."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    flags = {}
    for name in ("reset_terminated", "reset_time_outs", "reset_timeouts", "reset_buf"):
        value = getattr(base, name, None)
        if value is not None:
            flags[name] = value
    return flags


def term_bool(env, term: str):
    """A termination term's boolean tensor, or None when the term does not exist."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "termination_manager", None)
    if manager is None or term not in getattr(manager, "active_terms", []):
        return None
    value = manager.get_term(term)
    return value


def observation_layout(env) -> dict:
    """Observation group/term order and dims as the live manager reports them."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "observation_manager", None)
    out = {"groups": {}}
    if manager is None:
        return out
    active = getattr(manager, "active_terms", {}) or {}
    dims = getattr(manager, "group_obs_dim", None)
    for group, terms in active.items():
        names = [getattr(t, "name", str(t)) for t in terms]
        group_dims = None
        if isinstance(dims, dict) and group in dims:
            group_dims = dims[group]
        elif isinstance(dims, (list, tuple)) and len(dims) == len(active):
            group_dims = None  # older layout: one entry per group, matched by index below
        out["groups"][group] = {"terms": names, "term_dims": _dims_to_list(group_dims), "num_terms": len(names)}
    space = getattr(base, "observation_space", None) or getattr(base, "single_observation_space", None)
    if space is not None:
        out["space_shape"] = list(getattr(space, "shape", []) or [])
    return out


def _dims_to_list(value) -> list | None:
    """Normalise a per-term dim sequence: ``[(name, dim)]`` and ``[dim]`` both become ``[dim, ...]``."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    out = []
    try:
        for entry in value:
            if isinstance(entry, (list, tuple)):
                out.append(int(entry[-1]))
            else:
                out.append(int(entry))
    except (TypeError, ValueError):
        return None
    return out


def action_layout(env) -> dict:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    manager = getattr(base, "action_manager", None)
    out = {}
    if manager is not None:
        out["terms"] = [getattr(t, "name", str(t)) for t in getattr(manager, "active_terms", [])]
        out["total_action_dim"] = int(getattr(manager, "total_action_dim", -1))
        term_cfgs = {}
        for name, term_cfg in (getattr(manager, "active_terms", {}) or {}).items():
            params = {}
            for field in ("scale", "use_default_offset", "clip", "preserve_order", "joint_names", "joint_ids"):
                value = getattr(term_cfg, field, None)
                if value is not None:
                    params[field] = value if not hasattr(value, "tolist") else value.tolist()
            term_cfgs[name] = params
        out["term_cfgs"] = term_cfgs
    space = getattr(base, "action_space", None) or getattr(base, "single_action_space", None)
    if space is not None:
        out["space_shape"] = list(getattr(space, "shape", []) or [])
    return out


def joint_layout(env) -> dict:
    robot = env.unwrapped.scene["robot"]
    out = {
        "joint_names": list(robot.joint_names),
        "num_joints": int(robot.num_joints),
        "num_bodies": int(robot.num_bodies),
    }
    backend = getattr(robot, "backend_joint_names", None)
    if backend is not None:
        out["backend_joint_names"] = list(backend)
        out["joint_order_matches_backend"] = list(backend) == list(robot.joint_names)
    joint_ordering = getattr(robot, "joint_ordering", None)
    out["joint_ordering_active"] = joint_ordering is not None
    body_ordering = getattr(robot, "body_ordering", None)
    out["body_ordering_active"] = body_ordering is not None
    return out


def sim_info(env) -> dict:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    sim = getattr(base, "sim", None)
    cfg = getattr(base, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    physics = (getattr(sim, "physics", None) if sim is not None else None) or getattr(sim_cfg, "physics", None)
    info = {
        "physics_backend": type(physics).__name__ if physics is not None else None,
        "device": str(getattr(base, "device", "")),
    }
    if sim_cfg is not None:
        info["sim_dt"] = getattr(sim_cfg, "dt", None)
        info["use_newton_actuators"] = getattr(sim_cfg, "use_newton_actuators", None)
    info["decimation"] = getattr(cfg, "decimation", None)
    info["episode_length_s"] = getattr(cfg, "episode_length_s", None)
    dt, dec = info.get("sim_dt"), info.get("decimation")
    if dt is not None and dec:
        info["step_dt"] = float(dt) * int(dec)
    commands = getattr(cfg, "commands", None)
    vel = getattr(commands, "base_velocity", None) if commands is not None else None
    if vel is not None:
        info["command_ranges"] = {
            "lin_vel_x": list(getattr(vel.ranges, "lin_vel_x", []) or []),
            "lin_vel_y": list(getattr(vel.ranges, "lin_vel_y", []) or []),
            "ang_vel_z": list(getattr(vel.ranges, "ang_vel_z", []) or []),
        }
        info["resampling_time_range"] = list(getattr(vel, "resampling_time_range", []) or [])
        info["rel_standing_envs"] = getattr(vel, "rel_standing_envs", None)
        info["heading_command"] = getattr(vel, "heading_command", None)
    return info


def software_versions() -> dict:
    """Versions that decide whether a result is reproducible (Isaac Lab, Newton, Warp, torch, OpenUSD)."""
    import importlib.metadata as md

    out = {}
    for name in ("isaaclab", "isaaclab_newton", "isaaclab_tasks", "newton", "warp-lang", "torch",
                 "usd-exchange", "usd-core", "rsl-rl-lib", "numpy", "pyglet", "moviepy", "imageio-ffmpeg"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    try:
        from pxr import Usd

        out["openusd"] = ".".join(str(v) for v in Usd.GetVersion())
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out["openusd"] = f"unavailable: {type(exc).__name__}: {exc}"
    out["python"] = sys.version.split()[0]
    return out


def env_vars_that_matter() -> dict:
    keys = ("CUDA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "DISPLAY", "EGL_PLATFORM", "PYOPENGL_PLATFORM",
            "MDL_SYSTEM_PATH", "CERBERUS_ON_POD", "WP_CACHE_DIR")
    return {k: os.environ.get(k) for k in keys if k in os.environ}


def run_stage(main_fn) -> int:
    """Run a stage's ``main()`` and leave a traceback on disk if it dies; never hide the exit code."""
    try:
        return int(main_fn() or 0)
    except Exception:  # noqa: BLE001 - the spike reports failures instead of raising them at the shell
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        target = Path(os.environ.get("NEWTON_STAGE_OUT", "/workspace/stage_failure"))
        target.mkdir(parents=True, exist_ok=True)
        (target / "exception.txt").write_text(tb)
        return 1
