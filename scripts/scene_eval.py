#!/usr/bin/env python3
"""A2/A5/A6 driver: the frozen Go2 rough policy standing (and optionally walking a scripted path) in a GRScenes home.

What it proves, in one short GPU session:

* the scene env composes (Play cfg + scene USD + scan proxy) and the robot does not sink or fall through;
* the free assertions (``assertions.json``): joint order vs the contract and ``platforms.yaml``, a 235-D observation,
  the command slice at its contract offset, the height-scan constant on the floor, no NaN/inf over 200 steps;
* optionally (``--scripted``) the command hook, the policy and the video helper, by walking a seeded open-loop path.

It does NOT evaluate navigation -- that is ``scripts/nav_eval.py``.

Run on the pod (after the app-launcher environment is in place):

    /workspace/isaaclab/isaaclab.sh -p scripts/scene_eval.py \
        --scene grscenes:<id> --num_envs 4 --steps 200 --checkpoint <ckpt>.pt --headless

``--dry-run`` resolves the scene, spawns and output paths without Isaac (works locally).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd",
                        help="scene id resolved through the plain_usd backend")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="envs = validated spawn poses used; all envs share one scene prim, so keep it small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="zero-command stand length (the assertion window)")
    parser.add_argument("--task", default=None, help="Isaac task id (default: the Cerberus rough Play task)")
    parser.add_argument("--checkpoint", default="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt")
    parser.add_argument("--scripted", action="store_true", help="after the stand, walk a seeded scripted path (metrics only)")
    parser.add_argument("--record-clip", dest="record_clip", action="store_true",
                        help="also record the scripted-path clip; this enables cameras for the whole run")
    parser.add_argument("--scripted-waypoints", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--video", dest="video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--video-steps", type=int, default=300)
    parser.add_argument("--video-downscale", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=30, help="clip frame rate; 50 matches the 50 Hz control rate")
    parser.add_argument("--video-resolution", default=None, metavar="WxH",
                        help="viewport resolution for the clip, e.g. 1920x1080")
    parser.add_argument("--clip-name", default=None, help="clip filename (default <scene>_<seed>_scripted.mp4)")
    parser.add_argument("--demo-seconds", type=float, default=0.0,
                        help="repeat the seeded tour until it covers this much sim time (showcase clip)")
    parser.add_argument("--demo-speed", type=float, default=0.3, help="linear speed of the repeated demo tour")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                        help="rerun the stand test even if assertions.json already passed")
    parser.add_argument("--dry-run", action="store_true", help="resolve paths and print the plan; no Isaac")
    parser.add_argument("--build-proxy-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assert-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return _REPO_ROOT / "outputs" / "scene_eval" / stamp


def spawn_poses_for(scene_id: str, n: int) -> list[tuple[float, float, float, float]]:
    """The first ``n`` validated spawn poses from the admission manifest (never a silent (0, 0, 0) fallback)."""
    from common.cerberus_worlds.admission import spawn_pose

    if scene_id in {"stock", "grscenes:stock"}:
        return [(0.0, 0.0, 0.0, 0.0)] * n
    source = scene_id.split(":", 1)[0]
    poses = []
    for i in range(n):
        p = spawn_pose(source, scene_id, i)
        poses.append((p.x, p.y, p.yaw, p.floor_z))
    return poses


def ensure_scan_proxy(scene: dict, floor_z: float) -> Path:
    """Build ``scan_proxy.usd`` beside the scene USD when it is missing (pxr only; a few minutes on a big house)."""
    from common.cerberus_worlds.scene_geometry import build_scan_proxy, scan_proxy_path

    proxy = scan_proxy_path(scene["usd_path"])
    if proxy.is_file():
        return proxy
    print(f"[scene_eval] building height-scan proxy {proxy} (floor_z={floor_z})", flush=True)
    build_scan_proxy(scene["usd_path"], proxy, floor_z=floor_z)
    return proxy


def prebuild_scan_proxy(args) -> None:
    """Build ``scan_proxy.usd`` in a child interpreter BEFORE Kit starts.

    The proxy build is ~13 s of pure ``pxr`` work. Doing it on the main thread after AppLauncher blocks Kit's loop,
    and Kit exits with code 0 before the first gym.make (measured 2026-11-22 on 580.95.05).
    """
    import subprocess
    import sys as _sys

    from common.cerberus_worlds.registry import get_backend
    from common.cerberus_worlds.scene_geometry import scan_proxy_path

    if args.scene in {"stock", "grscenes:stock"}:
        return
    spawns = spawn_poses_for(args.scene, 1)
    info = get_backend("plain_usd").load_scene(args.scene)
    proxy = scan_proxy_path(info["usd_path"])
    if proxy.is_file():
        return
    print(f"[scene_eval] pre-building height-scan proxy (child) floor_z={spawns[0][3]}", flush=True)
    subprocess.run([_sys.executable, str(Path(__file__)), "--build-proxy-only", "--scene", args.scene],
                   check=True)


def run_assert_only(out_dir: Path) -> int:
    """Build and write the contract assertions from the inputs the sim dumped (a child process, no Kit)."""
    from common.cerberus_eval.scene_assertions import build_report

    data = json.loads((out_dir / "assert_inputs.json").read_text(encoding="utf-8"))
    report = build_report(
        joint_names=data["joint_names"],
        obs_dim=int(data["obs_dim"]),
        obs_row=data["obs_row"],
        command=data["command"],
        scan=data["scan"],
        base_height_m=data["base_height_m"],
        stand_rows=data["stand_rows"],
        stand_steps_expected=int(data["stand_steps_expected"]),
    )
    if data.get("scan_report") is not None:
        report.soft["height_scan_flat_ground"] = data["scan_report"]
    report.add("no_fall_during_stand", not data["fell_envs"], {"fell_envs": data["fell_envs"]})
    if data.get("min_base_minus_foot") is None:
        report.soft["robot_stands"] = "skipped (base/foot body names not found in the articulation)"
    else:
        report.add("robot_stands", data["min_base_minus_foot"] > 0.15,
                   {"min_base_minus_foot_z": data["min_base_minus_foot"]})
    report.soft["scene"] = data["scene"]
    report.write(out_dir / "assertions.json")
    print(f"[scene_eval] assertions: {'PASS' if report.passed else 'FAIL'} -> {out_dir / 'assertions.json'}")
    for check in report.checks:
        print(f"[scene_eval]   {check}")
    return 0 if report.passed else 1


def fresh_observations(env):
    """Recompute the observation after a command write; ``get_observations()`` can return the cached reset value."""
    return env.unwrapped.observation_manager.compute()


def policy_observation(obs):
    """The flat policy tensor from ``env.get_observations()`` (a TensorDict with a ``policy`` group on Lab 3.0)."""
    try:
        return obs["policy"]
    except (KeyError, TypeError):
        return obs


def run(args, app=None) -> int:
    import torch

    from common.cerberus_eval.nav_metrics import write_run_json
    from common.cerberus_eval.scene_assertions import (
        contract_terms,
        height_scan_report,
        term_offsets,
    )
    from common.cerberus_eval.scripted_path import scripted_path_for_seed
    from common.cerberus_eval.video_clips import ClipRecorder, metrics_filename
    from policy.locomotion.eval import scene_env as se

    task = args.task or se.LOCO_TASK
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        prior = out_dir / "assertions.json"
        clip_ok = (not args.scripted) or any((out_dir / "clips").glob("*_scripted.mp4"))
        if prior.is_file() and clip_ok:
            try:
                done = json.loads(prior.read_text(encoding="utf-8"))
            except ValueError:
                done = {}
            if done.get("passed"):
                print(f"[scene_eval] resume: {out_dir} already complete and passed; nothing to do")
                return 0

    if args.resume:
        prior = out_dir / "assertions.json"
        clip_ok = (not args.scripted) or any((out_dir / "clips").glob("*_scripted.mp4"))
        if prior.is_file() and clip_ok:
            try:
                done = json.loads(prior.read_text(encoding="utf-8"))
            except ValueError:
                done = {}
            if done.get("passed"):
                print(f"[scene_eval] resume: {out_dir} already complete and passed; nothing to do")
                return 0

    spawns = spawn_poses_for(args.scene, args.num_envs)
    scene = se.resolve_scene(args.scene)
    proxy = ensure_scan_proxy(scene, spawns[0][3])

    resolution = None
    if args.video_resolution:
        try:
            width, height = (int(v) for v in args.video_resolution.lower().split("x"))
        except ValueError:
            raise SystemExit("--video-resolution must look like 1920x1080") from None
        resolution = (width, height)
    env, scene = se.build_loco_env(
        args.scene, spawns, task=task,
        episode_length_s=max(120.0, args.steps * 0.02 + args.demo_seconds + 30.0),
        seed=args.seed, scan_proxy=proxy, timings_path=out_dir / "build_timings.json",
        with_cameras=bool(args.record_clip), camera_resolution=resolution,
    )
    hook = se.VelocityCommandHook(env)
    policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
    print("[scene_eval] POLICY_OK", flush=True)
    device = env.unwrapped.device
    n = env.unwrapped.num_envs

    joint_names = list(env.unwrapped.scene["robot"].joint_names)
    print(f"[scene_eval] JOINTS {len(joint_names)}", flush=True)
    hook.zero()
    print("[scene_eval] HOOK_ZERO", flush=True)
    obs = fresh_observations(env)
    print("[scene_eval] OBS_OK", flush=True)
    flat = policy_observation(obs)
    obs_row = flat[0].detach().cpu().tolist()
    obs_dim = int(flat.shape[-1])
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    print("[scene_eval] STAND_METRICS", flush=True)
    stand = se.stand_metrics(env)
    print("[scene_eval] STAND_METRICS_OK", flush=True)
    base_height = None
    if stand is not None:
        base_height = float(stand["base_minus_foot_z"].mean().item())
    scan_start, scan_stop = term_offsets(contract_terms())["height_scan"]
    scan = obs_row[scan_start:scan_stop] if obs_dim == 235 else None
    scan_report = height_scan_report(scan, base_height_m=base_height) if scan else None

    zero = torch.zeros((n, 3), device=device)
    rows: list[list[float]] = []
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    first_step_s = None
    for step_index in range(args.steps):
        hook.apply(zero)
        obs = fresh_observations(env)
        rows.append(policy_observation(obs)[0].detach().cpu().tolist())
        t_step = time.perf_counter()
        print(f"[scene_eval] STEP {step_index}", flush=True)
        obs, _, _, _ = env.step(policy(obs))
        if step_index == 0:
            first_step_s = time.perf_counter() - t_step
        fell |= se.read_fall_flags(env)
        if bool(fell.any()):
            break
    print("[scene_eval] LOOP_DONE", flush=True)
    print("[scene_eval] WRITE_TIMINGS", flush=True)
    if first_step_s is not None:
        se.write_timings(out_dir / "build_timings.json", {"first_step_s": first_step_s})
    print("[scene_eval] FINAL_STAND", flush=True)
    final_stand = se.stand_metrics(env)
    print("[scene_eval] FINAL_STAND_OK", flush=True)
    min_base_minus_foot = None
    if final_stand is not None:
        min_base_minus_foot = float(final_stand["base_minus_foot_z"].min().item())

    # Kit's native layer exits the process when the pure-Python report runs on the app thread. Dump the inputs and
    # build the report in a child interpreter instead; no assertion code ever runs inside the sim process.
    inputs = {
        "joint_names": joint_names,
        "obs_dim": obs_dim,
        "obs_row": obs_row,
        "command": command,
        "scan": scan,
        "base_height_m": base_height,
        "stand_rows": rows,
        "stand_steps_expected": args.steps,
        "scan_report": scan_report,
        "fell_envs": torch.nonzero(fell).flatten().tolist(),
        "min_base_minus_foot": min_base_minus_foot,
        "scene": {"scene_id": args.scene, "usd_path": scene.get("usd_path"), "num_envs": n,
                  "spawn_poses": spawns, "task": task, "checkpoint": args.checkpoint},
    }
    (out_dir / "assert_inputs.json").write_text(json.dumps(inputs, default=str), encoding="utf-8")
    assert_rc = subprocess.run([sys.executable, str(Path(__file__)), "--assert-only", "--out-dir", str(out_dir)],
                               check=False).returncode
    assertions_passed = assert_rc == 0

    # --- optional scripted path (command hook + policy + video helper) --------------------------------------
    summary: dict = {"assertions_passed": assertions_passed, "stand_steps": len(rows)}
    if args.scripted and assertions_passed:
        start = (spawns[0][0], spawns[0][1], spawns[0][2])
        from common.cerberus_eval.scripted_path import demo_path

        if args.demo_seconds > 0:
            path = demo_path(args.seed, start_pose=start, duration_s=args.demo_seconds,
                             dt=0.02, speed=args.demo_speed)
        else:
            path = scripted_path_for_seed(args.seed, start_pose=start, dt=0.02)
        clip = ClipRecorder(out_dir / "clips", fps=args.video_fps,
                            enabled=args.video and args.record_clip, downscale=args.video_downscale)
        name = args.clip_name or f"{args.scene.split(':', 1)[-1]}_{args.seed}_scripted.mp4"
        clip.begin(name)
        se.reset_envs_to(env, spawns)
        t0 = time.perf_counter()
        steps = min(args.video_steps, len(path.commands))
        with torch.inference_mode():
            for i in range(steps):
                cmd = path.commands[i]
                hook.apply(torch.tensor([[cmd["vx"], cmd["vy"], cmd["wz"]]] * n, device=device))
                obs = env.get_observations()
                obs, _, _, _ = env.step(policy(obs))
                if args.video and args.record_clip:
                    clip.frame(env.unwrapped, 0)
        clip.end()
        elapsed = time.perf_counter() - t0
        poses = se.read_poses(env)
        record = {
            "scene": args.scene,
            "seed": args.seed,
            "outcome": "scripted",
            "steps": steps,
            "sim_seconds": round(steps * path.dt, 3),
            "video_fps": args.video_fps,
            "video_resolution": list(resolution) if resolution else None,
            "wall_s": elapsed,
            "start_pose": start,
            "end_pose": [float(v) for v in poses[0].detach().cpu()],
            "expected_end_pose": list(path.expected_end_pose),
            "fell": bool(se.read_fall_flags(env).any()),
            "min_base_minus_foot_z": min_base_minus_foot,
        }
        write_run_json(out_dir / "clips" / metrics_filename(name), record)
        summary["scripted"] = record
        summary["clip"] = name
        print(f"[scene_eval] scripted path: {json.dumps(record, default=str)}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    if not assertions_passed:
        print("[scene_eval] FAIL: see assertions.json", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = "--dry-run" in argv
    proxy_only = "--build-proxy-only" in argv
    assert_only = "--assert-only" in argv
    parser = build_parser()
    if not dry and not proxy_only and not assert_only:
        from common.cerberus_compat.launcher import add_launcher_args

        add_launcher_args(parser)
    args = parser.parse_args(argv)

    if getattr(args, "assert_only", False):
        return run_assert_only(Path(args.out_dir) if args.out_dir else default_out_dir())

    if getattr(args, "build_proxy_only", False):
        spawns = spawn_poses_for(args.scene, 1)
        from common.cerberus_worlds.registry import get_backend

        scene = get_backend("plain_usd").load_scene(args.scene)
        print(ensure_scan_proxy(scene, spawns[0][3]))
        return 0

    if dry:
        spawns = spawn_poses_for(args.scene, args.num_envs)
        print(json.dumps({
            "scene": args.scene,
            "num_envs": args.num_envs,
            "spawn_poses": spawns,
            "out_dir": str(Path(args.out_dir) if args.out_dir else default_out_dir()),
            "scripted": args.scripted,
            "note": "dry run: no Isaac, no GPU; run this on the pod with isaaclab.sh -p",
        }, indent=2))
        return 0

    prebuild_scan_proxy(args)

    if getattr(args, "record_clip", False):
        args.enable_cameras = True  # the launcher only creates a capture-capable visualizer with this set
    # MDL_SYSTEM_PATH must be set BEFORE the app launches; the registry is pure python, so no Isaac import here.
    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    # Kit shuts itself down if the app loop is never driven; a few updates right after launch settle it. Keep this
    # short: each update can block until a frame, so a long loop here delays the build by minutes.
    for _ in range(5):
        simulation_app.update()
    try:
        return run(args, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
