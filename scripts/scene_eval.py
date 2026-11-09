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
import sys
import time
from datetime import datetime, timezone
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
    parser.add_argument("--scripted", action="store_true", help="after the stand, walk a seeded scripted path")
    parser.add_argument("--scripted-waypoints", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--video", dest="video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--video-steps", type=int, default=300)
    parser.add_argument("--video-downscale", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="resolve paths and print the plan; no Isaac")
    return parser


def default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _REPO_ROOT / "outputs" / "scene_eval" / stamp


def spawn_poses_for(scene_id: str, n: int) -> list[tuple[float, float, float, float]]:
    """The first ``n`` validated spawn poses from the admission manifest (never a silent (0, 0, 0) fallback)."""
    from common.cerberus_worlds.admission import spawn_pose

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


def run(args) -> int:
    import torch

    from common.cerberus_eval.nav_metrics import write_run_json
    from common.cerberus_eval.scene_assertions import (
        assert_all,
        build_report,
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

    spawns = spawn_poses_for(args.scene, args.num_envs)
    scene = se.resolve_scene(args.scene)
    proxy = ensure_scan_proxy(scene, spawns[0][3])

    env, scene = se.build_loco_env(
        args.scene, spawns, task=task, episode_length_s=max(120.0, args.steps * 0.02 + 30.0),
        seed=args.seed, scan_proxy=proxy,
    )
    hook = se.VelocityCommandHook(env)
    policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
    device = env.unwrapped.device
    n = env.unwrapped.num_envs

    joint_names = list(env.unwrapped.scene["robot"].joint_names)
    hook.zero()
    obs = env.get_observations()
    obs_row = obs[0].detach().cpu().tolist()
    obs_dim = int(obs.shape[-1])
    command = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    stand = se.stand_metrics(env)
    base_height = None
    if stand is not None:
        base_height = float(stand["base_minus_foot_z"].mean().item())
    scan_start, scan_stop = term_offsets(contract_terms())["height_scan"]
    scan = obs_row[scan_start:scan_stop] if obs_dim == 235 else None
    scan_report = height_scan_report(scan, base_height_m=base_height) if scan else None

    # --- the 200-step zero-command stand -------------------------------------------------------------------
    zero = torch.zeros((n, 3), device=device)
    rows: list[list[float]] = []
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(args.steps):
        hook.apply(zero)
        obs = env.get_observations()
        rows.append(obs[0].detach().cpu().tolist())
        obs, _, _, _ = env.step(policy(obs))
        fell |= se.read_fall_flags(env)
        if bool(fell.any()):
            break
    final_stand = se.stand_metrics(env)
    min_base_minus_foot = None
    if final_stand is not None:
        min_base_minus_foot = float(final_stand["base_minus_foot_z"].min().item())

    report = build_report(
        joint_names=joint_names,
        obs_dim=obs_dim,
        obs_row=obs_row,
        command=command,
        scan=scan,
        base_height_m=base_height,
        stand_rows=rows,
        stand_steps_expected=args.steps,
    )
    if scan_report is not None:
        report.soft["height_scan_flat_ground"] = scan_report
    report.add("no_fall_during_stand", not bool(fell.any()), {"fell_envs": torch.nonzero(fell).flatten().tolist()})
    if min_base_minus_foot is None:
        report.soft["robot_stands"] = "skipped (base/foot body names not found in the articulation)"
    else:
        report.add("robot_stands", min_base_minus_foot > 0.15, {"min_base_minus_foot_z": min_base_minus_foot})
    report.soft["scene"] = {"scene_id": args.scene, "usd_path": scene.get("usd_path"), "num_envs": n,
                            "spawn_poses": spawns, "task": task, "checkpoint": args.checkpoint}
    report.write(out_dir / "assertions.json")
    print(f"[scene_eval] assertions: {'PASS' if report.passed else 'FAIL'} -> {out_dir / 'assertions.json'}")
    for check in report.checks:
        print(f"[scene_eval]   {check}")

    # --- optional scripted path (command hook + policy + video helper) --------------------------------------
    summary: dict = {"assertions_passed": report.passed, "stand_steps": len(rows)}
    if args.scripted and report.passed:
        start = (spawns[0][0], spawns[0][1], spawns[0][2])
        path = scripted_path_for_seed(args.seed, start_pose=start, dt=0.02)
        clip = ClipRecorder(out_dir / "clips", enabled=args.video, downscale=args.video_downscale)
        name = f"{args.scene.split(':', 1)[-1]}_{args.seed}_scripted.mp4"
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
                if args.video:
                    clip.frame(env.unwrapped, 0)
        clip.end()
        elapsed = time.perf_counter() - t0
        poses = se.read_poses(env)
        record = {
            "scene": args.scene,
            "seed": args.seed,
            "outcome": "scripted",
            "steps": steps,
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
    try:
        assert_all(report)
    except AssertionError as e:
        print(f"[scene_eval] FAIL: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = "--dry-run" in argv
    parser = build_parser()
    if not dry:
        from common.cerberus_compat.launcher import add_launcher_args

        add_launcher_args(parser)
    args = parser.parse_args(argv)

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

    # MDL_SYSTEM_PATH must be set BEFORE the app launches; the registry is pure python, so no Isaac import here.
    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        return run(args)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
