#!/usr/bin/env python3
"""Render a camera view from a record-once tour log, without stepping physics.

Sets the robot's root pose and joint positions through the articulation write path each frame, then renders and
captures. This is the "record once, render many" path for the showcase views (CHASE, EGO, close-up): one physics
run, then any camera at any native resolution.

    /isaac-sim/python.sh scripts/replay_render.py --log tour.npz --camera chase --width 960 --height 1080 \
        --scene grscenes:<id> --out outputs/scratch/replay/chase.mp4

Validate a replay once against a live clip: same start pose, then compare a frame (the caller records the pixel
difference); the log must carry `pose_xyzw` and `joint_pos`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="tour npz (pose_xyzw, joint_pos, t)")
    parser.add_argument("--camera", default="chase", choices=["chase", "ego", "closeup"])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    parser.add_argument("--embodiment", default="go2_arx5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pose-out", default=None, help="write a cerberus.poses.v1 log beside the clip")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--near-clip", type=float, default=0.12)
    parser.add_argument("--boom-m", type=float, default=1.4)
    parser.add_argument("--boom-height-m", type=float, default=0.9)
    parser.add_argument("--grid", default=None, help="occupancy.npz for the chase boom (default: beside the scene)")
    parser.add_argument("--scan-proxy", default=None,
                        help="tiny mesh for the (unused) height scanner: avoids loading the 62M-face house proxy")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True   # the capture path needs a render product
    args.headless = True         # no display on the VM; a windowed launch crashes at startup

    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    for _ in range(5):
        simulation_app.update()
    try:
        return render(args)
    except Exception:  # noqa: BLE001 - AppLauncher.close() eats the traceback
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1
    finally:
        simulation_app.close()


def render(args) -> int:
    import imageio
    import torch

    from common.cerberus_description.go2_arx5_spec import GO2_ARX5_BASE_PRIM_REL_PATH
    from common.cerberus_eval import camera_rig as cr
    from planning.occupancy import load_npz
    from policy.locomotion.eval import scene_env as se

    log = np.load(args.log, allow_pickle=True)
    poses = np.asarray(log["pose_xyzw"], dtype=np.float64)   # x, y, z, qx, qy, qz, qw
    joints = np.asarray(log["joint_pos"], dtype=np.float32) if "joint_pos" in log else None
    times = np.asarray(log["t"], dtype=np.float64)
    if len(times) < 2:
        raise SystemExit("log has fewer than 2 steps")
    duration = float(times[-1] - times[0])
    if args.max_seconds:
        keep = times <= times[0] + args.max_seconds
        poses, joints, times = poses[keep], (joints[keep] if joints is not None else None), times[keep]
        duration = float(times[-1] - times[0])

    task = se.task_for_embodiment(args.embodiment)
    spawn = (float(poses[0, 0]), float(poses[0, 1]), 0.0, float(poses[0, 2]))
    env, _scene = se.build_loco_env(args.scene, [spawn], task=task, num_envs=1, seed=0,
                                    episode_length_s=max(60.0, duration + 30.0), with_cameras=False,
                                    scan_proxy=args.scan_proxy)
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    sim = env.unwrapped.sim

    robot_prim = (f"/World/envs/env_0/Robot/{GO2_ARX5_BASE_PRIM_REL_PATH}"
                  if args.embodiment == "go2_arx5" else "/World/envs/env_0/Robot/base")
    if args.camera == "chase":
        spec = cr.CameraSpec("chase", "chase", (args.width, args.height), hfov_deg=70.0,
                             distance=args.boom_m, height=args.boom_height_m, pitch_down_deg=12.0,
                             near_clip_m=args.near_clip)
    elif args.camera == "ego":
        spec = cr.CameraSpec("ego", "ego", (args.width, args.height), hfov_deg=90.0,
                             offset=(0.30, 0.0, 0.08), pitch_down_deg=17.0, near_clip_m=args.near_clip)
    else:
        spec = cr.CameraSpec("closeup", "closeup", (args.width, args.height), hfov_deg=60.0,
                             offset=(0.55, -0.35, 0.75), pitch_down_deg=25.0, near_clip_m=args.near_clip)
    rig = cr.CameraRig(robot_prim_path=robot_prim, specs=[spec])
    print("[replay] env and camera rig ready", flush=True)
    log_pose = cr.PoseLog(camera=args.camera, width=args.width, height=args.height, hfov_deg=spec.hfov_deg)

    occupancy = None
    if args.camera == "chase":
        try:
            if args.grid:
                occupancy = load_npz(args.grid)
            else:
                from scripts.nav_eval import scene_grid

                occupancy = scene_grid(args.scene, None)[0]
        except Exception as exc:  # noqa: BLE001 - the boom pull-in is optional
            print(f"[replay] no grid for the chase boom ({exc}); using the full boom", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, macro_block_size=None)
    print("[replay] writer ready; rendering frames", flush=True)
    frames = max(1, round(duration * args.fps) + 1)
    zeros_vel = torch.zeros((1, 6), device=device)
    steps = 0
    try:
        for k in range(frames):
            t = times[0] + k / float(args.fps)
            idx = int(np.searchsorted(times, t, side="right") - 1)
            idx = max(0, min(idx, len(poses) - 1))
            px, py, pz, qx, qy, qz, qw = poses[idx]
            robot.write_root_pose_to_sim(torch.tensor([[px, py, pz, qw, qx, qy, qz]], dtype=torch.float32,
                                                      device=device))
            robot.write_root_velocity_to_sim(zeros_vel)
            if joints is not None:
                jp = torch.tensor([joints[idx]], dtype=torch.float32, device=device)
                robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            if k == 0:
                print("[replay] first sim.render()", flush=True)
            sim.render()
            if k in (0, 10, 100):
                read = robot.data.root_pos_w[0].detach().cpu().numpy()
                err = float(np.linalg.norm(read - np.array([px, py, pz])))
                print(f"[replay] pose write/read error at frame {k}: {err * 1000:.2f} mm", flush=True)
            rig.update(base_pos=(px, py, pz), base_quat_wxyz=(qw, qx, qy, qz), occupancy=occupancy)
            frame = rig.capture(args.camera)
            writer.append_data(frame)
            cam_pos, cam_quat = rig.world_pose(args.camera)
            log_pose.append(t=float(t), base_pos=(px, py, pz), base_quat_wxyz=(qw, qx, qy, qz),
                            cam_pos=cam_pos, cam_quat_wxyz=cam_quat)
            steps += 1
            if k % max(args.fps, 1) == 0:
                print(f"[replay] {k}/{frames}", flush=True)
    finally:
        writer.close()
        rig.close()

    pose_path = Path(args.pose_out) if args.pose_out else out.with_suffix(".poses.json")
    log_pose.write(pose_path)
    out.with_suffix(".json").write_text(json.dumps({
        "clip": str(out), "poses": str(pose_path), "camera": args.camera, "scene": args.scene,
        "frames": steps, "fps": args.fps, "resolution": [args.width, args.height],
        "source_log": str(args.log), "mode": "replay (root pose + joints written, no physics stepping)",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[replay] wrote {out} ({steps} frames) and {pose_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
