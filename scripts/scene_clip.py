#!/usr/bin/env python3
"""Record one camera clip of the real scene with the Go2 + ARX5 (EGO, CHASE, OVERVIEW or a robot close-up).

Every frame writes the base and camera world poses next to the video (``*.poses.json``), so
``scripts/host/clip_qa.py`` can prove the camera is rigidly attached (EGO), yaw-only and behind the robot (CHASE), or
fixed (OVERVIEW) instead of trusting the label.

Two motions are scripted here (no tuning, no learning):
* ``--walk``  -- a seeded open-loop velocity path (the same scripted path the scene demo uses);
* ``--arm-demo`` -- the arm's own joint targets: reach forward, wrist turn, gripper open and close, back to stowed.

Usage (inside the Isaac container)::

    /workspace/isaaclab/isaaclab.sh -p scripts/scene_clip.py --camera chase --seconds 20 \
        --out-dir ~/cerberus-results/clips --scene grscenes:MV7J... --embodiment go2_arx5 --arm-demo
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    parser.add_argument("--embodiment", default="go2_arx5", choices=["go2", "go2_arx5"])
    parser.add_argument("--checkpoint", default="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt")
    parser.add_argument("--camera", default="chase", choices=["ego", "chase", "overview", "closeup"])
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--name", default=None, help="clip base name (default <camera>_<embodiment>)")
    parser.add_argument("--walk", action="store_true", help="drive the scripted velocity path")
    parser.add_argument("--arm-demo", action="store_true", help="play the arm reach/turn/grip script")
    parser.add_argument("--label", default="", help="label written into the clip JSON (e.g. policy not yet retrained)")
    parser.add_argument("--start-state", default=None,
                        help="resume from this state JSON (a segment's end state from an earlier launch)")
    parser.add_argument("--end-state", default=None,
                        help="write the robot state here when the segment finishes (needed to stitch segments)")
    parser.add_argument("--start-step", type=int, default=0,
                        help="global step offset: keeps the scripted path and the arm clock continuous across segments")
    parser.add_argument("--mdl", action="store_true")
    parser.add_argument("--replay-log", default=None,
                        help="npz from a tour run: set root pose/joints per frame and render (no physics, no policy)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn", default="0,0,0", help="x,y,yaw_rad of the robot in the scene (default 0,0,0)")
    parser.add_argument("--chase-distance", type=float, default=1.6)
    parser.add_argument("--chase-height", type=float, default=0.9)
    parser.add_argument("--chase-pitch", type=float, default=12.0)
    parser.add_argument("--ego-offset", default="0.30,0.0,0.08")
    parser.add_argument("--ego-pitch", type=float, default=17.0)
    parser.add_argument("--exposure-stops", type=float, default=None,
                        help="RTX tonemapper exposure compensation in stops (one setting for every camera)")
    parser.add_argument("--dome-intensity", type=float, default=None,
                        help="override the env dome light intensity (the scene carries its own lights)")
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


# ---- scripted arm motion ---------------------------------------------------------------------------------------

def arm_demo_targets(t: float, *, stowed: dict[str, float], span: float = 4.0) -> dict[str, float]:
    """Arm joint targets over time (rad / m): reach forward, wrist turn, grip, return to stowed.

    Pure and deterministic; the same function is unit-tested off-pod. Phases (seconds, cycling over ``span``):
    0.0-0.15 stow, 0.15-0.45 reach+turn, 0.45-0.55 close, 0.55-0.7 open, 0.7-1.0 return.
    """
    phase = (t % span) / span
    target = dict(stowed)
    if phase < 0.15 or phase >= 0.7:
        return target
    if phase < 0.45:
        a = (phase - 0.15) / 0.30
        ease = 0.5 - 0.5 * math.cos(math.pi * a)
        target["joint1"] = stowed["joint1"] + 0.45 * ease     # shoulder swing forward
        target["joint2"] = stowed["joint2"] - 0.60 * ease     # lift
        target["joint3"] = stowed["joint3"] + 0.35 * ease
        target["joint5"] = stowed["joint5"] + 0.8 * ease      # wrist turn
        return target
    if phase < 0.55:  # close
        target["joint1"] = stowed["joint1"] + 0.45
        target["joint2"] = stowed["joint2"] - 0.60
        target["joint3"] = stowed["joint3"] + 0.35
        target["joint5"] = stowed["joint5"] + 0.8
        target["joint7"] = 0.0
        target["joint8"] = 0.0
        return target
    if phase < 0.7:  # open again
        target["joint1"] = stowed["joint1"] + 0.45
        target["joint2"] = stowed["joint2"] - 0.60
        target["joint3"] = stowed["joint3"] + 0.35
        target["joint5"] = stowed["joint5"] + 0.8
        target["joint7"] = stowed["joint7"]
        target["joint8"] = stowed["joint8"]
        return target
    return target


# ---- segment state (crash-tolerant rendering) -------------------------------------------------------------------

def capture_state(robot) -> dict:
    """The robot state a later launch needs to continue exactly: root pose/velocity and every joint position/velocity."""
    return {
        "root_pos_w": [float(v) for v in robot.data.root_pos_w[0].detach().cpu().tolist()],
        "root_quat_wxyz": [float(v) for v in robot.data.root_quat_w[0].detach().cpu().tolist()],
        "root_lin_vel_w": [float(v) for v in robot.data.root_lin_vel_w[0].detach().cpu().tolist()],
        "root_ang_vel_w": [float(v) for v in robot.data.root_ang_vel_w[0].detach().cpu().tolist()],
        "joint_pos": [float(v) for v in robot.data.joint_pos[0].detach().cpu().tolist()],
        "joint_vel": [float(v) for v in robot.data.joint_vel[0].detach().cpu().tolist()],
    }


def load_state(path: str | Path) -> dict:
    import json

    data = json.loads(Path(path).read_text())
    for key in ("root_pos_w", "root_quat_wxyz", "joint_pos", "joint_vel"):
        if key not in data:
            raise ValueError(f"state file {path} has no {key}")
    return data


def apply_state(robot, state: dict, device) -> None:
    """Write a captured state into the articulation (pose, velocity, joints) so the segment continues where it stopped."""
    import torch

    def t(values):
        return torch.tensor([values], device=device, dtype=torch.float32)

    root_pose = t(state["root_pos_w"] + state["root_quat_wxyz"])
    robot.write_root_pose_to_sim(root_pose)
    if "root_lin_vel_w" in state and "root_ang_vel_w" in state:
        robot.write_root_velocity_to_sim(t(state["root_lin_vel_w"] + state["root_ang_vel_w"]))
    robot.write_joint_state_to_sim(t(state["joint_pos"]), t(state["joint_vel"]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mdl:
        os.environ["CERBERUS_SKIP_MDL_PATH"] = "0"

    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    for _ in range(5):
        simulation_app.update()
    try:
        rc = record(args)
    except Exception:  # noqa: BLE001 - SimulationApp.close() exits the process with code 0 and eats the traceback
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)  # keep the traceback observable instead of a silent "clean" exit
    simulation_app.close()
    return rc


def record(args) -> int:
    import numpy as np
    import torch

    from common.cerberus_description.go2_arx5_spec import (
        ARX5_ARM_JOINTS,
        ARX5_STOWED_JOINT_POS,
        GO2_ARX5_BASE_PRIM_REL_PATH,
        GRIPPER_JOINT_NAMES,
    )
    from common.cerberus_eval import camera_rig as cr
    from common.cerberus_eval.atomic_io import write_json_atomic
    from common.cerberus_eval.scripted_path import scripted_path_for_seed
    from policy.locomotion.eval import scene_env as se

    out_dir = Path(args.out_dir) if args.out_dir else Path.home() / "cerberus-results" / f"clips-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    name = args.name or f"{args.camera}_{args.embodiment}"
    clip_path = out_dir / f"{name}.mp4"
    poses_path = out_dir / f"{name}.poses.json"

    task = se.task_for_embodiment(args.embodiment)
    x, y, yaw0 = (float(v) for v in args.spawn.split(","))
    env, scene = se.build_loco_env(args.scene, [(x, y, yaw0, 0.0)], task=task, num_envs=1,
                                   episode_length_s=max(120.0, args.seconds + 30.0), seed=args.seed,
                                   with_cameras=False)
    if args.exposure_stops is not None:
        import carb.settings

        settings = carb.settings.get_settings()
        for key in ("/rtx/post/tonemap/exposure", "/rtx/post/tonemap/exposureCompensation"):
            try:
                settings.set_float(key, float(args.exposure_stops))
            except Exception:  # noqa: BLE001 - a stack without that key simply ignores it
                pass
        print(f"[exposure] stops -> {args.exposure_stops}", flush=True)
    if args.dome_intensity is not None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/skyLight")
        if prim and prim.IsValid():
            attr = prim.GetAttribute("inputs:intensity")
            if attr:
                attr.Set(float(args.dome_intensity))
                print(f"[clip] dome intensity -> {args.dome_intensity}", flush=True)

    policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
    hook = se.VelocityCommandHook(env)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    # settle, so the clip does not open on the spawn pose; a resumed segment starts from the saved state instead
    zero = torch.zeros((1, 3), device=device)
    start_state = load_state(args.start_state) if args.start_state else None
    if start_state is None:
        for _ in range(90):
            hook.apply(zero)
            env.step(policy(se.fresh_observations(env)))
    else:
        apply_state(robot, start_state, device)

    robot_prim = (f"/World/envs/env_0/Robot/{GO2_ARX5_BASE_PRIM_REL_PATH}" if args.embodiment == "go2_arx5"
                  else "/World/envs/env_0/Robot/base")

    base_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))

    if args.camera == "ego":
        ego_offset = tuple(float(v) for v in args.ego_offset.split(","))
        spec = cr.CameraSpec("ego", "ego", (width, height), hfov_deg=90.0, offset=ego_offset,
                             pitch_down_deg=args.ego_pitch)
    elif args.camera == "chase":
        spec = cr.CameraSpec("chase", "chase", (width, height), hfov_deg=70.0, distance=args.chase_distance,
                             height=args.chase_height, pitch_down_deg=args.chase_pitch, near_clip_m=0.12)
    elif args.camera == "closeup":
        # to the robot's front-right at arm height, looking at the arm's workspace
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        eye = base_pos + 1.1 * fwd + 0.7 * side + np.array([0.0, 0.0, 0.75])
        look_yaw = math.atan2(base_pos[1] - eye[1], base_pos[0] - eye[0])
        spec = cr.CameraSpec("closeup", "overview", (width, height), hfov_deg=60.0,
                             world_pose=(eye[0], eye[1], eye[2], look_yaw, 18.0))
    else:
        look_yaw = math.atan2(base_pos[1] - (base_pos[1] - 2.5), base_pos[0] - (base_pos[0] - 2.5))
        spec = cr.CameraSpec("overview", "overview", (width, height), hfov_deg=80.0,
                             world_pose=(base_pos[0] - 2.5, base_pos[1] - 2.5, base_pos[2] + 2.2, look_yaw, 35.0))

    rig = cr.CameraRig(robot_prim_path=robot_prim, specs=[spec])
    log = cr.pose_log_for(spec)

    arm_ids = None
    stowed = dict(ARX5_STOWED_JOINT_POS)
    if args.arm_demo and args.embodiment == "go2_arx5":
        names = list(ARX5_ARM_JOINTS) + list(GRIPPER_JOINT_NAMES)
        arm_ids, _ = robot.find_joints(names, preserve_order=True)
        arm_ids = torch.tensor(arm_ids, device=device)

    path = scripted_path_for_seed(args.seed) if args.walk else None
    steps = int(args.seconds * args.fps)
    import imageio

    writer = imageio.get_writer(str(clip_path), fps=args.fps, macro_block_size=None)
    sim = env.unwrapped.sim
    if args.replay_log:
        import numpy as _np

        data = _np.load(args.replay_log, allow_pickle=True)
        poses, times = _np.asarray(data["pose_xyzw"], dtype=float), _np.asarray(data["t"], dtype=float)
        occupancy = None
        try:
            from planning.occupancy import load_npz

            usd = Path(scene.get("usd_path", ""))
            occupancy = load_npz(usd.parent / "occupancy.npz")
            print(f"[clip] replay occupancy: {usd.parent / 'occupancy.npz'}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the boom stays at full length
            print(f"[clip] replay has no grid for the boom ({type(exc).__name__}); full boom", flush=True)
        joints = _np.asarray(data["joint_pos"], dtype=_np.float32) if "joint_pos" in data else None
        zero6 = torch.zeros((1, 6), device=device)
        n_frames = max(1, round(float(times[-1] - times[0]) * args.fps) + 1)
        for k in range(n_frames):
            t = times[0] + k / float(args.fps)
            idx = max(0, min(int(_np.searchsorted(times, t, side="right") - 1), len(poses) - 1))
            px, py, pz, qx, qy, qz, qw = poses[idx]
            robot.write_root_pose_to_sim(torch.tensor([[px, py, pz, qx, qy, qz, qw]], dtype=torch.float32,
                                                       device=device))
            robot.write_root_velocity_to_sim(zero6)
            if joints is not None:
                jp = torch.tensor([joints[idx]], dtype=torch.float32, device=device)
                if jp.shape[1] == robot.num_joints:
                    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            sim.render()
            # low_pass=1.0 keeps the chase offset exactly constant in the base frame (the gate checks that)
            rig.update(base_pos=(px, py, pz), base_quat_wxyz=(qw, qx, qy, qz), low_pass=1.0,
                       occupancy=occupancy)
            writer.append_data(rig.capture(spec.name))
            cam_pos, cam_quat = rig.world_pose(spec.name)
            log.append(t=float(t), base_pos=(px, py, pz), base_quat_wxyz=(qw, qx, qy, qz),
                       cam_pos=cam_pos, cam_quat_wxyz=cam_quat)
            if k % max(args.fps, 1) == 0:
                print(f"[clip] replay {k}/{n_frames}", flush=True)
        writer.close()
        rig.close()
        log.write(poses_path)
        write_json_atomic(out_dir / f"{name}.json", {
            "clip": str(clip_path), "poses": str(poses_path), "camera": args.camera,
            "mode": "replay (root pose + joints, no physics)", "source_log": str(args.replay_log),
            "frames": n_frames, "fps": args.fps, "resolution": [width, height],
        })
        print(f"[clip] wrote {clip_path} and {poses_path}", flush=True)
        return 0
    started = time.perf_counter()
    try:
        for step in range(steps):
            global_step = args.start_step + step
            sim_t = global_step / float(args.fps)
            if path is not None:
                idx = min(int(global_step * 1.0), len(path.commands) - 1)
                entry = path.commands[idx]
                # commands are dicts {vx, vy, wz} (the contract's velocity triple)
                values = ([entry.get("vx", 0.0), entry.get("vy", 0.0), entry.get("wz", 0.0)]
                          if isinstance(entry, dict) else list(entry))
                cmd = torch.tensor([values], device=device, dtype=torch.float32)
            else:
                cmd = zero
            hook.apply(cmd)
            if arm_ids is not None:
                targets = arm_demo_targets(sim_t, stowed=stowed)
                values = [targets[j] for j in list(ARX5_ARM_JOINTS) + list(GRIPPER_JOINT_NAMES)]
                robot.set_joint_position_target(torch.tensor([values], device=device), joint_ids=arm_ids)
            env.step(policy(se.fresh_observations(env)))
            # the robots live in Fabric: read the physics state, never the (stale) USD xform
            bpos = robot.data.root_pos_w[0].detach().cpu().numpy()
            bquat = robot.data.root_quat_w[0].detach().cpu().numpy()
            # Isaac Lab 3.0 stores root_quat_w as XYZW; the rig wants WXYZ
            q = [float(v) for v in bquat]
            rig.update(base_pos=bpos, base_quat_wxyz=(q[3], q[0], q[1], q[2]))
            frame = rig.capture(spec.name)
            writer.append_data(frame)
            cam_pos, cam_quat = rig.world_pose(spec.name)
            log.append(t=sim_t, base_pos=bpos, base_quat_wxyz=(q[3], q[0], q[1], q[2]),
                       cam_pos=cam_pos, cam_quat_wxyz=cam_quat)
            if step % max(args.fps, 1) == 0:
                print(f"[clip] {step}/{steps} t={sim_t:.1f}s fps={step / max(sim_t, 1e-6):.1f}", flush=True)
    finally:
        writer.close()
        rig.close()

    log.write(poses_path)
    if args.end_state:
        write_json_atomic(Path(args.end_state), capture_state(robot))
    write_json_atomic(out_dir / f"{name}.json", {
        "clip": str(clip_path), "poses": str(poses_path), "camera": args.camera, "kind": spec.kind,
        "scene": args.scene, "embodiment": args.embodiment, "checkpoint": args.checkpoint,
        "seconds": args.seconds, "fps": args.fps, "resolution": [width, height],
        "walk": bool(args.walk), "arm_demo": bool(args.arm_demo), "label": args.label,
        "mdl_system_path": os.environ.get("MDL_SYSTEM_PATH", ""),
        "wall_clock_s": round(time.perf_counter() - started, 1),
        "note": "camera poses are simulator ground truth logged per frame; motion is scripted or policy-driven "
                "as stated by walk/arm_demo",
    })
    print(f"[clip] wrote {clip_path} and {poses_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
