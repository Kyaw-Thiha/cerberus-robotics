#!/usr/bin/env python3
"""Scripted arm grasp in the real scene: table + cube spawn, IK to the cube, close, lift, and a close-up clip.

What it is, and what it is not:

* the **legs** stand still (zero command) under the frozen/served locomotion checkpoint;
* the **arm** follows IK joint targets computed by ``planning.skills`` in the arm's own frame -- scripted, not
  learned, not tuned;
* if the physics gripper does not hold the cube (it usually does not without contact tuning), the cube is **attached
  kinematically** to the gripper after the close, and every output says ``magic_grasp: true``. The handoff's rule is
  explicit: try the physical grasp first, label the fallback.

Usage (inside the Isaac container)::

    /workspace/isaaclab/isaaclab.sh -p scripts/scene_arm_grasp.py --scene grscenes:MV7J... \
        --out-dir ~/cerberus-results/grasp --seconds 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    parser.add_argument("--checkpoint", default="policy/locomotion/checkpoints/go2_arx5_locomotion_ga_v1_candidate.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--mdl", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mdl:
        os.environ["CERBERUS_SKIP_MDL_PATH"] = "0"
    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    app = launcher.app
    for _ in range(5):
        app.update()
    try:
        return run(args)
    finally:
        app.close()


def run(args) -> int:
    import imageio
    import torch
    from isaaclab.assets import RigidObject
    from isaaclab.sim import RigidBodyPropertiesCfg, RigidObjectCfg

    from common.cerberus_description.go2_arx5_spec import (
        ARX5_ARM_JOINTS,
        GO2_ARX5_BASE_PRIM_REL_PATH,
        GRIPPER_JOINT_NAMES,
    )
    from common.cerberus_eval import camera_rig as cr
    from common.cerberus_eval.atomic_io import write_json_atomic
    from planning.skills import ArmModel, differential_ik, pick_waypoints
    from policy.locomotion.eval import scene_env as se

    out = Path(args.out_dir) if args.out_dir else Path.home() / "cerberus-results" / "grasp"
    out.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.resolution.lower().split("x"))

    task = se.task_for_embodiment("go2_arx5")
    env, scene = se.build_loco_env(args.scene, [(0.0, 0.0, 0.0, 0.0)], task=task, num_envs=1,
                                   episode_length_s=180.0, seed=args.seed, with_cameras=False)
    policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
    hook = se.VelocityCommandHook(env)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    zero = torch.zeros((1, 3), device=device)
    for _ in range(90):
        hook.apply(zero)
        env.step(policy(se.fresh_observations(env)))

    base = robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])

    # ---- props: a tabletop slab and a 4 cm cube on it, in front of the robot
    table_top = 0.35
    slab_centre = base + 0.55 * fwd
    slab_centre[2] = table_top - 0.02
    cube_centre = base + 0.50 * fwd
    cube_centre[2] = table_top + 0.02
    from isaaclab.sim import sim_utils

    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_0/props/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.04),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in slab_centre)),
    )
    cube_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_0/props/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in cube_centre)),
    )
    table = RigidObject(table_cfg)
    cube = RigidObject(cube_cfg)
    env.unwrapped.sim.reset()
    print(f"[grasp] table at {slab_centre.round(3).tolist()} cube at {cube_centre.round(3).tolist()}", flush=True)

    # ---- cameras: chase and a close-up of the arm
    robot_prim = f"/World/envs/env_0/Robot/{GO2_ARX5_BASE_PRIM_REL_PATH}"
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    eye = base + 1.0 * fwd + 0.8 * side + np.array([0.0, 0.0, 0.85])
    look_yaw = math.atan2(cube_centre[1] - eye[1], cube_centre[0] - eye[0])
    specs = [
        cr.CameraSpec("chase", "chase", (width, height), hfov_deg=70.0, distance=1.8, height=1.0,
                      pitch_down_deg=14.0),
        cr.CameraSpec("closeup", "overview", (width, height), hfov_deg=60.0,
                      world_pose=(eye[0], eye[1], eye[2], look_yaw, 20.0)),
    ]
    rig = cr.CameraRig(robot_prim_path=robot_prim, specs=specs)
    logs = {spec.name: cr.pose_log_for(spec) for spec in specs}
    writers = {}
    for name in logs:
        writers[name] = imageio.get_writer(str(out / f"grasp_{name}.mp4"), fps=args.fps, macro_block_size=None)

    # ---- scripted pick in the arm frame
    arm_ids, _ = robot.find_joints(list(ARX5_ARM_JOINTS) + list(GRIPPER_JOINT_NAMES), preserve_order=True)
    arm_ids = torch.tensor(arm_ids, device=device)
    arm_model = ArmModel()
    cube_local = cube_centre - np.array([base[0], base[1], 0.0])
    plan = pick_waypoints(cube_local[:2], table_top)
    q_stow = np.array([0.0, 0.6, -1.0])
    q_pre = differential_ik(arm_model, plan["pre_grasp"], q_stow)["q"]
    q_grasp = differential_ik(arm_model, plan["descend"], q_pre)["q"]

    def set_arm(q, grip):
        values = [float(v) for v in q] + [0.0, 0.0, 0.0] + [grip, grip]
        robot.set_joint_position_target(torch.tensor([values[:len(arm_ids)]], device=device), joint_ids=arm_ids)

    phases = [("stow", q_stow, 0.044, 1.0), ("reach", q_pre, 0.044, 1.0), ("descend", q_grasp, 0.044, 1.0),
              ("close", q_grasp, 0.0, 0.8), ("lift", q_pre, 0.0, 1.5)]
    steps = int(args.seconds * args.fps)
    phase_index = 0
    phase_t = 0.0
    magic = False
    moved_with_gripper = 0
    sim = env.unwrapped.sim
    t0 = time.perf_counter()
    try:
        for step in range(steps):
            if phase_index < len(phases):
                name, q, grip, duration = phases[phase_index]
                set_arm(q, grip)
                phase_t += 1.0 / args.fps
                if phase_t >= duration:
                    phase_index += 1
                    phase_t = 0.0
                if name == "close" and phase_t >= duration - 0.02:
                    tcp = arm_model.fk(q) + np.array([base[0], base[1], 0.0])
                    dist = float(np.linalg.norm(tcp - cube.data.root_pos_w[0].detach().cpu().numpy()))
                    print(f"[grasp] gripper closed, TCP-cube distance {dist:.3f} m", flush=True)
                    if dist > 0.03:
                        magic = True  # physics hold is not going to happen; label and attach kinematically
            hook.apply(zero)
            env.step(policy(se.fresh_observations(env)))
            if magic and phase_index >= 4:
                # kinematic attach, labelled everywhere: the cube follows the gripper
                q_now = q_pre
                tcp = arm_model.fk(q_now) + np.array([base[0], base[1], 0.0])
                cube.write_root_pose_to_sim(torch.tensor([[tcp[0], tcp[1], tcp[2], 1.0, 0.0, 0.0, 0.0]],
                                                         device=device))
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
                moved_with_gripper += 1
            rig.update()
            sim.render()
            for name in rig.specs:
                frame = rig.capture(name)
                writers[name].append_data(frame)
                cam_pos, cam_quat = rig.world_pose(name)
                bpos = robot.data.root_pos_w[0].detach().cpu().numpy()
                bquat = robot.data.root_quat_w[0].detach().cpu().numpy()
                logs[name].append(t=step / args.fps, base_pos=bpos, base_quat_wxyz=tuple(float(v) for v in bquat),
                                  cam_pos=cam_pos, cam_quat_wxyz=cam_quat)
    finally:
        for writer in writers.values():
            writer.close()
        rig.close()

    cube_final = cube.data.root_pos_w[0].detach().cpu().numpy()
    lifted = bool(cube_final[2] > table_top + 0.06)
    for name, log in logs.items():
        log.write(out / f"grasp_{name}.poses.json")
    result = {
        "clip_chase": str(out / "grasp_chase.mp4"), "clip_closeup": str(out / "grasp_closeup.mp4"),
        "magic_grasp": magic, "lifted": lifted, "cube_start": cube_centre.tolist(),
        "cube_end": cube_final.tolist(), "table_top_z": table_top,
        "phases": [p[0] for p in phases], "moved_with_gripper_steps": moved_with_gripper,
        "checkpoint": args.checkpoint, "scene": args.scene, "resolution": [width, height], "fps": args.fps,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "label": ("scripted IK arm motion; " + ("MAGIC GRASP (kinematic attach)" if magic else "physical grasp")),
        "mdl_system_path": os.environ.get("MDL_SYSTEM_PATH", ""),
    }
    write_json_atomic(out / "grasp_result.json", result)
    print("[grasp] " + json.dumps({k: result[k] for k in ("lifted", "magic_grasp", "moved_with_gripper_steps")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
