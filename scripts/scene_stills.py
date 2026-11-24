#!/usr/bin/env python3
"""Render stills of the real scene with the Go2 + ARX5 standing in it: the look check, before any clip.

Why stills first: each still is ~20 s of GPU, and the acceptance numbers (mean luminance 70-170, at most 5 percent
clipped, at least 3 hue bins of 30 degrees with 5 percent each) are what tell a textured scene from the flat
red-orange render that made the last demo clip unusable.

Usage (inside the Isaac container)::

    /workspace/isaaclab/isaaclab.sh -p scripts/scene_stills.py --scene grscenes:MV7J... \
        --embodiment go2_arx5 --out-dir ~/cerberus-results/stills --views 5 --resolution 1920x1080

Notes
-----
* MDL materials only resolve when ``MDL_SYSTEM_PATH`` is set *before* the app launches; ``--mdl`` turns that on
  (``CERBERUS_SKIP_MDL_PATH=0``). Without it the scene renders with USD defaults -- the flat colour.
* The five viewpoints are robot-relative (front-left, front-right, behind, side, high wide) so the check does not
  depend on knowing the floor plan; they are logged with their world poses.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--dome-intensity", type=float, default=None,
                        help="override the scene dome light intensity (default: the scene cfg's own value)")
    parser.add_argument("--mdl", action="store_true", help="set MDL_SYSTEM_PATH before the app launches")
    parser.add_argument("--steps", type=int, default=60, help="settle steps before the first still")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn", default="0,0,0", help="x,y,yaw_rad of the robot in the scene (default 0,0,0)")
    parser.add_argument("--chase-distance", type=float, default=1.6)
    parser.add_argument("--chase-height", type=float, default=0.9)
    parser.add_argument("--chase-pitch", type=float, default=12.0)
    parser.add_argument("--ego-offset", default="0.30,0.0,0.08")
    parser.add_argument("--ego-pitch", type=float, default=17.0)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    # AppLauncher-compatible extras
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def inspect_views(base_pos, yaw: float) -> list[tuple[str, tuple[float, float, float, float, float]]]:
    """Five robot-relative viewpoints: (name, (x, y, z, yaw, pitch_down_deg))."""
    import math

    poses = []
    specs = [
        ("front_left", 2.6, math.radians(35), 15.0),
        ("front_right", -2.6, math.radians(-35), 15.0),
        ("behind", 0.0, math.pi, 12.0),
        ("side", 0.0, math.radians(90), 18.0),
    ]
    for name, lateral, angle, pitch in specs:
        dist = 2.8
        x = base_pos[0] + dist * math.cos(yaw + angle) + lateral * 0.0
        y = base_pos[1] + dist * math.sin(yaw + angle)
        look_yaw = math.atan2(base_pos[1] - y, base_pos[0] - x)
        poses.append((name, (x, y, base_pos[2] + 1.4, look_yaw, pitch)))
    poses.append(("high_wide", (base_pos[0] - 1.0, base_pos[1] - 1.0, base_pos[2] + 2.6,
                                math.atan2(base_pos[1] - (base_pos[1] - 1.0), base_pos[0] - (base_pos[0] - 1.0)), 55.0)))
    return poses


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mdl:
        os.environ["CERBERUS_SKIP_MDL_PATH"] = "0"

    from common.cerberus_worlds.registry import get_backend

    out_dir = Path(args.out_dir) if args.out_dir else Path.home() / "cerberus-results" / f"stills-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.resolution.lower().split("x"))

    get_backend("plain_usd").prepare_scene_env(args.scene)
    print(f"[stills] MDL_SYSTEM_PATH={os.environ.get('MDL_SYSTEM_PATH', '')!r} "
          f"skip={os.environ.get('CERBERUS_SKIP_MDL_PATH')}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    for _ in range(5):
        simulation_app.update()

    try:
        rc = render(args, out_dir, width, height)
    except Exception:  # noqa: BLE001 - SimulationApp.close() exits the process with code 0 and eats the traceback
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)  # keep the traceback observable instead of a silent "clean" exit
    simulation_app.close()
    return rc


def render(args, out_dir: Path, width: int, height: int) -> int:
    import numpy as np
    import torch

    from common.cerberus_eval import camera_rig as cr
    from policy.locomotion.eval import scene_env as se

    task = se.task_for_embodiment(args.embodiment)
    x, y, yaw0 = (float(v) for v in args.spawn.split(","))
    spawns = [(x, y, yaw0, 0.0)]
    env, scene = se.build_loco_env(args.scene, spawns, task=task, num_envs=1, episode_length_s=120.0,
                                  seed=args.seed, with_cameras=False)
    if args.dome_intensity is not None:
        light_prim = "/World/skyLight"
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(light_prim)
        if prim and prim.IsValid():
            attr = prim.GetAttribute("inputs:intensity")
            if attr:
                attr.Set(float(args.dome_intensity))
                print(f"[stills] dome intensity -> {args.dome_intensity}", flush=True)

    policy = se.load_policy(env, task=task, checkpoint=args.checkpoint)
    zero = torch.zeros((1, 3), device=env.unwrapped.device)
    hook = se.VelocityCommandHook(env)
    for _ in range(max(args.steps, 1)):
        hook.apply(zero)
        env.step(policy(se.fresh_observations(env)))

    robot = env.unwrapped.scene["robot"]
    base_pose = robot.data.root_pos_w[0].detach().cpu().numpy()
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()  # (w, x, y, z) in Isaac Lab
    yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                           1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
    print(f"[stills] robot at {base_pose.tolist()} yaw {yaw:.3f}", flush=True)

    # camera prims: the base-link prim path is the spec constant under the robot prim
    from common.cerberus_description.go2_arx5_spec import GO2_ARX5_BASE_PRIM_REL_PATH
    robot_prim = f"/World/envs/env_0/Robot/{GO2_ARX5_BASE_PRIM_REL_PATH}" if args.embodiment == "go2_arx5" \
        else "/World/envs/env_0/Robot/base"

    specs = [
        cr.CameraSpec("ego", "ego", (width, height), hfov_deg=90.0,
                      offset=tuple(float(v) for v in args.ego_offset.split(",")),
                      pitch_down_deg=args.ego_pitch),
        cr.CameraSpec("chase", "chase", (width, height), hfov_deg=70.0, distance=args.chase_distance,
                      height=args.chase_height, pitch_down_deg=args.chase_pitch),
    ]
    for name, pose in inspect_views(base_pose, yaw)[: max(args.views, 0)]:
        specs.append(cr.CameraSpec(name, "overview", (width, height), hfov_deg=80.0, world_pose=pose))

    rig = cr.CameraRig(robot_prim_path=robot_prim, specs=specs)
    sim = env.unwrapped.sim
    results = []

    settings = {"scene": args.scene, "embodiment": args.embodiment, "resolution": [width, height],
                "dome_intensity": args.dome_intensity, "mdl": bool(args.mdl),
                "mdl_system_path": os.environ.get("MDL_SYSTEM_PATH", "")}

    for spec in specs:
        robot_state = env.unwrapped.scene["robot"].data
        q = [float(v) for v in robot_state.root_quat_w[0].detach().cpu().tolist()]
        # Isaac Lab 3.0 stores root_quat_w as XYZW; the rig wants WXYZ
        rig.update(base_pos=robot_state.root_pos_w[0].detach().cpu().numpy(),
                   base_quat_wxyz=(q[3], q[0], q[1], q[2]))
        sim.render()
        frame = rig.capture(spec.name)
        print(f"[stills] {spec.name} frame {frame.shape} {frame.dtype}", flush=True)
        png = out_dir / f"still_{spec.name}.png"
        # imageio, not PIL: the PIL shipped inside Sim 6.2 fails on these frames with
        # `AttributeError: '_idat' object has no attribute 'fileno'` (measured 2026-11-23).
        import imageio

        imageio.imwrite(str(png), np.ascontiguousarray(frame))
        cam_pos, cam_quat = rig.world_pose(spec.name)
        results.append({"camera": spec.name, "kind": spec.kind, "png": str(png),
                        "cam_pos": cam_pos.tolist(), "cam_quat_wxyz": list(cam_quat),
                        "base_pos": base_pose.tolist()})
        print(f"[stills] wrote {png} ({frame.shape[1]}x{frame.shape[0]})", flush=True)

    from common.cerberus_eval.atomic_io import write_json_atomic
    write_json_atomic(out_dir / "stills.json", {"settings": settings, "views": results,
                                                "robot_prim": robot_prim,
                                                "task": task})
    rig.close()
    print(f"[stills] done: {len(results)} stills in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
