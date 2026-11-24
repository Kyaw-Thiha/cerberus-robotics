#!/usr/bin/env python3
"""JEPLO navigation in our scene: the released checkpoint driven by OUR LiDAR simulation, through the tour loop.

The policy is a B1 consumer: pure pursuit emits a LocoCommand, the adapter builds the 45-dim proprioception plus the
25x60 depth image from the scene's ray caster, and the raw 12 actions go to the env's joint-position term (the env
adds the default pose and the 0.25 scale, exactly as JEPLO's training stack did).

    /isaac-sim/python.sh scripts/jeplo_nav.py --headless --scene grscenes:<id> --start=-3.75,0.73 --seed 0 \
        --models third_party/jeplo/training/logs/rsl_rl/go2_loco/pretrained/exported \
        --out outputs/scratch/jeplo/nav_seed0

Same seeds, goals, planner and controller as scripts/explore_tour.py (PPO), so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cerberus_compat.quat import projected_gravity_from_quat
from planning.local_planner.core.pure_pursuit import PurePursuitConfig
from planning.tour import (
    TourConfig,
    pick_tour_goals,
    run_tour,
    tour_npz_payload,
)

OFFSET_POS = (0.275, 0.0, 0.148)
OFFSET_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="grscenes:MV7J6NIKTKJZ2AABAAAAADQ8_usd")
    parser.add_argument("--start", required=True, help="x,y[,yaw] spawn")
    parser.add_argument("--goals", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-tolerance", type=float, default=0.3)
    parser.add_argument("--min-net", type=float, default=6.0)
    parser.add_argument("--max-total-time", type=float, default=180.0)
    parser.add_argument("--models", default="third_party/jeplo/training/logs/rsl_rl/go2_loco/pretrained/exported")
    parser.add_argument("--embodiment", default="go2", choices=["go2", "go2_arx5"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--settle-steps", type=int, default=90)
    parser.add_argument("--debug", action="store_true", help="print actions/depth/pose every 50 steps")
    parser.add_argument("--goals-xy", default=None,
                        help="explicit goals 'x,y;x,y' (overrides the seeded goal picker)")
    parser.add_argument("--record-dir", default=None,
                        help="also record CHASE and EGO halves (960x1080, 30 fps) during the live run")
    return parser


class JeploIsaacBackend:
    """TourBackend: commands in, JEPLO actions into the Isaac env, with the scene's LiDAR as the only perception."""

    def __init__(self, env, scene_env_module, sensor_name: str = "jeplo_lidar") -> None:
        from isaaclab.utils import (
            math as lab_math,  # noqa: F401  (documented dependency of the env)
        )

        from plugins.jeplo.backend import JeploBackend
        from plugins.jeplo.sensor import SphericalDepth

        self.env = env
        self.se = scene_env_module
        self.robot = env.unwrapped.scene["robot"]
        self.sensor = env.unwrapped.scene[sensor_name]
        self.policy = JeploBackend()
        self.policy.load(os.environ.get("JEPLO_MODELS"))
        # the ray caster refreshes at 10 Hz (one frame per 5 policy steps); two frames = the 0.2 s window JEPLO trained on
        self.depth = SphericalDepth(history=2, update_every=1, seed=0)
        self.num_envs = 1
        self._depth_latest = np.ones((25, 60), dtype=np.float32)
        self._depth_rows: list[np.ndarray] = []
        self._joint_names = list(self.robot.joint_names) if hasattr(self.robot, "joint_names") else []
        self.occupancy = None
        self.record_dir = None
        self._rig = None
        self._writers = {}
        self._frame_accum = 0.0

    def start_recording(self, record_dir, occupancy, base_prim: str = "/World/envs/env_0/Robot/base") -> None:
        import imageio

        from common.cerberus_eval import camera_rig as cr

        self.occupancy = occupancy
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        base = base_prim
        specs = [
            cr.CameraSpec("chase", "chase", (960, 1080), hfov_deg=70.0, distance=1.4, height=0.9,
                          pitch_down_deg=12.0, near_clip_m=0.12),
            cr.CameraSpec("ego", "ego", (960, 1080), hfov_deg=90.0, offset=(0.30, 0.0, 0.08),
                          pitch_down_deg=17.0, near_clip_m=0.12),
        ]
        self._rig = cr.CameraRig(robot_prim_path=base, specs=specs)
        self._writers = {name: imageio.get_writer(str(self.record_dir / f"{name}.mp4"), fps=30,
                                                  macro_block_size=None) for name in ("chase", "ego")}
        print("[jeplo] recording started after settle", flush=True)

    def _capture(self) -> None:
        if self._rig is None:
            return

        self._frame_accum += 50.0 / 30.0
        if self._frame_accum < 1.0:
            return
        self._frame_accum -= 1.0
        data = self.robot.data
        pos = [float(v) for v in data.root_pos_w[0].detach().cpu()]
        q = [float(v) for v in data.root_quat_w[0].detach().cpu()]
        self.env.unwrapped.sim.render()
        self._rig.update(base_pos=pos, base_quat_wxyz=(q[3], q[0], q[1], q[2]), occupancy=self.occupancy)
        for name, writer in self._writers.items():
            writer.append_data(self._rig.capture(name))

    # -- TourBackend ---------------------------------------------------------------------------
    def reset(self, poses) -> None:
        """Teleport env 0 (nav eval episodes) and clear the policy's history, hidden state and depth buffer."""
        self.se.reset_envs_to(self.env, list(poses))
        self.policy.reset()
        self.depth.reset()
        self._depth_latest = np.ones((25, 60), dtype=np.float32)
        self._depth_rows = []

    def step(self, commands) -> None:
        import torch

        command = commands[0]
        self.policy.set_command(command, dt=0.02)
        actions = self.policy.act(self.observation(command))["actions"]
        tensor = torch.tensor([actions], device=self.env.unwrapped.device, dtype=torch.float32)
        self.env.step(tensor)
        self._record_depth()
        self._capture()
        self._steps = getattr(self, "_steps", 0) + 1
        if getattr(self, "debug", False) and self._steps % 50 == 0:
            data = self.robot.data
            pos = [float(v) for v in data.root_pos_w[0].detach().cpu()]
            print(f"[jeplo] step {self._steps} pos=({pos[0]:.2f},{pos[1]:.2f}) cmd={command} "
                  f"|a|={float(np.abs(actions).mean()):.2f} depth_mean={float(self._depth_latest.mean()):.2f}",
                  flush=True)

    def poses(self):
        return [tuple(float(v) for v in row) for row in self.se.read_poses(self.env).detach().cpu()]

    def falls(self):
        return [bool(v) for v in self.se.read_fall_flags(self.env).detach().cpu()]

    def observation(self, command) -> dict:
        data = self.robot.data
        q_xyzw = [float(v) for v in data.root_quat_w[0].detach().cpu()]
        q_wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
        ang = getattr(data, "root_com_ang_vel_b", data.root_ang_vel_b)[0].detach().cpu().numpy()
        joint_pos = [float(v) for v in data.joint_pos[0].detach().cpu()]
        joint_vel = [float(v) for v in data.joint_vel[0].detach().cpu()]
        pos = [float(v) for v in data.root_pos_w[0].detach().cpu()]
        hits = self.sensor.data.ray_hits_w[0].detach().cpu().numpy()
        sensor_pos, sensor_quat = sensor_pose_w(pos, q_wxyz)
        published = self.depth.update_rays(hits, sensor_pos)
        if published is not None:
            self._depth_latest = published
        names = self._joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        return {
            "base_ang_vel": ang,
            "projected_gravity": projected_gravity_from_quat(q_xyzw),
            "joint_pos": dict(zip(names, joint_pos)),
            "joint_vel": dict(zip(names, joint_vel)),
            "command": command,
            "depth": self._depth_latest,
        }

    def _record_depth(self) -> None:
        self._depth_rows.append(self._depth_latest.astype(np.float16))


def sensor_pose_w(base_pos, base_quat_wxyz):
    """Sensor world pose from the base pose and the authors' offset (pos and 180-deg x rotation)."""
    from plugins.jeplo.sensor import quat_wxyz_to_matrix

    R = quat_wxyz_to_matrix(base_quat_wxyz)
    pos = np.asarray(base_pos, dtype=float) + R @ np.asarray(OFFSET_POS, dtype=float)
    # q_sensor = q_base * q_offset
    w1, x1, y1, z1 = base_quat_wxyz
    w2, x2, y2, z2 = OFFSET_QUAT_WXYZ
    q = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    return pos, q


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args(argv)

    from common.cerberus_worlds.registry import get_backend

    get_backend("plain_usd").prepare_scene_env(args.scene)
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    try:
        return run(args)
    except Exception:  # noqa: BLE001 - SimulationApp.close() exits 0 and eats the traceback
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    finally:
        app.close()


def run(args) -> int:
    from isaaclab.sensors import RayCasterCfg

    from plugins.jeplo.pattern import RaySphericalSlicePatternCfg
    from policy.locomotion.eval import scene_env as se

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from scripts.explore_tour import load_grid, nearest_free

    grid = load_grid(args.scene, None)
    parts = [float(v) for v in args.start.split(",")]
    start = nearest_free(grid, parts[0], parts[1])
    yaw = parts[2] if len(parts) > 2 else 0.0

    lidar_cfg = RayCasterCfg(
        prim_path="/World/envs/env_0/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=OFFSET_POS, rot=OFFSET_QUAT_WXYZ),
        ray_alignment="base",
        pattern_cfg=RaySphericalSlicePatternCfg(),
        max_distance=2.0, mesh_prim_paths=["/World/scan_proxy"], update_period=0.1, debug_vis=False)
    env, _scene = se.build_loco_env(args.scene, [(start[0], start[1], yaw, grid.floor_z)],
                                    task=se.task_for_embodiment(args.embodiment), seed=args.seed,
                                    episode_length_s=max(120.0, args.max_total_time + 60.0), with_cameras=False,
                                    extra_sensors={"jeplo_lidar": lidar_cfg})
    os.environ["JEPLO_MODELS"] = str(Path(args.models))
    backend = JeploIsaacBackend(env, se)
    backend.debug = args.debug
    if args.record_dir:
        from common.cerberus_description.go2_arx5_spec import (
            GO2_ARX5_BASE_PRIM_REL_PATH,
        )

        base_prim = (f"/World/envs/env_0/Robot/{GO2_ARX5_BASE_PRIM_REL_PATH}"
                     if args.embodiment == "go2_arx5" else "/World/envs/env_0/Robot/base")
        backend.start_recording(args.record_dir, grid, base_prim)
    robot = env.unwrapped.scene["robot"]
    if args.settle_steps > 0:
        import torch

        zero = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        for _ in range(args.settle_steps):
            backend.policy.set_command(zero, dt=0.02)
            actions = backend.policy.act(backend.observation(zero))["actions"]
            env.step(torch.tensor([actions], device=env.unwrapped.device, dtype=torch.float32))
        print(f"[jeplo] settled for {args.settle_steps} zero-command steps", flush=True)

    def extra():
        data = robot.data
        return {
            "root_pos": [float(v) for v in data.root_pos_w[0].detach().cpu()],
            "root_quat_xyzw": [float(v) for v in data.root_quat_w[0].detach().cpu()],
            "joint_pos": [float(v) for v in data.joint_pos[0].detach().cpu()],
        }

    if args.goals_xy:
        goals = [tuple(float(v) for v in part.split(",")) for part in args.goals_xy.split(";") if part.strip()]
    else:
        goals = pick_tour_goals(grid, start, seed=args.seed, n=args.goals, min_total_m=args.min_net)
    print(f"[jeplo] start={start} goals={[(round(x, 2), round(y, 2)) for x, y in goals]}", flush=True)
    config = TourConfig(goal_tolerance_m=args.goal_tolerance, max_total_time_s=args.max_total_time)
    result = run_tour(backend, grid, goals, config=config,
                      pursuit=PurePursuitConfig(dt=config.dt, goal_tolerance_m=config.goal_tolerance_m),
                      extra=extra)
    extra_arrays = {
        "lidar_depth": np.asarray(backend._depth_rows, dtype=np.float16).reshape(-1, 25, 60),
        "lidar_depth_scale": np.array([2.0]),
    }
    payload = tour_npz_payload(grid, result, scene_id=args.scene, extra_arrays=extra_arrays)
    np.savez_compressed(out / "tour.npz", **payload)
    for writer in backend._writers.values():
        writer.close()
    summary = {
        "policy": "jeplo", "scene": args.scene, "seed": args.seed, "embodiment": args.embodiment,
        "start": [start[0], start[1], yaw], "goals": goals, "reached": result.reached,
        "final_distances_m": result.final_distances_m, "net_displacement_m": result.net_displacement_m,
        "path_length_m": result.path_length_m, "duration_s": result.duration_s, "fallen": result.fallen,
        "notes": result.notes, "steps": len(result.rows), "npz": str(out / "tour.npz"),
    }
    (out / "tour.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ok = result.success and result.net_displacement_m >= args.min_net
    print(("PASS " if ok else "FAIL ") + f"jeplo tour net={result.net_displacement_m:.2f} m")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
