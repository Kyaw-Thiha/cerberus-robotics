# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Isaac-Lab-side half of the sim-to-ROS2 bridge for a remote SLAM eval
# mission. Boots a trained locomotion checkpoint (rsl-rl 5.x .pt, or an
# exported policy .onnx) against one of slam/slam/eval/sim_env_cfg.py's
# SLAM-eval env cfgs (Go2 flat-terrain Play env + one of the locked sensor
# suites), optionally inside a GRScenes household scene (--scene, see
# slam/slam/eval/scene_env_cfg.py), drives it through
# slam/slam/eval/scripted_motion.py's canned schedule (the SAME schedule
# record_mission_bag.py's ScriptedMotionPublisher plays over `cmd_vel` --
# applied here directly to the velocity-command term instead, since this
# process has no rclpy: see slam/slam/ground_truth/sim_ipc.py's module
# docstring for why ground-truth/sensor data crosses to the separate rclpy
# process, ros2_bridge_relay.py, over a local zmq socket rather than living
# in one process), and each step publishes ground-truth pose + raw sensor
# data over that socket for ros2_bridge_relay.py to republish onto ROS2.
#
# Pod-script conventions (scripts/arena_spike_go2.py): a JSON report is
# written FIRST (and rewritten at the end), then os._exit(code) with the real
# exit code -- SimulationApp.close() exits 0 whatever happened and has hung at
# shutdown. --dry-run resolves scene/spawn/checkpoint without importing Isaac.
# Exit codes: 0 mission completed, 1 failure, 2 bad arguments.
#
# Verified against Isaac Lab release/3.0.0 @ dcabdcda SOURCE (not a live run):
# the command term attribute path command_manager.get_term("base_velocity")
# .vel_command_b (velocity_command.py); env.step_dt (manager_based_env.py);
# robot.data.root_pos_w/root_quat_w/root_lin_vel_w/root_ang_vel_w exist and are
# ProxyArray (root_*_vel_w are the COM-frame shorthands); Imu data has
# ang_vel_b/lin_acc_b only (no quat_w); camera.data.output is
# dict[str, ProxyArray] of shape (N, H, W, C) (rgb is a 3-channel view of
# rgba); RayCasterData.ray_hits_w is a ProxyArray resolving to (N, B, 3) with
# inf for misses, and MultiMeshRayCasterData inherits it unchanged;
# SensorBase._timestamp_last_update is now a warp array; RslRlVecEnvWrapper
# .step returns (obs TensorDict, rew, dones, extras).
#
# UNVERIFIED: (a) the whole flow -- nothing here has run on the new stack.
# UNVERIFIED: (b) the MultiMeshRayCaster builds a warp mesh from a GRScenes
#   house (instanced prims, mesh size, startup time) and its scan returns
#   plausible ranges once re-expressed in the lidar frame
#   (lidar_points_in_sensor_frame).
# UNVERIFIED: (c) the RGB cameras see a lit scene (scene_env_cfg.py's dome
#   light) and the scene colliders' own friction (the stock ground used 1.0)
#   does not break the frozen policy's gait.
# UNVERIFIED: (d) Camera.set_world_poses(convention="world") frames the
#   showcase camera along the robot's heading.
# UNVERIFIED: (e) the ONNX path (onnxruntime ships in the image's Isaac
#   python) and the rsl-rl 5.x make_runner/load path.
# UNVERIFIED: (f) the first env.step() after env.reset() reports no done, the
#   robot is at the spawn within tolerance, and physics_manager reports PhysX.
#
# Records a "collision" block in the report (contact always, LiDAR clearance for the rigs that have a LiDAR):
# the canned schedule is open loop, so a run that pushed into a wall or furniture would score drift that
# measures the collision. collision_monitor.py owns the thresholds and the valid_for_scoring verdict; the
# loop only feeds it per-step observations.
#
# Does NOT publish /showcase_camera/rgb over the ROS2 bridge (one of
# record_mission_bag.py's own CONTRACT_TOPICS -- still an open gap for that
# bag topic specifically). It DOES write a showcase-camera video directly to
# an .mp4 via imageio when --backend/--run-timestamp are both given (see
# _capture_showcase_frame() below and showcase_camera_video.py), same
# per-step compute_chase_cam_pose()-driven transform sensor_configs.py's
# SHOWCASE_CAMERA_CFG comment describes.

"""Publish ground-truth pose + raw sensor data for one SLAM eval mission."""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.cerberus_compat.isaac_data import as_tensor
from common.cerberus_compat.launcher import add_launcher_args
from common.cerberus_compat.quat import from_isaac, to_isaac
from policy.cli_common import check_gpu_driver_for_rendering
from slam.slam.eval.scripted_motion import (
    MotionSegment,
    canned_schedule,
    command_at,
)
from slam.slam.eval.showcase_camera_follow import compute_chase_cam_pose
from slam.slam.eval.showcase_camera_video import build_frame_writer_path
from slam.slam.eval.sim_run_args import (
    ResolvedScene,
    RunArgError,
    checkpoint_kind,
    report_path,
    resolve_scene,
    task_id_for_sensor_rig,
    write_report,
)
from slam.slam.ground_truth.collision_monitor import (
    CollisionMonitor,
    band_clearance_m,
    offending_bodies,
)
from slam.slam.ground_truth.sim_ipc import (
    IPC_ADDRESS,
    TOPIC_CAMERA_FRONT_DEPTH,
    TOPIC_CAMERA_FRONT_RGB,
    TOPIC_CAMERA_FRONT_RIGHT_RGB,
    TOPIC_CAMERA_REAR_DEPTH,
    TOPIC_CAMERA_REAR_RGB,
    TOPIC_CAMERA_REAR_RIGHT_RGB,
    TOPIC_IMU,
    TOPIC_LIDAR,
    TOPIC_POSE,
    ImagePayload,
    ImuPayload,
    PointCloudPayload,
    PosePayload,
    pack_image,
    pack_imu,
    pack_point_cloud,
    pack_pose,
)

SPAWN_XY_TOLERANCE_M = 0.25
SPAWN_Z_TOLERANCE_M = 0.5
"""How far the robot may be from the requested spawn after the first step before the run is aborted."""


@dataclass(frozen=True)
class RunPlan:
    """Everything resolvable without Isaac: what this run will do."""

    task_id: str
    schedule: list[MotionSegment]
    duration_s: float  # mission schedule + settle
    scene: ResolvedScene
    checkpoint: str
    checkpoint_kind: str
    report: Path

    def metadata(self, args: argparse.Namespace) -> dict:
        return {
            "mission": args.mission,
            "sensor_rig": args.sensor_rig,
            "task_id": self.task_id,
            "duration_s": self.duration_s,
            "backend": args.backend,
            "run_timestamp": args.run_timestamp,
            "checkpoint": self.checkpoint,
            "checkpoint_kind": self.checkpoint_kind,
            "report": str(self.report),
            **self.scene.to_metadata(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mission", type=str, required=True, help="Mission name, see slam/slam/eval/scripted_motion.py.")
    parser.add_argument("--sensor-rig", choices=["lidar_camera", "camera_only", "camera_stereo"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy: an rsl-rl 5.x .pt checkpoint or an exported .onnx (48-dim obs for the flat Go2); chosen by extension.")
    parser.add_argument("--scene", type=str, default="stock", help="'stock' (Isaac Lab's flat ground) or grscenes:<folder>, resolved through the plain_usd world backend.")
    parser.add_argument("--spawn-index", type=int, default=None, help="Which validated spawn pose of the scene's admission-manifest row to use (default 0). Scenes only.")
    parser.add_argument("--spawn", type=str, default=None, metavar="X,Y,YAW,FLOOR_Z", help="Explicit spawn in the scene frame (m, m, rad, m), for smoke tests before an admission manifest exists. Scenes only.")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Extra run time after the mission's own schedule ends, matching record_mission_bag.py's SETTLE_SECONDS.")
    parser.add_argument("--backend", type=str, default=None, help="SLAM backend name, only used to name the showcase-camera video's output path (slam/slam/eval/results/<backend>/<run-timestamp>/videos/) and this run's report. No video is written if omitted.")
    parser.add_argument("--run-timestamp", type=str, default=None, help="Must match record_mission_bag.py's own --run-timestamp for this run, so the video lands in the same results dir as its bag. No video is written if omitted.")
    parser.add_argument("--report", type=str, default=None, help="JSON report path (default: sim_loop_report.json in the run's results dir, else data/reports/slam_sim_loop.json).")
    parser.add_argument("--watchdog_s", type=float, default=900.0, help="Dump all thread stacks every N s if hung (0 disables).")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Print the resolved run plan and exit; imports no Isaac.")
    return parser


def prepare_run(args: argparse.Namespace) -> RunPlan:
    """Pure-python argument resolution; raises RunArgError. Sets MDL_SYSTEM_PATH for a scene (before the app launches)."""
    try:
        schedule = canned_schedule(args.mission)
    except ValueError as e:
        raise RunArgError(str(e)) from e
    kind = checkpoint_kind(args.checkpoint)
    if not Path(args.checkpoint).is_file():
        raise RunArgError(f"--checkpoint {args.checkpoint!r} is not a file")
    if args.settle_seconds < 0:
        raise RunArgError("--settle-seconds must be >= 0")
    scene = resolve_scene(args.scene, args.spawn_index, args.spawn)
    return RunPlan(
        task_id=task_id_for_sensor_rig(args.sensor_rig),
        schedule=schedule,
        duration_s=sum(segment.duration_seconds for segment in schedule) + args.settle_seconds,
        scene=scene,
        checkpoint=args.checkpoint,
        checkpoint_kind=kind,
        report=report_path(args.backend, args.run_timestamp, args.report),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, hydra_args = parser.parse_known_args(argv)
    if not args.dry_run:
        # needs isaaclab; a Lab 3.0 AppLauncher registers neither --headless nor --enable_cameras, so
        # add_launcher_args adds the compat --headless and the camera flag is set below.
        add_launcher_args(parser)
        args, hydra_args = parser.parse_known_args(argv)
    try:
        plan = prepare_run(args)
    except (RunArgError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    print(f"[sim_ground_truth_loop] run plan:\n{json.dumps(plan.metadata(args), indent=2, sort_keys=True, default=str)}", flush=True)
    if args.dry_run:
        return 0

    # Hydra (inside hydra_task_config) parses sys.argv itself and chokes on our own --mission/--sensor-rig/
    # --checkpoint args if they stay in it -- same reset push_recovery_eval.py does.
    sys.argv = [sys.argv[0]] + hydra_args
    check_gpu_driver_for_rendering()  # every rig renders cameras; driver 595.x crashes RTX
    # Not registered by a Lab 3.0 launcher, but it reads the key from the namespace (isaaclab/app/sim_launcher.py);
    # the deprecated AppLauncher adapter passes no cfg to scan, so cameras are not auto-enabled.
    args.enable_cameras = True
    from isaaclab.app import AppLauncher

    AppLauncher(args)
    return _run_in_sim(args, plan)


def _run_in_sim(args: argparse.Namespace, plan: RunPlan) -> int:
    """Runs the mission and never returns normally: report first, then os._exit with the real exit code."""
    report = {
        "passed": False,
        "failure_reasons": ["incomplete (still running or killed)"],
        "metadata": plan.metadata(args),
        "steps_completed": 0,
    }
    exit_code = 1
    try:
        if args.watchdog_s > 0:
            faulthandler.enable()
            faulthandler.dump_traceback_later(args.watchdog_s, repeat=True, exit=False)
        write_report(plan.report, report)
        try:
            _simulate(args, plan, report)
            report["passed"] = True
            report["failure_reasons"] = []
        except BaseException as e:  # noqa: BLE001 -- includes SystemExit/KeyboardInterrupt
            traceback.print_exc()
            report["failure_reasons"] = [f"{type(e).__name__}: {e}"]
            report["error"] = traceback.format_exc()
        if not write_report(plan.report, report):  # report FIRST, before any teardown
            report["passed"] = False
        exit_code = 0 if report["passed"] else 1
        print(f"[sim_ground_truth_loop] {'PASS' if report['passed'] else 'FAIL'}: {report['failure_reasons']} (report: {plan.report})", flush=True)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001, S110 -- nothing sensible to do if flushing fails while exiting
            pass
        # never simulation_app.close(): it exits 0 whatever happened and has hung >10 min at shutdown
        os._exit(exit_code)
    return exit_code  # unreachable


def _simulate(args: argparse.Namespace, plan: RunPlan, report: dict) -> None:
    from common.cerberus_compat.isaaclab_tasks import install_legacy_task_path_aliases
    from common.cerberus_compat.rsl_rl_compat import hydra_task_config

    install_legacy_task_path_aliases()  # the Cerberus Go2 cfg imports Lab 2.x task paths
    from slam.slam.eval.sim_env_cfg import register_slam_eval_tasks

    register_slam_eval_tasks()

    @hydra_task_config(plan.task_id, "rsl_rl_cfg_entry_point")
    def hydra_main(env_cfg, agent_cfg) -> None:
        _run_mission(args, plan, report, env_cfg, agent_cfg)

    hydra_main()


def _run_mission(args: argparse.Namespace, plan: RunPlan, report: dict, env_cfg, agent_cfg) -> None:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from slam.slam.eval.scene_env_cfg import apply_mission, apply_scene, physics_summary

    apply_mission(env_cfg, plan.duration_s)
    if plan.scene.scene is not None:
        apply_scene(env_cfg, plan.scene.scene, plan.scene.spawn)
    metadata = report["metadata"]
    metadata.update(physics_summary(env_cfg))  # raises unless PhysX + legacy actuators

    env = gym.make(plan.task_id, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    dt = raw_env.step_dt
    total_steps = round(plan.duration_s / dt)
    metadata.update(
        physics_manager=getattr(raw_env.sim.physics_manager, "__name__", str(raw_env.sim.physics_manager)),
        step_dt=dt,
        total_steps=total_steps,
        episode_length_s=env_cfg.episode_length_s,
    )
    policy = _load_policy(plan.checkpoint_kind, plan.checkpoint, env, agent_cfg)
    print(f"[sim_ground_truth_loop] run metadata:\n{json.dumps(metadata, indent=2, sort_keys=True, default=str)}", flush=True)

    # Same technique push_recovery_eval.py's apply_command_floor() uses to
    # drive the velocity command directly, bypassing the real cmd_vel
    # ROS2 topic (this process has no rclpy -- see module docstring).
    command_term = raw_env.command_manager.get_term("base_velocity")

    # Collision/clearance bookkeeping (collision_monitor.py). Written into the report before the first step and
    # every step after, so a report from a crashed or killed run still carries the verdict, and read by
    # smoke_check.py / run_slam_eval_remote.sh before any scoring happens.
    floor_z = plan.scene.spawn.floor_z if plan.scene.spawn is not None else 0.0
    monitor = CollisionMonitor()
    report["collision"] = monitor.summary()

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 2000)  # let the last messages out before os._exit
    socket.bind(IPC_ADDRESS)
    # Give ros2_bridge_relay.py's SUB socket time to connect -- zmq PUB
    # drops messages sent before a SUB has connected ("slow joiner"), same
    # concern record_mission_bag.py's own STARTUP_GRACE_SECONDS handles for
    # `ros2 bag record`.
    time.sleep(2.0)

    # Showcase-camera video -- opt-in on --backend/--run-timestamp both
    # being given (matches run_slam_eval_remote.sh's usage; a bare smoke
    # test invocation without them just skips video, same as before this
    # was added). Frames written directly via imageio, same convention
    # policy/locomotion/core/video_capture.py's record_condition_clips()
    # uses -- see showcase_camera_video.py's module docstring for why (no
    # RecordVideo wrapper fits a per-step-driven camera transform).
    video_writer = None
    if args.backend and args.run_timestamp:
        import imageio

        video_path = build_frame_writer_path(args.backend, args.mission, args.run_timestamp)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=round(1 / dt))

    try:
        # applies the deterministic reset events; the commands are resampled by the reset, so the very first
        # policy call (20 ms) sees one stale random command before the schedule's own takes over
        obs, _ = env.reset()
        max_tilt_deg = 0.0
        with torch.inference_mode():
            for step in range(total_steps):
                elapsed = step * dt
                vx, vy, omega_z = command_at(plan.schedule, elapsed)
                command_term.vel_command_b[:, 0] = vx
                command_term.vel_command_b[:, 1] = vy
                command_term.vel_command_b[:, 2] = omega_z

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if bool(dones.any()):
                    # the env auto-resets (pose, joints), which would silently splice two trajectories into the bag
                    raise RuntimeError(f"env terminated at step {step} (t={elapsed:.2f}s): the robot fell or its base touched something")
                if step == 0 and plan.scene.spawn is not None:
                    _check_spawn(as_tensor(robot.data.root_pos_w)[0].tolist(), plan.scene.spawn, env_cfg)

                sim_time_ns = round(elapsed * 1e9)
                _publish_pose(socket, sim_time_ns, robot)
                _publish_sensors(socket, sim_time_ns, raw_env)

                base_xy = as_tensor(robot.data.root_pos_w)[0, :2].cpu().numpy()
                monitor.update(elapsed, _contact_offenders(raw_env), _lidar_clearance(raw_env, base_xy, floor_z))
                report["collision"] = monitor.summary()

                if video_writer is not None:
                    _capture_showcase_frame(video_writer, raw_env, robot)
                report["steps_completed"] = step + 1
                max_tilt_deg = max(max_tilt_deg, tilt_deg_from_quat_wxyz(from_isaac(as_tensor(robot.data.root_quat_w)[0]).tolist()))
        root_pos_w = as_tensor(robot.data.root_pos_w)[0].tolist()
        report["final_state"] = {
            "root_pos_w": root_pos_w,
            "base_height_above_floor_m": root_pos_w[2] - floor_z,
            "tilt_deg": tilt_deg_from_quat_wxyz(from_isaac(as_tensor(robot.data.root_quat_w)[0]).tolist()),
            "max_tilt_deg": max_tilt_deg,
        }
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[sim_ground_truth_loop] wrote showcase video to {video_path}")
        socket.close()
        context.term()


def _contact_offenders(raw_env) -> list[str] | None:
    """Names of the robot bodies touching the scene, or None when the env has no usable contact sensor.

    The stock Go2 task's ``contact_forces`` sensor is unfiltered and covers every robot body, so the floor shows
    up in it; ``collision_monitor.IGNORED_BODY_SUFFIXES`` drops the feet. Returns None (never ``[]``) when the
    sensor or its body list is unavailable, so "cannot tell" is never recorded as "no collision".
    """
    scene = raw_env.scene
    if "contact_forces" not in scene.sensors:
        return None
    import torch

    sensor = scene["contact_forces"]
    body_names = [str(name) for name in (getattr(sensor, "body_names", None) or ())]
    forces = as_tensor(sensor.data.net_forces_w)
    if forces is None or len(forces) == 0:  # ContactSensorData.net_forces_w is None before its first update
        return None
    force_norms = torch.linalg.vector_norm(forces[0].float(), dim=-1).cpu().numpy()
    if len(body_names) != len(force_norms):
        return None  # cannot attribute the forces to bodies; report "unknown" rather than guess
    return offending_bodies(body_names, force_norms)


def _lidar_clearance(raw_env, base_xy: np.ndarray, floor_z: float) -> float | None:
    """Horizontal distance from the base to the nearest scene geometry inside the robot's height band.

    Computed only when the LiDAR buffer was genuinely recomputed -- its real cadence is 10 Hz against a 50 Hz
    control step -- reusing the same SensorBase staleness bookkeeping ``_publish_sensors()`` relies on (and
    accessing ``.data`` every call for that bookkeeping to run at all). None when the rig has no LiDAR.
    """
    scene = raw_env.scene
    if "lidar" not in scene.sensors:
        return None
    lidar = scene["lidar"]
    lidar_data = lidar.data
    update_time = float(lidar._timestamp_last_update.numpy()[0])
    if update_time == _lidar_clearance._last_update_time:
        return None
    _lidar_clearance._last_update_time = update_time
    hits_w = as_tensor(lidar_data.ray_hits_w)[0].cpu().numpy()
    return band_clearance_m(hits_w, base_xy, floor_z)


_lidar_clearance._last_update_time = None  # function-attribute state, same pattern as _publish_sensors'


def _check_spawn(root_pos_w: list[float], spawn, env_cfg) -> None:
    """The robot must be where the validated spawn says (catches a wrong env origin or a scene offset)."""
    expected_z = env_cfg.scene.robot.init_state.pos[2]  # floor_z + standing height, set by apply_scene
    dx, dy, dz = root_pos_w[0] - spawn.x, root_pos_w[1] - spawn.y, root_pos_w[2] - expected_z
    if math.hypot(dx, dy) > SPAWN_XY_TOLERANCE_M or abs(dz) > SPAWN_Z_TOLERANCE_M:
        raise RuntimeError(
            f"robot is at {root_pos_w} after the first step, expected about ({spawn.x}, {spawn.y}, {expected_z}) "
            "(validated spawn): wrong env origin or scene offset, or it was dropped through the floor"
        )


def _load_policy(kind: str, path: str, env, agent_cfg):
    """obs TensorDict -> actions tensor, from an rsl-rl runner checkpoint or an exported ONNX policy."""
    if kind == "onnx":
        return _onnx_policy(path, env.unwrapped.device)
    from common.cerberus_compat.rsl_rl_compat import (
        check_checkpoint_format,
        make_runner,
    )

    # clear error if this is a rsl-rl 3.x (model_state_dict) file on rsl-rl 5.x: use the exported .onnx instead
    check_checkpoint_format(path)
    runner = make_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(path)
    return runner.get_inference_policy(device=env.unwrapped.device)


def _onnx_policy(path: str, device):
    """The frozen old-stack policies exist as ONNX (observation normalizer baked in), loaded with onnxruntime on the CPU provider."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    obs_dim = inp.shape[-1]
    fixed_batch_one = isinstance(inp.shape[0], int) and inp.shape[0] == 1

    def policy(obs):
        x = obs["policy"].detach().cpu().numpy().astype(np.float32)
        if x.shape[-1] != obs_dim:
            raise RuntimeError(f"ONNX policy expects {obs_dim}-dim observations, the env's policy group is {x.shape[-1]}-dim")
        if fixed_batch_one:
            out = np.concatenate([session.run(None, {inp.name: row[None]})[0] for row in x])
        else:
            out = session.run(None, {inp.name: x})[0]
        return torch.from_numpy(out).to(device)

    return policy


def _capture_showcase_frame(video_writer, raw_env, robot) -> None:
    # NOT independently verified against a live Isaac Lab install -- same
    # pod-side-only caveat this module's header already carries. Isaac
    # Lab's Camera sensor exposes set_world_poses(positions, orientations,
    # env_ids=None, convention=...) to move a camera prim explicitly every
    # step, matching sensor_configs.py's SHOWCASE_CAMERA_CFG comment
    # ("this prim's transform must be set explicitly every step from
    # compute_chase_cam_pose()'s output instead of relying on a fixed
    # offset").
    import torch

    root_pos_w = as_tensor(robot.data.root_pos_w)
    yaw = _yaw_from_quat_wxyz(from_isaac(as_tensor(robot.data.root_quat_w)[0]).tolist())
    position, orientation_xyzw = compute_chase_cam_pose(tuple(root_pos_w[0].tolist()), yaw)
    camera = raw_env.scene["showcase_camera"]
    device = root_pos_w.device
    camera.set_world_poses(
        positions=torch.tensor([position], device=device),
        # to_isaac: Cerberus w-first -> Isaac native order (wxyz on Lab 2.x, xyzw on 3.x).
        # Verified at Lab release/3.0.0 (isaaclab/sensors/camera/camera.py set_world_poses
        # docstring: "orientations: quaternion in (x, y, z, w)"). `convention` is the camera-AXIS
        # convention the quaternion is given in: compute_chase_cam_pose() returns a plain yaw about
        # +z, i.e. "world" (forward +X, up +Z); "ros" would point the camera along the world +z axis.
        orientations=torch.tensor([to_isaac((orientation_xyzw[3], orientation_xyzw[0], orientation_xyzw[1], orientation_xyzw[2]))], device=device),
        convention="world",
    )
    frame = as_tensor(camera.data.output["rgb"])[0, ..., :3].contiguous().to(torch.uint8).cpu().numpy()
    video_writer.append_data(frame)


def tilt_deg_from_quat_wxyz(quat_wxyz: list[float]) -> float:
    """Angle between the body z axis and world z (0 = upright), from a (w, x, y, z) quaternion."""
    _, x, y, _ = quat_wxyz
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _yaw_from_quat_wxyz(quat_wxyz: list[float]) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _publish_pose(socket, sim_time_ns: int, robot) -> None:
    pos = as_tensor(robot.data.root_pos_w)[0].tolist()
    quat = from_isaac(as_tensor(robot.data.root_quat_w)[0]).tolist()  # Cerberus internal (w, x, y, z)
    lin_vel = as_tensor(robot.data.root_lin_vel_w)[0].tolist()
    ang_vel = as_tensor(robot.data.root_ang_vel_w)[0].tolist()
    payload = PosePayload(
        sim_time_ns=sim_time_ns,
        position=tuple(pos),
        orientation_wxyz=tuple(quat),
        linear_velocity=tuple(lin_vel),
        angular_velocity=tuple(ang_vel),
    )
    socket.send_multipart([TOPIC_POSE, pack_pose(payload)])


def _publish_sensors(socket, sim_time_ns: int, raw_env) -> None:
    scene = raw_env.scene
    if "lidar" in scene.sensors:
        lidar = scene["lidar"]
        # Publish only when the lidar's OWN buffer was recomputed this step. env.step() runs at 50 Hz (physics dt
        # 0.005 * decimation 4) but LIDAR_CFG updates at 10 Hz; publishing every step declared a fresh 100 ms scan
        # (sensor_msg_builders._SCAN_PERIOD_US) 5x too often, so each scan's declared end ran ahead of the real IMU
        # data and FAST-LIO2 logged "Not enough IMU data" forever (confirmed on a pod).
        # Gate on SensorBase's `_timestamp_last_update` (bumped in `_update_outdated_buffers()` only once
        # update_period has really elapsed) instead of a hand-rolled step counter; RayCaster has no public accessor
        # for it. `.data` must still be read every step so that lazy staleness update runs at all.
        lidar_data = lidar.data  # accessing .data also runs SensorBase's lazy staleness update
        # a warp array on Lab 3.0 (SensorBase._timestamp_last_update), not a tensor
        # UNVERIFIED: .numpy() on the device-resident warp array (source says wp.zeros_like(...))
        lidar_update_time = float(lidar._timestamp_last_update.numpy()[0])
        if lidar_update_time != _publish_sensors._last_lidar_update_time:
            _publish_sensors._last_lidar_update_time = lidar_update_time
            _publish_lidar(socket, sim_time_ns, lidar, lidar_data)

    if "imu" in scene.sensors:
        imu = scene["imu"].data
        # Verified at Lab release/3.0.0 (isaaclab_physx/sensors/imu/imu_data.py,
        # isaaclab/sensors/imu/base_imu_data.py): ang_vel_b / lin_acc_b are
        # ProxyArray (as_tensor unwraps), but ImuData has NO quat_w (it did on
        # v2.3.2). IMU_CFG is mounted on Robot/base with identity offset, so the
        # base root orientation is the IMU orientation: robot.data.root_quat_w
        # (ProxyArray, xyzw on 3.x).
        quat = from_isaac(as_tensor(scene["robot"].data.root_quat_w)[0]).tolist()  # Cerberus internal (w, x, y, z)
        ang_vel = as_tensor(imu.ang_vel_b)[0].tolist()
        lin_acc = as_tensor(imu.lin_acc_b)[0].tolist()
        payload = ImuPayload(
            sim_time_ns=sim_time_ns,
            orientation_wxyz=tuple(quat),
            angular_velocity=tuple(ang_vel),
            linear_acceleration=tuple(lin_acc),
        )
        # IMU is published every env.step() UNCHANGED (no gating) -- IMU_CFG's
        # own real update_period=1/200 (0.005s) is faster than env.step_dt
        # (0.02s), so real IMU data genuinely refreshes on every single
        # env.step already; this is also exactly what lets real IMU data race
        # ahead of and past each (now correctly ~10Hz) lidar scan's declared
        # end time before FAST-LIO2 needs it, instead of perpetually trailing
        # it as the bug this function used to have.
        socket.send_multipart([TOPIC_IMU, pack_imu(payload)])

    _publish_camera_if_present(socket, scene, "front_camera", TOPIC_CAMERA_FRONT_RGB, TOPIC_CAMERA_FRONT_DEPTH, sim_time_ns)
    _publish_camera_if_present(socket, scene, "rear_camera", TOPIC_CAMERA_REAR_RGB, TOPIC_CAMERA_REAR_DEPTH, sim_time_ns)
    # RIGHT-eye cameras: only present on CAMERA_STEREO_SUITE (multicam_slam's
    # rig) -- rgb-only, no depth_topic (data_types=["rgb"] only, see
    # sensor_configs.py's FRONT_CAMERA_RIGHT_CFG/REAR_CAMERA_RIGHT_CFG), so
    # depth_topic=None here and _publish_camera_if_present skips depth
    # publishing whenever the sensor output has no "depth" key regardless.
    _publish_camera_if_present(socket, scene, "front_camera_right", TOPIC_CAMERA_FRONT_RIGHT_RGB, None, sim_time_ns)
    _publish_camera_if_present(socket, scene, "rear_camera_right", TOPIC_CAMERA_REAR_RIGHT_RGB, None, sim_time_ns)


# Function-attribute state (persists across this module's one long-lived sim
# process, single env instance -- see apply_mission()'s num_envs = 1)
# tracking the lidar sensor's own last real update timestamp, so
# _publish_sensors() above can tell a genuinely-new scan apart from a
# repeated call against unchanged cached data. None is never equal to a real
# float timestamp, so the very first call always publishes.
_publish_sensors._last_lidar_update_time = None


def lidar_points_in_sensor_frame(hits_w: np.ndarray, sensor_pos_w: np.ndarray, sensor_yaw: float):
    """Finite ray hits in the lidar frame, plus the mask of rays that hit.

    ``RayCasterData.ray_hits_w`` is in the WORLD frame with ``inf`` for misses, but the cloud goes out as
    frame_id "lidar" (is_dense, FAST-LIO2 reads x/y/z as sensor-frame). LIDAR_CFG uses ``ray_alignment="yaw"``,
    so the scan frame is the sensor position rotated by the base yaw only (no roll/pitch).
    """
    keep = np.isfinite(hits_w).all(axis=-1)
    delta = hits_w[keep] - sensor_pos_w
    cos_yaw, sin_yaw = math.cos(sensor_yaw), math.sin(sensor_yaw)
    local = np.empty_like(delta)
    local[:, 0] = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local[:, 1] = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    local[:, 2] = delta[:, 2]
    return local.astype(np.float32), keep


def _publish_lidar(socket, sim_time_ns: int, lidar, lidar_data) -> None:
    """Builds and publishes one lidar scan -- split out of _publish_sensors()
    so the latter can gate the call on the sensor's own real update cadence
    (see _publish_sensors()'s own comment for why)."""
    hits_w = as_tensor(lidar_data.ray_hits_w)[0].cpu().numpy()
    sensor_pos_w = as_tensor(lidar_data.pos_w)[0].cpu().numpy()
    yaw = _yaw_from_quat_wxyz(from_isaac(as_tensor(lidar_data.quat_w)[0]).tolist())
    points, keep = lidar_points_in_sensor_frame(hits_w, sensor_pos_w, yaw)
    # ring = per-point scan-line index (0..channels-1), required by
    # FAST-LIO2-family backends' point-cloud parser -- see
    # sensor_msg_builders.py's _POINT_STEP comment. Derived from
    # LidarPatternCfg's own channel-major point ordering (channel varies
    # slowest, azimuth fastest -- confirmed against
    # isaaclab.sensors.ray_caster.patterns.patterns.lidar_pattern()'s
    # `torch.meshgrid(vertical_angles, horizontal_angles,
    # indexing="ij").reshape(-1, 3)` on a real pod, not guessed, and again against the
    # Lab 3.0 source), read
    # from the sensor's own cfg rather than hardcoded so this tracks
    # LIDAR_CFG if its channel count ever changes. Computed over ALL rays,
    # then filtered with the same mask as the points.
    num_points = hits_w.shape[0]
    channels = lidar.cfg.pattern_cfg.channels
    azimuth_count = num_points // channels
    ring = (np.arange(num_points, dtype=np.int64) // azimuth_count).astype("<u2")[keep]
    socket.send_multipart([TOPIC_LIDAR, pack_point_cloud(PointCloudPayload(sim_time_ns, points.tobytes(), ring.tobytes()))])


def _publish_camera_if_present(socket, scene, sensor_name: str, rgb_topic: bytes, depth_topic: bytes | None, sim_time_ns: int) -> None:
    if sensor_name not in scene.sensors:
        return
    import torch

    camera = scene[sensor_name]
    output = camera.data.output  # dict[str, ProxyArray]; unwrapped per-key below
    if "rgb" in output:
        rgb = as_tensor(output["rgb"])[0, ..., :3].contiguous().to(torch.uint8).cpu().numpy()
        height, width = rgb.shape[0], rgb.shape[1]
        socket.send_multipart([
            rgb_topic,
            pack_image(ImagePayload(sim_time_ns, height, width, "rgb8", rgb.tobytes())),
        ])
    if "depth" in output:
        depth = as_tensor(output["depth"])[0, ..., 0].contiguous().to(torch.float32).cpu().numpy()
        height, width = depth.shape[0], depth.shape[1]
        socket.send_multipart([
            depth_topic,
            pack_image(ImagePayload(sim_time_ns, height, width, "32FC1", depth.tobytes())),
        ])


if __name__ == "__main__":
    sys.exit(main())
